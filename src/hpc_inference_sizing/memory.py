"""Memory calculator: predict, before renting anything, how many tokens fit in the KV cache.

For a model, a GPU, a number of GPUs and a named split, it computes what each GPU holds
(weights by component, runtime overhead, KV cache) and turns the free memory into cached tokens
and concurrent requests. The prediction is checked against the KV-cache size vLLM reports at
start-up (hypothesis H6 in docs/PROTOCOL.md).

Each split type has one rule (see `_weights_on_gpu` and `_kv_layers_on_gpu`):

    split    weights on each GPU                        KV cache on each GPU
    single   everything (1 GPU)                         all layers
    tp       every layer cut in n                       all layers, copied (MLA latent)
    ep       attention cut in n, experts spread         all layers, copied
    pp       n_layers / n consecutive layers            its own layers only
    dp_ep    attention copied, experts spread           all layers, its own requests only
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

GIB = 1024**3
MIB = 1024**2

SPLITS = ("single", "tp", "ep", "pp", "dp_ep")


@dataclass(frozen=True)
class ModelSpec:
    """Component sizes of a checkpoint, as measured by scripts/measure_checkpoint.py."""

    name: str
    n_layers: int
    kv_bytes_per_token_per_layer: int
    layered_bytes: dict[str, int]  # totals over all layers
    global_bytes: dict[str, int]  # components that appear once

    @property
    def total_bytes(self) -> int:
        return sum(self.layered_bytes.values()) + sum(self.global_bytes.values())


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_bytes: int  # total memory as nvidia-smi reports it


# Total memory reported by nvidia-smi (MiB). A40 and RTX A6000: Vast.ai offer listings,
# 2026-10-27. L40S and H100 PCIe: usual nvidia-smi values, to confirm on the first run.
GPUS = {
    "a40": GPUSpec("NVIDIA A40", 46068 * MIB),
    "a6000": GPUSpec("NVIDIA RTX A6000", 49140 * MIB),
    "l40s": GPUSpec("NVIDIA L40S", 46068 * MIB),
    "h100-pcie": GPUSpec("NVIDIA H100 PCIe", 81559 * MIB),
}

MODELS = ("mistral-small-4-nvfp4", "mistral-small-4-fp8")


def load_model(name: str) -> ModelSpec:
    raw = json.loads(resources.files(__package__).joinpath(f"data/{name}.json").read_text())
    return ModelSpec(
        name=raw["name"],
        n_layers=raw["n_layers"],
        kv_bytes_per_token_per_layer=raw["kv_bytes_per_token_per_layer"],
        layered_bytes=raw["layered_bytes"],
        global_bytes=raw["global_bytes"],
    )


@dataclass(frozen=True)
class GPUPlan:
    index: int
    weights_bytes: dict[str, int]  # by component
    overhead_bytes: int
    budget_bytes: int  # memory_bytes * gpu_memory_utilization
    kv_bytes_per_token: int
    free_bytes: int  # budget - weights - overhead (negative: does not fit)

    @property
    def weights_total(self) -> int:
        return sum(self.weights_bytes.values())

    @property
    def fits(self) -> bool:
        return self.free_bytes > 0

    @property
    def cache_tokens(self) -> int:
        return max(self.free_bytes, 0) // self.kv_bytes_per_token


@dataclass(frozen=True)
class MemoryPlan:
    model: str
    gpu: str
    n_gpus: int
    split: str
    request_tokens: int
    gpus: list[GPUPlan]
    cache_tokens: int  # tokens the whole unit can keep in cache
    concurrent_requests: int

    @property
    def fits(self) -> bool:
        return all(g.fits for g in self.gpus)


def _layers_per_stage(n_layers: int, n_gpus: int) -> list[int]:
    """Consecutive layers given to each pipeline stage; the first stages take the remainder."""
    base, extra = divmod(n_layers, n_gpus)
    return [base + (1 if i < extra else 0) for i in range(n_gpus)]


def _weights_on_gpu(model: ModelSpec, split: str, n_gpus: int, i: int) -> dict[str, int]:
    L, G = model.layered_bytes, model.global_bytes
    replicated = {"vision": G.get("vision", 0), "other": G.get("other", 0)}
    if split == "single":
        return {**L, **G}
    if split == "pp":
        share = _layers_per_stage(model.n_layers, n_gpus)[i] / model.n_layers
        w = {k: round(v * share) for k, v in L.items()}
        if i == 0:  # first stage: embedding and vision encoder
            w["embedding"] = G.get("embedding", 0)
            w["vision"] = G.get("vision", 0)
        if i == n_gpus - 1:  # last stage: output head and final norm
            w["lm_head"] = G.get("lm_head", 0)
            w["other"] = G.get("other", 0)
        return w
    if split not in ("tp", "ep", "dp_ep"):
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    # Attention, shared expert, embedding and output head: cut under tp/ep, copied under dp_ep.
    divisor = 1 if split == "dp_ep" else n_gpus
    dense = {
        "attention": L["attention"] // divisor,
        "shared_expert": L["shared_expert"] // divisor,
        "embedding": G.get("embedding", 0) // divisor,
        "lm_head": G.get("lm_head", 0) // divisor,
    }
    return {
        **dense,
        "routed_experts": L["routed_experts"] // n_gpus,  # cut (tp) or spread (ep, dp_ep)
        "router": L["router"],  # small, copied
        "layer_norms": L["layer_norms"],  # small, copied
        **replicated,
    }


def _kv_layers_on_gpu(model: ModelSpec, split: str, n_gpus: int, i: int) -> int:
    if split == "pp":
        return _layers_per_stage(model.n_layers, n_gpus)[i]
    return model.n_layers  # MLA latent is not split by heads: every GPU keeps all layers


def plan(
    model: ModelSpec,
    gpu: GPUSpec,
    n_gpus: int,
    split: str,
    *,
    request_tokens: int = 6_400,
    gpu_memory_utilization: float = 0.90,
    overhead_bytes: int = 2 * GIB,
) -> MemoryPlan:
    """Compute what each GPU holds and how many tokens and requests the unit can keep in cache."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if n_gpus < 1:
        raise ValueError("n_gpus must be at least 1")
    if (split == "single") != (n_gpus == 1):
        raise ValueError("split 'single' means exactly one GPU, and one GPU means split 'single'")
    if split == "pp" and n_gpus > model.n_layers:
        raise ValueError("more pipeline stages than layers")

    budget = int(gpu.memory_bytes * gpu_memory_utilization)
    gpus = []
    for i in range(n_gpus):
        weights = _weights_on_gpu(model, split, n_gpus, i)
        kv = _kv_layers_on_gpu(model, split, n_gpus, i) * model.kv_bytes_per_token_per_layer
        free = budget - sum(weights.values()) - overhead_bytes
        gpus.append(GPUPlan(i, weights, overhead_bytes, budget, kv, free))

    if split == "dp_ep":
        # Each GPU caches only its own requests: capacities add up, a request stays on one GPU.
        cache_tokens = sum(g.cache_tokens for g in gpus)
        requests = sum(g.cache_tokens // request_tokens for g in gpus)
    else:
        # Every GPU holds (a slice of) every cached token: the fullest GPU sets the limit.
        cache_tokens = min(g.cache_tokens for g in gpus)
        requests = cache_tokens // request_tokens
    if not all(g.fits for g in gpus):
        cache_tokens, requests = 0, 0

    return MemoryPlan(
        model=model.name,
        gpu=gpu.name,
        n_gpus=n_gpus,
        split=split,
        request_tokens=request_tokens,
        gpus=gpus,
        cache_tokens=cache_tokens,
        concurrent_requests=requests,
    )


def format_plan(p: MemoryPlan) -> str:
    """Human-readable table of a plan, one column per GPU."""
    gb = 1e9
    components = sorted({k for g in p.gpus for k, v in g.weights_bytes.items() if v})
    width = 16
    head = f"{p.model} on {p.n_gpus} x {p.gpu}, split {p.split}"
    lines = [head, "-" * len(head)]
    lines.append("".ljust(24) + "".join(f"GPU {g.index}".rjust(width) for g in p.gpus))
    for c in components:
        cells = "".join(f"{g.weights_bytes.get(c, 0) / gb:.2f} GB".rjust(width) for g in p.gpus)
        lines.append(f"  {c}".ljust(24) + cells)
    rows = [
        ("weights", lambda g: f"{g.weights_total / gb:.2f} GB"),
        ("overhead (assumed)", lambda g: f"{g.overhead_bytes / gb:.2f} GB"),  # 2 GiB = 2.15 GB
        ("budget (x util.)", lambda g: f"{g.budget_bytes / gb:.2f} GB"),
        ("free for KV cache", lambda g: f"{g.free_bytes / gb:.2f} GB"),
        ("KV per token", lambda g: f"{g.kv_bytes_per_token / 1e3:.1f} KB"),
        ("tokens in cache", lambda g: f"{g.cache_tokens:,}"),
    ]
    for label, fmt in rows:
        lines.append(label.ljust(24) + "".join(fmt(g).rjust(width) for g in p.gpus))
    lines.append("")
    if not p.fits:
        lines.append("DOES NOT FIT: the weights exceed the memory budget of at least one GPU.")
    else:
        lines.append(f"unit cache:          {p.cache_tokens:,} tokens")
        lines.append(f"concurrent requests: {p.concurrent_requests} of {p.request_tokens:,} tokens")
    return "\n".join(lines)
