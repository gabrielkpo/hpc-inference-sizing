"""Measure the size of each component of a Hugging Face checkpoint from its safetensors headers.

Only the headers are downloaded (a few kilobytes per shard), never the weights. The result is
the model data file read by the memory calculator.

Usage:
    python scripts/measure_checkpoint.py mistralai/Mistral-Small-4-119B-2603-NVFP4 \
        src/hpc_inference_sizing/data/mistral-small-4-nvfp4.json --name mistral-small-4-nvfp4
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request
from collections import Counter

HF = "https://huggingface.co"
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I32": 4,
    "F8_E8M0": 1,
    "BOOL": 1,
}

# Mistral-format tensor names. Components repeated in every layer are the ones a pipeline split
# divides between GPUs; global components appear once.
LAYERED = ("attention", "routed_experts", "shared_expert", "router", "layer_norms")


def classify(name: str) -> str:
    if re.search(r"\.experts\.\d+\.", name):
        return "routed_experts"
    if "shared_expert" in name:
        return "shared_expert"
    if name.startswith("layers."):
        if ".attention." in name:
            return "attention"
        if name.endswith("gate.weight"):
            return "router"
        if "norm" in name:
            return "layer_norms"
    if "tok_embeddings" in name:
        return "embedding"
    if name.startswith("output."):
        return "lm_head"
    if "vision" in name or "patch" in name or name.startswith("pre_mm_projector"):
        return "vision"
    return "other"


def read_header(url: str) -> dict:
    head = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    length = struct.unpack("<Q", urllib.request.urlopen(head).read())[0]
    body = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + length}"})
    header = json.loads(urllib.request.urlopen(body).read())
    header.pop("__metadata__", None)
    return header


def measure(repo: str) -> tuple[Counter, int]:
    api = json.load(urllib.request.urlopen(f"{HF}/api/models/{repo}"))
    # Mistral repos may ship the same weights twice (consolidated-* and model-*): keep one format.
    files = [
        s["rfilename"]
        for s in api["siblings"]
        if s["rfilename"].startswith("consolidated") and s["rfilename"].endswith(".safetensors")
    ]
    if not files:
        raise SystemExit(f"{repo}: no consolidated-*.safetensors (Mistral format) to measure")
    sizes: Counter = Counter()
    layers: set[int] = set()
    for f in files:
        for tensor, meta in read_header(f"{HF}/{repo}/resolve/main/{f}").items():
            if meta["dtype"] not in DTYPE_BYTES:
                raise SystemExit(f"{f}: tensor {tensor} has unsupported dtype {meta['dtype']}")
            n = DTYPE_BYTES[meta["dtype"]]
            for dim in meta["shape"]:
                n *= dim
            sizes[classify(tensor)] += n
            if m := re.match(r"layers\.(\d+)\.", tensor):
                layers.add(int(m.group(1)))
    if not layers:
        raise SystemExit(f"{repo}: no layers.N.* tensors found; not a Mistral-format checkpoint?")
    return sizes, len(layers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("output")
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--kv-values-per-layer",
        type=int,
        default=320,
        help="values cached per token and layer (MLA: kv_lora_rank + qk_rope_head_dim)",
    )
    parser.add_argument("--kv-bytes-per-value", type=int, default=2, help="2 for a bf16 cache")
    args = parser.parse_args()

    sizes, n_layers = measure(args.repo)
    data = {
        "name": args.name,
        "source": f"{HF}/{args.repo}",
        "n_layers": n_layers,
        "kv_bytes_per_token_per_layer": args.kv_values_per_layer * args.kv_bytes_per_value,
        "layered_bytes": {k: sizes.get(k, 0) for k in LAYERED},
        "global_bytes": {k: v for k, v in sorted(sizes.items()) if k not in LAYERED},
    }
    with open(args.output, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    total = sum(sizes.values())
    print(f"{args.name}: {n_layers} layers, {total / 1e9:.2f} GB -> {args.output}")


if __name__ == "__main__":
    main()
