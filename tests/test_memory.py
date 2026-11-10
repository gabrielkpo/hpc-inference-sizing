import pytest

from hpc_inference_sizing.cli import main
from hpc_inference_sizing.memory import GIB, GPUS, GPUSpec, ModelSpec, load_model, plan

# A toy model with round numbers, so every expected value can be checked by hand.
TOY = ModelSpec(
    name="toy",
    n_layers=4,
    kv_bytes_per_token_per_layer=1_000,
    layered_bytes={
        "attention": 4 * GIB,
        "routed_experts": 40 * GIB,
        "shared_expert": 2 * GIB,
        "router": 0,
        "layer_norms": 0,
    },
    global_bytes={"embedding": 2 * GIB, "lm_head": 2 * GIB, "vision": 0, "other": 0},
)
GPU_100 = GPUSpec("toy-gpu", 100 * GIB)


def toy_plan(n_gpus, split, **kw):
    kw = {"gpu_memory_utilization": 1.0, "overhead_bytes": 0, "request_tokens": 1_000, **kw}
    return plan(TOY, GPU_100, n_gpus, split, **kw)


def test_single_gpu_holds_everything():
    p = toy_plan(1, "single")
    g = p.gpus[0]
    assert g.weights_total == 50 * GIB
    assert g.kv_bytes_per_token == 4 * 1_000  # all 4 layers
    assert p.cache_tokens == 50 * GIB // 4_000
    assert p.concurrent_requests == p.cache_tokens // 1_000


def test_tp_cuts_weights_and_copies_the_cache():
    p = toy_plan(2, "tp")
    for g in p.gpus:
        assert g.weights_total == 25 * GIB  # everything cut in two
        assert g.kv_bytes_per_token == 4_000  # all layers on each GPU
    assert p.cache_tokens == 75 * GIB // 4_000  # same tokens on both GPUs: not added


def test_ep_has_the_same_memory_as_tp():
    tp, ep = toy_plan(2, "tp"), toy_plan(2, "ep")
    assert [g.weights_total for g in ep.gpus] == [g.weights_total for g in tp.gpus]
    assert ep.cache_tokens == tp.cache_tokens


def test_pp_splits_layers_and_the_cache():
    p = toy_plan(2, "pp")
    first, last = p.gpus
    # 2 layers each: half of the layered weights, embedding first, output head last
    assert first.weights_total == 23 * GIB + 2 * GIB
    assert last.weights_total == 23 * GIB + 2 * GIB
    assert first.kv_bytes_per_token == last.kv_bytes_per_token == 2_000
    assert p.cache_tokens == 75 * GIB // 2_000  # twice the tp capacity
    assert p.cache_tokens == 2 * toy_plan(2, "tp").cache_tokens


def test_pp_with_uneven_layers_is_limited_by_the_fullest_stage():
    p = toy_plan(3, "pp")  # 4 layers on 3 GPUs: 2, 1, 1
    assert [g.kv_bytes_per_token for g in p.gpus] == [2_000, 1_000, 1_000]
    assert p.cache_tokens == min(g.cache_tokens for g in p.gpus)


def test_dp_ep_copies_attention_and_adds_up_capacities():
    p = toy_plan(2, "dp_ep")
    for g in p.gpus:
        # attention, shared expert, embedding, output head copied (10 GiB); experts spread (20 GiB)
        assert g.weights_total == 30 * GIB
        assert g.kv_bytes_per_token == 4_000
    per_gpu = 70 * GIB // 4_000
    assert p.cache_tokens == 2 * per_gpu  # each GPU caches its own requests
    assert p.concurrent_requests == 2 * (per_gpu // 1_000)


def test_overhead_and_utilization_reduce_free_memory():
    g = toy_plan(1, "single", gpu_memory_utilization=0.9, overhead_bytes=2 * GIB).gpus[0]
    assert g.free_bytes == int(100 * GIB * 0.9) - 50 * GIB - 2 * GIB


def test_does_not_fit_gives_zero_capacity():
    p = plan(TOY, GPUSpec("small", 40 * GIB), 1, "single")
    assert not p.fits
    assert p.cache_tokens == 0 and p.concurrent_requests == 0


@pytest.mark.parametrize(
    "n_gpus, split", [(2, "single"), (1, "tp"), (0, "single"), (2, "zigzag"), (5, "pp")]
)
def test_invalid_combinations_are_rejected(n_gpus, split):
    with pytest.raises(ValueError):
        toy_plan(n_gpus, split)


def test_real_model_data_matches_the_checkpoint():
    nvfp4 = load_model("mistral-small-4-nvfp4")
    assert nvfp4.n_layers == 36
    assert nvfp4.kv_bytes_per_token_per_layer == 640  # (256 + 64) values x 2 bytes
    assert 70.7e9 < nvfp4.total_bytes < 70.9e9
    assert 120.8e9 < load_model("mistral-small-4-fp8").total_bytes < 121.0e9


def test_small_4_needs_two_a40_but_fits_one_h100():
    model = load_model("mistral-small-4-nvfp4")
    assert not plan(model, GPUS["a40"], 1, "single").fits
    assert plan(model, GPUS["h100-pcie"], 1, "single").fits
    tp = plan(model, GPUS["a40"], 2, "tp")
    pp = plan(model, GPUS["a40"], 2, "pp")
    assert tp.fits and pp.fits
    assert 1.8 < pp.cache_tokens / tp.cache_tokens < 2.2


def test_cli_prints_a_plan(capsys):
    main(["memory", "--gpu", "a40", "--gpus", "2", "--split", "pp"])
    out = capsys.readouterr().out
    assert "split pp" in out and "concurrent requests" in out
