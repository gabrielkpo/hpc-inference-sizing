"""Command line: `hpc-sizing memory --gpu a40 --gpus 2 --split pp`."""

from __future__ import annotations

import argparse

from hpc_inference_sizing.memory import GIB, GPUS, MODELS, SPLITS, format_plan, load_model, plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hpc-sizing")
    sub = parser.add_subparsers(dest="command", required=True)

    mem = sub.add_parser("memory", help="predict KV-cache capacity for a model, GPU and split")
    mem.add_argument("--model", choices=MODELS, default="mistral-small-4-nvfp4")
    mem.add_argument("--gpu", choices=sorted(GPUS), default="a40")
    mem.add_argument("--gpus", type=int, default=2, help="number of GPUs in the serving unit")
    mem.add_argument("--split", choices=SPLITS, default="tp")
    mem.add_argument("--request-tokens", type=int, default=6_400, help="input + output tokens")
    mem.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    mem.add_argument("--overhead-gib", type=float, default=2.0, help="assumed per-GPU overhead")

    args = parser.parse_args(argv)
    if args.command == "memory":
        result = plan(
            load_model(args.model),
            GPUS[args.gpu],
            args.gpus,
            args.split,
            request_tokens=args.request_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            overhead_bytes=int(args.overhead_gib * GIB),
        )
        print(format_plan(result))


if __name__ == "__main__":
    main()
