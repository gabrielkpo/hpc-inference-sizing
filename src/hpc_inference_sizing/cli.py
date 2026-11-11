"""Command line.

hpc-sizing memory --gpu a40 --gpus 2 --split pp          predicted KV-cache capacity
hpc-sizing bench fetch                                    download PIAF and the tokenizer
hpc-sizing bench step  --url URL --model M --rate 1      one load step
hpc-sizing bench sweep --url URL --model M --output F    ladder + bisection, run file
hpc-sizing campaign run --matrix campaign/a40-x2.toml    every run of a matrix, resumable
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from hpc_inference_sizing.memory import GIB, GPUS, MODELS, SPLITS, format_plan, load_model, plan


def _memory(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        result = plan(
            load_model(args.model),
            GPUS[args.gpu],
            args.gpus,
            args.split,
            request_tokens=args.request_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            overhead_bytes=int(args.overhead_gib * GIB),
        )
    except ValueError as err:
        parser.error(str(err))  # e.g. one GPU needs --split single
    print(format_plan(result))


def _parse_meta(pairs: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    meta = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            parser.error(f"--meta expects KEY=VALUE, got {pair!r}")
        meta[key] = value
    return meta


def _bench(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # Imported here so that `hpc-sizing memory` does not load the bench dependencies.
    from hpc_inference_sizing.bench import data
    from hpc_inference_sizing.bench.runner import Bench, metadata, write_run
    from hpc_inference_sizing.bench.workload import PROFILES, Workload

    if args.action == "fetch":
        print(data.fetch(data.PIAF_URL, "piaf-train.parquet"))
        print(data.fetch(data.TOKENIZER_URL, "small4-tekken.json"))
        return

    if args.output and Path(args.output).exists() and not args.overwrite:
        parser.error(f"{args.output} exists; pass --overwrite to replace it")
    meta_extra = _parse_meta(args.meta, parser)
    workload = Workload(
        data.load_piaf(), data.MistralTokenizer(), PROFILES[args.profile], args.seed
    )
    bench = Bench(
        args.url.rstrip("/"),
        args.model,
        workload,
        args.step_duration,
        seed=args.seed,
        min_requests=args.min_requests,
    )

    result = None
    if args.action == "step":
        summary = asyncio.run(bench.step(args.rate))
        print(_format_summary(summary))
    else:
        result = asyncio.run(
            bench.sweep(
                args.targets,
                start_rate=args.start_rate,
                max_rate=args.max_rate,
                bisect_steps=args.bisect_steps,
                tpot_bound_s=args.tpot_bound_ms / 1000,
            )
        )
        for s in result.steps:
            print(_format_summary(s))
        for target, cap in result.capacity.items():
            print(
                f"capacity at TTFT p95 <= {target:g} s: {cap if cap is not None else 'none'} req/s"
            )
    if args.output:
        write_run(Path(args.output), metadata(bench, meta_extra), bench, result)
        print(f"run file: {args.output}")


def _campaign(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from hpc_inference_sizing.campaign import Limits, load_matrix, run_campaign, status

    matrix = load_matrix(Path(args.matrix))
    if args.action == "list":
        print("\n".join(matrix.run_ids()))
        return
    if args.action == "status":
        st = status(matrix)
        print(f"done ({len(st.done)}): {' '.join(st.done)}")
        print(f"failed ({len(st.failed)}): {' '.join(st.failed)}")
        print(f"pending ({len(st.pending)}): {' '.join(st.pending)}")
        return
    try:
        limits = Limits(
            args.max_hours,
            args.budget_usd,
            args.price_per_hour,
            clock_file=matrix.results_dir / "campaign-clock.json",
        )
        final = run_campaign(
            matrix,
            limits,
            _parse_meta(args.meta, parser),
            only=args.only or None,
            retry_failed=args.retry_failed,
        )
    except ValueError as err:
        parser.error(str(err))
    print(f"done {len(final.done)}, failed {len(final.failed)}, pending {len(final.pending)}")


def _format_summary(s) -> str:
    return (
        f"rate {s.rate:6.3f} req/s | {s.n_requests:4d} req, {s.n_errors} err | "
        f"TTFT p50 {s.ttft_p50:6.2f} s p95 {s.ttft_p95:6.2f} s | "
        f"TPOT p50 {s.tpot_p50 * 1e3:6.1f} ms p95 {s.tpot_p95 * 1e3:6.1f} ms | "
        f"{s.output_tokens_per_s:7.1f} tok/s"
    )


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
    mem.add_argument(
        "--overhead-gib",
        type=float,
        default=2.0,
        help="assumed per-GPU overhead in GiB (printed in GB: 2 GiB = 2.15 GB)",
    )

    bench = sub.add_parser("bench", help="measure an OpenAI-compatible server under load")
    actions = bench.add_subparsers(dest="action", required=True)
    actions.add_parser("fetch", help="download PIAF and the Mistral Small 4 tokenizer")
    for name, helptext in (("step", "one load step"), ("sweep", "ladder + bisection")):
        p = actions.add_parser(name, help=helptext)
        p.add_argument("--url", required=True, help="server base URL, e.g. http://localhost:8000")
        p.add_argument("--model", required=True, help="model name the server expects")
        p.add_argument("--profile", choices=["rag", "chat"], default="rag")
        p.add_argument("--step-duration", type=float, default=180.0, help="seconds per step")
        p.add_argument(
            "--min-requests",
            type=int,
            default=20,
            help="lengthen a step so that at least this many requests are expected",
        )
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--output", help="write a run file (JSON) here")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--meta", action="append", default=[], help="KEY=VALUE run metadata")
        if name == "step":
            p.add_argument("--rate", type=float, required=True, help="requests per second")
        else:
            p.add_argument("--targets", type=float, nargs="+", default=[1.0, 2.0, 3.0])
            p.add_argument("--start-rate", type=float, default=0.25)
            p.add_argument("--max-rate", type=float, default=64.0)
            p.add_argument("--bisect-steps", type=int, default=2)
            p.add_argument(
                "--tpot-bound-ms",
                type=float,
                default=1000 / 15,
                help="TPOT p95 bound of the SLO (protocol: 66.7 ms); change only for dry runs",
            )

    campaign = sub.add_parser("campaign", help="run a matrix of (unit, profile) sweeps")
    cactions = campaign.add_subparsers(dest="action", required=True)
    for name, helptext in (
        ("list", "print the run ids of a matrix"),
        ("status", "done, failed and pending runs"),
        ("run", "run the pending runs, resumable"),
    ):
        p = cactions.add_parser(name, help=helptext)
        p.add_argument("--matrix", required=True, help="campaign matrix (TOML)")
        if name == "run":
            p.add_argument("--only", action="append", default=[], help="run id (repeatable)")
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--max-hours", type=float, help="stop starting runs after this")
            p.add_argument("--budget-usd", type=float, help="stop starting runs past this spend")
            p.add_argument("--price-per-hour", type=float, default=0.0, help="instance USD/h")
            p.add_argument("--meta", action="append", default=[], help="KEY=VALUE run metadata")

    args = parser.parse_args(argv)
    if args.command == "memory":
        _memory(args, mem)
    elif args.command == "bench":
        _bench(args, bench)
    else:
        _campaign(args, campaign)


if __name__ == "__main__":
    main()
