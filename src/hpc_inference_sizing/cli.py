"""Command line.

hpc-sizing memory --gpu a40 --gpus 2 --split pp          predicted KV-cache capacity
hpc-sizing bench fetch                                    download PIAF and the tokenizer
hpc-sizing bench step  --url URL --model M --rate 1      one load step
hpc-sizing bench sweep --url URL --model M --output F    ladder + bisection, run file
hpc-sizing campaign run --matrix campaign/a40-x2.toml    every run of a matrix, resumable
hpc-sizing vast offers --gpu A40 --num-gpus 2            cheapest matching offers (no key)
hpc-sizing vast rent --offer ID --max-price 1.5 ...      plan; rents only with --yes, then guards
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


def _vast(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import os

    from hpc_inference_sizing import vast

    client = vast.VastClient()
    try:
        if args.action == "offers":
            query = vast.OfferQuery(
                gpu_name=args.gpu,
                num_gpus=args.num_gpus,
                disk_gb=args.disk,
                max_dph=args.max_price,
                limit=args.limit,
            )
            offers = client.search(query)
            print("\n".join(vast.offer_row(o) for o in offers) or "no matching offer")
        elif args.action == "balance":
            print(f"credit: {client.balance():.2f} USD")
        elif args.action == "list":
            for inst in client.instances():
                print(
                    f"{inst.get('id')}  {inst.get('label')}  {inst.get('actual_status')}  "
                    f"{inst.get('num_gpus')} x {inst.get('gpu_name')}"
                )
        elif args.action == "show":
            info = client.show(args.instance)
            print(
                f"{args.instance}: status {info.get('actual_status')}, "
                f"{info.get('dph_total')} USD/h"
            )
            if ssh := vast.ssh_command(info):
                print(ssh)
        elif args.action == "destroy":
            ok = vast.destroy_with_retries(client, args.instance)
            if not ok:
                raise SystemExit(1)
        elif args.action == "guard":
            result = vast.guard(
                client,
                args.instance,
                args.max_hours,
                args.budget_usd,
                price_hint=args.price_per_hour,
            )
            print(f"guard ended: {result.reason}; destroyed: {result.destroyed}")
        elif args.action == "rent":
            env = {}
            if args.pass_hf_token:
                if not os.environ.get("HF_TOKEN"):
                    parser.error("--pass-hf-token: HF_TOKEN is not set in the environment")
                env["HF_TOKEN"] = os.environ["HF_TOKEN"]
            spec = vast.RentSpec(
                image=args.image, disk_gb=args.disk, max_hours=args.max_hours, env=env
            )
            rental = vast.rent(client, args.offer, spec, args.max_price, confirm=args.yes)
            if rental is None:
                return
            if args.no_guard:
                print(
                    f"no guard: run `hpc-sizing vast guard {rental.instance_id} --max-hours "
                    f"{args.max_hours} --budget-usd {args.budget_usd} "
                    f"--price-per-hour {rental.dph_total:.3f}` now"
                )
                return
            result = vast.guard(
                client,
                rental.instance_id,
                args.max_hours,
                args.budget_usd,
                price_hint=rental.dph_total,
                count_instance_age=False,  # fresh rental: age 0, start_date not yet checked
            )
            print(f"guard ended: {result.reason}; destroyed: {result.destroyed}")
    except vast.VastError as err:
        parser.error(str(err))


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

    vastp = sub.add_parser("vast", help="find, rent, guard and destroy a Vast.ai instance")
    vactions = vastp.add_subparsers(dest="action", required=True)
    p = vactions.add_parser("offers", help="cheapest matching offers (no key needed)")
    p.add_argument("--gpu", default="A40")
    p.add_argument("--num-gpus", type=int, default=2)
    p.add_argument("--disk", type=int, default=150, help="GB, included in the price")
    p.add_argument("--max-price", type=float, help="USD/h")
    p.add_argument("--limit", type=int, default=10)
    vactions.add_parser("balance", help="prepaid credit left (key needed)")
    vactions.add_parser("list", help="my instances (key needed)")
    for name, helptext in (("show", "status, price, ssh command"), ("destroy", "destroy now")):
        vactions.add_parser(name, help=helptext).add_argument("instance", type=int)
    p = vactions.add_parser("guard", help="destroy at the time or budget limit, on Ctrl-C")
    p.add_argument("instance", type=int)
    p.add_argument("--max-hours", type=float, required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--price-per-hour", type=float, default=0.0, help="until the API gives it")
    p = vactions.add_parser("rent", help="plan a rental; rents only with --yes, then guards it")
    p.add_argument("--offer", type=int, required=True)
    p.add_argument("--max-price", type=float, required=True, help="USD/h cap on the live price")
    p.add_argument("--max-hours", type=float, required=True, help="deadline of the instance")
    p.add_argument("--budget-usd", type=float, required=True, help="spend cap for the guard")
    p.add_argument("--image", default="vllm/vllm-openai:v0.32.0")
    p.add_argument("--disk", type=int, default=150, help="GB")
    p.add_argument("--pass-hf-token", action="store_true", help="give HF_TOKEN to the instance")
    p.add_argument("--yes", action="store_true", help="really rent (paid)")
    p.add_argument("--no-guard", action="store_true", help="do not watch it from here")

    args = parser.parse_args(argv)
    if args.command == "memory":
        _memory(args, mem)
    elif args.command == "bench":
        _bench(args, bench)
    elif args.command == "campaign":
        _campaign(args, campaign)
    else:
        _vast(args, vastp)


if __name__ == "__main__":
    main()
