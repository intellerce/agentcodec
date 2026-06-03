"""
agentcodec — command-line entry point.

Subcommands:
    run        Run the configured strategy on a prompt or JSONL file.
    eval       Compare 2-N deployment configs on a shared eval prompt set.
    inspect    Print metadata about a router weights JSON.

SemKNN training is not part of the public release. The trained artifacts
are proprietary and live behind the SemKNN backend service; see
COMMERCIAL.md if you need on-premise access.

The console script is registered by ``pyproject.toml`` as ``agentcodec``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    from .api import ReliabilityModule

    mod = ReliabilityModule.from_yaml(args.config)

    # Single prompt
    if args.prompt:
        out = mod.run(
            args.prompt,
            category=args.category,
            return_trace=args.trace,
        )
        if args.trace:
            print(json.dumps(out.to_dict(), indent=2))
        else:
            print(out.text)
        return 0

    # Batch from JSONL
    if args.prompts_file:
        out_f = open(args.out, "w") if args.out else sys.stdout
        try:
            with open(args.prompts_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rec = json.loads(line)
                    out = mod.run(
                        rec["prompt"],
                        category=rec.get("category"),
                        reference=rec.get("reference"),
                        task_id=rec.get("id"),
                        return_trace=args.trace,
                    )
                    payload = out.to_dict() if args.trace else {
                        "id": rec.get("id"),
                        "text": out.text,
                        "technique_used": out.technique_used,
                        "cost_usd": out.cost_usd,
                        "cost_source": out.cost_source,
                        "latency_s": out.latency_s,
                        "thinking_used": out.thinking_used,
                    }
                    out_f.write(json.dumps(payload) + "\n")
                    out_f.flush()
        finally:
            if args.out:
                out_f.close()
        return 0

    print("ERROR: pass --prompt TEXT or --prompts-file PATH", file=sys.stderr)
    return 2


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import Evaluator

    # Parse --config name=path entries.
    configs: dict[str, str] = {}
    for spec in args.config:
        if "=" not in spec:
            print(f"ERROR: --config expects NAME=PATH; got {spec!r}", file=sys.stderr)
            return 2
        name, _, path = spec.partition("=")
        configs[name.strip()] = path.strip()

    if len(configs) < 2:
        print("ERROR: pass at least two --config NAME=PATH entries to compare.", file=sys.stderr)
        return 2

    eval_judge = None
    if args.eval_judge_model:
        eval_judge = {"model": args.eval_judge_model}
        if args.eval_judge_base_url:
            eval_judge["base_url"] = args.eval_judge_base_url

    ev = Evaluator(
        configs=configs,
        prompts_file=args.prompts_file,
        repeats=args.repeats,
        parallel_prompts=args.parallel_prompts,
        eval_judge=eval_judge,
        cache_dir=args.cache_dir,
        on_error=args.on_error,
    )
    report = ev.run(baseline=args.baseline, alpha=args.alpha)
    report.summary()

    if args.out_json:
        report.to_json(args.out_json)
        print(f"Wrote {args.out_json}")
    if args.out_md:
        report.to_markdown(args.out_md)
        print(f"Wrote {args.out_md}")

    # CI gating
    if args.gate_on:
        if not args.baseline:
            print("ERROR: --gate-on requires --baseline.", file=sys.stderr)
            return 2
        candidate = args.gate_candidate or next(c for c in configs if c != args.baseline)
        passed, reason = report.gate(
            baseline=args.baseline,
            candidate=candidate,
            metric=args.gate_on,
            max_regression=args.max_regression,
            alpha=args.alpha,
        )
        print(f"Gate: {reason}")
        return 0 if passed else 1
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    p = Path(args.path)
    blob = json.loads(p.read_text())
    rt = blob.get("router_type", "linear")
    print(f"Router type:    {rt}")
    print(f"Trained on:     {blob.get('trained_on')}")
    print(f"N tasks:        {blob.get('n_tasks')}")
    print(f"Candidates:     {len(blob.get('technique_classes', []))}")
    for t in blob.get("technique_classes", []):
        print(f"   - {t}")
    if rt == "semknn_cost_aware":
        print(f"BGE model:      {blob.get('bge_model')}")
        print(f"k / lambda:     {blob['knn']['k']} / {blob['knn']['lambda']}")
    print(f"Train acc:      {blob.get('train_acc'):.3f}")
    print(f"Train mean q:   {blob.get('train_mean_q'):.4f}")
    cv_q = blob.get("cv_mean_q")
    cv_s = blob.get("cv_std_q")
    if cv_q is not None and cv_s is not None:
        print(f"CV mean q:      {cv_q:.4f} ± {cv_s:.4f}")
    if "cv_mean_cost" in blob:
        print(f"CV mean cost:   ${blob['cv_mean_cost']:.5f} ± ${blob.get('cv_std_cost', 0):.5f}")
    if "fixed_best_technique" in blob:
        print(
            f"Fixed best:     {blob['fixed_best_technique']} "
            f"(q={blob.get('fixed_best_mean_q'):.4f})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentcodec", description=__doc__)
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Run the strategy on a prompt / JSONL file")
    p_run.add_argument("--config", "-c", required=True)
    p_run.add_argument("--prompt")
    p_run.add_argument("--prompts-file")
    p_run.add_argument("--category", choices=["qa", "reasoning", "creative", "code"])
    p_run.add_argument("--out", help="output JSONL path (when --prompts-file is set)")
    p_run.add_argument("--trace", action="store_true",
                       help="emit the full trace dict (return_trace=True)")
    p_run.set_defaults(func=_cmd_run)

    # eval
    p_eval = sub.add_parser(
        "eval",
        help="Compare 2-N deployment configs on a shared eval prompt set.",
    )
    p_eval.add_argument(
        "--config", "-c", action="append", required=True,
        metavar="NAME=PATH",
        help="A named config entry, e.g. -c prod=configs/lib/fixed_harq.yaml. "
             "Pass at least twice.",
    )
    p_eval.add_argument("--prompts-file", required=True,
                        help="JSONL: {id, prompt, category, reference?} per line")
    p_eval.add_argument("--repeats", type=int, default=1)
    p_eval.add_argument("--parallel-prompts", type=int, default=1)
    p_eval.add_argument("--cache-dir", default="results/eval")
    p_eval.add_argument("--on-error", default="continue",
                        choices=["continue", "raise", "fallback_baseline"])
    p_eval.add_argument("--eval-judge-model",
                        help="Override every config's judge with this model "
                             "(recommended for fair head-to-head).")
    p_eval.add_argument("--eval-judge-base-url",
                        help="base_url for --eval-judge-model")
    p_eval.add_argument("--baseline",
                        help="Compare every other config vs this one. "
                             "Required for --gate-on.")
    p_eval.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for paired Wilcoxon (BH-corrected)")
    p_eval.add_argument("--out-json",
                        help="Write the full machine-readable report here")
    p_eval.add_argument("--out-md",
                        help="Write a human-readable Markdown report here")
    # CI gating
    p_eval.add_argument("--gate-on", choices=["quality", "cost_usd", "latency_s"],
                        help="CI mode: exit 1 if --gate-candidate regresses on this "
                             "metric vs --baseline by more than --max-regression.")
    p_eval.add_argument("--gate-candidate",
                        help="Config to gate (defaults to the first non-baseline)")
    p_eval.add_argument("--max-regression", type=float, default=0.0,
                        help="Maximum tolerated regression on --gate-on metric")
    p_eval.set_defaults(func=_cmd_eval)

    # inspect
    p_insp = sub.add_parser("inspect", help="Inspect a router weights JSON")
    p_insp.add_argument("path")
    p_insp.set_defaults(func=_cmd_inspect)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
