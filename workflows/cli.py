"""Command-line entry point: ``python -m workflows <command> --spec study.json`` (one JSON line on stdout)."""

from __future__ import annotations

import argparse
import json
import sys

from workflows import cost
from workflows.spec import StudySpec


def cmd_estimate(args) -> dict:
    spec = StudySpec.load(args.spec)
    from workflows.adapter import load_adapter

    adapter = load_adapter(spec.model.adapter, spec.model.model_dir)
    low, high = cost.planned_simulation_range(spec)
    workers = spec.compute.max_workers or cost.usable_workers(
        spec.compute.threads_per_member, spec.compute.memory_per_member_gb
    )
    estimate = cost.estimate(
        high, adapter.cost_hint(), workers, extra_runs=int(args.extra_runs)
    )
    return {"command": "estimate", "planned_range": [low, high], **estimate.to_dict()}


def cmd_run(args) -> dict:
    spec = StudySpec.load(args.spec)
    if spec.workflow == "ensemble":
        from workflows.ensemble import run_study

        return {
            "command": "run",
            **run_study(spec, args.study, resume=not args.no_resume),
        }
    if spec.workflow == "hm-esmda":
        from workflows.esmda import run_esmda

        summary = run_esmda(spec, args.study)
        return {
            "command": "run",
            "study": args.study,
            "steps": len(summary["steps"]),
            "final": summary["steps"][-1],
        }
    if spec.workflow == "optimize":
        from workflows.optimize import run_exhaustive

        summary = run_exhaustive(spec, args.study)
        return {
            "command": "run",
            "study": args.study,
            "n_candidates": summary["n_candidates"],
            "best": summary["best"],
        }
    raise SystemExit(f"workflow {spec.workflow!r} is not implemented yet")


def cmd_truth(args) -> dict:
    spec = StudySpec.load(args.spec)
    if spec.workflow != "hm-esmda":
        raise SystemExit("truth generation is implemented for hm-esmda specs")
    from workflows.esmda import make_truth

    truth = make_truth(spec, args.study)
    return {
        "command": "truth",
        "study": args.study,
        "nd": len(truth["d_obs"]),
        "n_train": int(sum(truth["train_mask"])),
    }


def cmd_analyze(args) -> dict:
    from workflows.ensemble import analyze

    analysis = analyze(args.study)
    return {
        "command": "analyze",
        "study": args.study,
        "n_failed": analysis["n_failed"],
        "quantities": sorted(analysis["quantities"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m workflows", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("estimate", help="pre-flight cost estimate for a study spec")
    p.add_argument("--spec", required=True)
    p.add_argument(
        "--extra-runs",
        default=0,
        help="post-validation or reference runs beyond the design",
    )
    p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("run", help="run a study (resumes completed members by default)")
    p.add_argument("--spec", required=True)
    p.add_argument("--study", required=True, help="study directory")
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser(
        "truth", help="generate an identical-twin truth for a history-matching study"
    )
    p.add_argument("--spec", required=True)
    p.add_argument("--study", required=True)
    p.set_defaults(func=cmd_truth)
    p = sub.add_parser("analyze", help="analyze a finished ensemble study")
    p.add_argument(
        "--spec",
        required=False,
        help="accepted for symmetry; the study dir holds the spec",
    )
    p.add_argument("--study", required=True)
    p.set_defaults(func=cmd_analyze)
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
