"""Tabulate run records: ``python -m workflows.evals.report [--runs DIR] [--json]``.

One row per record (case, model, git head, pass, gate value, turns, cost, wall, CPU, broker
requests) sorted by case and time, plus per-case/model aggregates used by the acceptance rule in
the README (pass count, median cost, median turns).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RUNS = Path(__file__).with_name("runs")


def load_records(runs: Path) -> list:
    records = []
    for path in sorted(runs.glob("*/*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        record["_path"] = str(path.relative_to(runs))
        records.append(record)
    return records


def gate_value(record: dict) -> str:
    score = record.get("score") or {}
    if "regret" in score:
        return f"regret={score['regret']}"
    if "held_out_rmse" in score:
        cov = score.get("coverage80_held_out")
        text = f"rmse={score['held_out_rmse']:.2f}"
        return text + (f" cov={cov:.2f}" if cov is not None else "")
    checks = score.get("checks")
    if checks:
        return f"checks={sum(bool(v) for v in checks.values())}/{len(checks)}"
    return score.get("error", "-")[:40]


def rows(records: list) -> list:
    out = []
    for r in records:
        out.append(
            {
                "case": r["case"],
                "model": r["model"],
                "head": r.get("git_head"),
                "utc": r.get("started_utc"),
                "pass": bool((r.get("score") or {}).get("passed")),
                "gate": gate_value(r),
                "turns": r.get("num_turns"),
                "usd": (
                    round(r["total_cost_usd"], 3)
                    if r.get("total_cost_usd") is not None
                    else None
                ),
                "wall_s": r.get("wall_s"),
                "cpu_s": r.get("cpu_s"),
                "broker": len(r.get("broker_requests") or []),
            }
        )
    return sorted(out, key=lambda x: (x["case"], x["utc"] or ""))


def aggregates(table: list) -> list:
    groups: dict = {}
    for row in table:
        groups.setdefault((row["case"], row["model"], row["head"]), []).append(row)
    out = []
    for (case, model, head), items in sorted(groups.items()):
        usd = [x["usd"] for x in items if x["usd"] is not None]
        turns = [x["turns"] for x in items if x["turns"] is not None]
        out.append(
            {
                "case": case,
                "model": model,
                "head": head,
                "runs": len(items),
                "passes": sum(x["pass"] for x in items),
                "median_usd": round(statistics.median(usd), 3) if usd else None,
                "median_turns": statistics.median(turns) if turns else None,
            }
        )
    return out


def format_table(table: list, columns: list) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in table)) for c in columns}
    lines = [" | ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("-+-".join("-" * widths[c] for c in columns))
    for r in table:
        lines.append(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m workflows.evals.report")
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    table = rows(load_records(Path(args.runs)))
    agg = aggregates(table)
    if args.json:
        print(json.dumps({"runs": table, "aggregates": agg}, indent=1, sort_keys=True))
        return 0
    cols = [
        "case",
        "model",
        "head",
        "utc",
        "pass",
        "gate",
        "turns",
        "usd",
        "wall_s",
        "cpu_s",
        "broker",
    ]
    print(format_table(table, cols))
    print()
    print(
        format_table(
            agg,
            ["case", "model", "head", "runs", "passes", "median_usd", "median_turns"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
