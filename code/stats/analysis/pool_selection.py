#!/usr/bin/env python3
"""Select encoder-family pools on repeated calibration/evaluation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.common import write_json
from analysis.pool_config import (
    DEFAULT_OUTPUT,
    DEFAULT_TABLE,
    RULE_DEFINITION,
    SPLIT_COUNT,
    scope_tasks,
)
from analysis.pool_reporting import table_markdown
from analysis.pool_task import run_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--split-count", type=int, default=SPLIT_COUNT)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--tasks", nargs="+", help="restrict to these study tasks (default: all 7); "
                        "the output then carries an explicit RESTRICTED SCOPE label")
    args = parser.parse_args(argv)
    try:
        tasks, scope = scope_tasks(args.tasks)
    except ValueError as exc:
        parser.error(str(exc))
    study_outputs = {DEFAULT_OUTPUT.resolve(), DEFAULT_TABLE.resolve()}
    if scope and study_outputs & {args.output.resolve(), args.table.resolve()}:
        parser.error("a --tasks restriction needs --output and --table away from the full seven-task files")
    previous = {}
    if args.report_only and args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
    payload = (
        previous
        if args.report_only
        else run_analysis(
            args.results_root, args.split_count, tasks
        )
    )
    if scope and not args.report_only:
        payload["scope"] = scope
    write_json(args.output, payload)
    args.table.write_text(table_markdown(payload), encoding="utf-8")
    print(f"{payload['status']}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
