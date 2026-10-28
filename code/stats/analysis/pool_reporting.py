"""Markdown reporting for encoder-family pool results."""

from __future__ import annotations

from typing import Any

from analysis.pool_config import TASKS


def _tasks(payload: dict[str, Any]) -> list[str]:
    return [task for task in TASKS if task in payload["tasks"]]


def _metric(row: dict[str, Any]) -> str:
    if row["value"] is None:
        return "NA"
    ci = row["ci95"]
    return f"{row['value']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _metric_section(payload: dict[str, Any], title: str, metric: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| task | encoder-family selection [95% CI] |",
        "|---|---:|",
    ]
    for task in _tasks(payload):
        row = payload["tasks"][task]["thresholds"]["empirical_p33"]
        lines.append(f"| {task} | {_metric(row['rules']['architecture_diversity'][metric])} |")
    return lines + [""]


def table_markdown(payload: dict[str, Any]) -> str:
    lines = [f"**{payload['scope']['label']}**", ""] if payload.get("scope") else []
    lines += [
        "Primary threshold: calibration-only empirical per-case P33 (calibration-best member). "
        f"Values are means over {payload['protocol']['n_repeated_splits']} repeated paired splits "
        "with split-bootstrap 95% CIs.",
        "",
    ]
    sections = (
        ("Conditional recovery (higher is better)", "conditional_recovery"),
        ("All-fail probability (lower is better)", "all_fail_probability"),
        ("Held-out marginal Dice (higher is better)", "marginal_dice"),
    )
    for title, metric in sections:
        lines.extend(_metric_section(payload, title, metric))
    return "\n".join(lines) + "\n"
