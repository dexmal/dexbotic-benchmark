"""Shared genie_sim official-score aggregation."""

from __future__ import annotations

import math
from typing import Any


OFFICIAL_SCORE_METRIC = "sample_weighted_native_step_score"
WEIGHTED_SCORE_METRIC = "sample_count_weighted_episode_step_score"
POUR_WORKPIECE_CONFIG = "g2op_manip_pour_workpiece"


def category_from_config(config_name: str) -> str:
    if config_name.startswith("g2op_if_"):
        return "instruction"
    if config_name.startswith("g2op_manip_"):
        return "manipulation"
    if config_name.startswith("g2op_spatial_"):
        return "spatial"
    for suffix in ("backgen", "camposgen", "camqualgen", "instructgen", "posegen"):
        if config_name.startswith(f"g2op_robust_{suffix}_"):
            return f"robust_{suffix}"
    if config_name.startswith("g2op_robust_"):
        return "robust"
    if config_name.startswith("g1op_s2r_"):
        return "s2r"
    raise ValueError(f"cannot infer GenieSim category from config {config_name!r}")


def summarize_official_score(
    units: list[dict[str, Any]], *, expected_units: int
) -> dict[str, Any]:
    """Weight step means by episodes, preserving native scoring semantics.

    Use weighted_step_scores instead of the rounded statistics.average or the
    episode STEPS list. ``pour_workpiece`` is the exception: its four
    consecutive Inside checkers are collapsed by the native summarizer, so use
    the aggregate reconstructed from the raw progress rows by the evaluator.
    """
    errors: list[str] = []
    step_score_sum = 0.0
    scored_episodes = 0
    total_episodes = sum(int(unit.get("episodes", 0)) for unit in units)
    if len(units) != expected_units or not units:
        errors.append(f"expected {expected_units} task/seed runs, got {len(units)}")
    for unit in units:
        label = f"{unit.get('task_config', 'unknown')} seed={unit.get('seed', '?')}"
        if unit["status"] != "completed":
            errors.append(f"{label}: status={unit['status']}")
            continue
        try:
            if unit.get("task_config") == POUR_WORKPIECE_CONFIG:
                unit_episodes = unit["progress_score_samples"]
                unit_sum = float(unit["progress_score_sum"])
                if type(unit_episodes) is not int or unit_episodes <= 0:
                    raise ValueError("invalid progress_score_samples")
                if not math.isfinite(unit_sum) or not 0 <= unit_sum <= unit_episodes:
                    raise ValueError("invalid progress_score_sum")
            else:
                native_statistics = unit.get("native_statistics")
                if not isinstance(native_statistics, list) or not native_statistics:
                    raise ValueError("missing native_statistics")
                unit_episodes = 0
                unit_sum = 0.0
                for native in native_statistics:
                    statistics = native["statistics"]
                    counts = statistics["task_counts"]
                    if (
                        not isinstance(counts, dict)
                        or not counts
                        or any(
                            type(count) is not int or count <= 0
                            for count in counts.values()
                        )
                    ):
                        raise ValueError("invalid native task_counts")
                    episodes = sum(counts.values())
                    steps = statistics["weighted_step_scores"]
                    if not isinstance(steps, list) or not steps:
                        raise ValueError("missing weighted_step_scores")
                    scores = [float(step["score"]) for step in steps]
                    if any(
                        not math.isfinite(score) or not 0 <= score <= 1
                        for score in scores
                    ):
                        raise ValueError(
                            "invalid native step score (expected finite [0, 1])"
                        )
                    unit_sum += episodes * (sum(scores) / len(scores))
                    unit_episodes += episodes
            if unit_episodes != int(unit.get("episodes", 0)):
                raise ValueError(
                    f"score sample total {unit_episodes} differs from "
                    f"parsed episodes {unit.get('episodes', 0)}"
                )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{label}: {exc}")
            continue
        step_score_sum += unit_sum
        scored_episodes += unit_episodes
    complete = not errors and scored_episodes > 0
    return {
        "metric": OFFICIAL_SCORE_METRIC,
        "official_score": step_score_sum / scored_episodes if complete else None,
        "complete": complete,
        "episodes": total_episodes,
        "scored_episodes": scored_episodes,
        "step_score_sum": step_score_sum,
        "total_units": len(units),
        "expected_units": expected_units,
        "errors": errors,
    }


def summarize_weighted_score(
    units: list[dict[str, Any]], *, expected_units: int
) -> dict[str, Any]:
    """Average per-episode STEPS means, weighting tasks by sample count."""
    errors: list[str] = []
    score_sum = 0.0
    sample_count = 0
    if len(units) != expected_units or not units:
        errors.append(f"expected {expected_units} task/seed runs, got {len(units)}")
    for unit in units:
        label = f"{unit.get('task_config', 'unknown')} seed={unit.get('seed', '?')}"
        if unit["status"] != "completed":
            errors.append(f"{label}: status={unit['status']}")
            continue
        try:
            unit_samples = unit["weighted_score_samples"]
            if type(unit_samples) is not int or unit_samples <= 0:
                raise ValueError("invalid weighted_score_samples")
            unit_sum = float(unit["weighted_score_sum"])
            if not math.isfinite(unit_sum) or not 0 <= unit_sum <= unit_samples:
                raise ValueError("invalid weighted_score_sum")
            if unit_samples != int(unit.get("episodes", 0)):
                raise ValueError(
                    f"weighted score sample total {unit_samples} differs from "
                    f"parsed episodes {unit.get('episodes', 0)}"
                )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"{label}: {exc}")
            continue
        score_sum += unit_sum
        sample_count += unit_samples
    complete = not errors and sample_count > 0
    return {
        "weighted_score_metric": WEIGHTED_SCORE_METRIC,
        "weighted_score": score_sum / sample_count if complete else None,
        "weighted_score_complete": complete,
        "weighted_score_samples": sample_count,
        "weighted_score_sum": score_sum,
        "weighted_score_errors": errors,
    }


def summarize_scores(
    units: list[dict[str, Any]], *, expected_units: int
) -> dict[str, Any]:
    """Return native official and direct sample-weighted scores together."""
    summary = summarize_official_score(units, expected_units=expected_units)
    weighted = summarize_weighted_score(units, expected_units=expected_units)
    summary.update(weighted)
    summary["complete"] = summary["complete"] and weighted["weighted_score_complete"]
    return summary
