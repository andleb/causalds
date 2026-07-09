#!/usr/bin/env python3
"""
Run a cheap structure-only preflight over a quota-driven scene blueprint manifest.

The preflight samples graphs and estimates task yield without variable mapping,
story generation, or full synthetic-data generation.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "causalds"))

from causalds.blueprint import (
    IDENTIFIABILITY_IDENTIFIABLE,
    IDENTIFIABILITY_NONIDENTIFIABLE,
    CompositionConfig,
    SceneBlueprint,
    load_blueprint_manifest,
    sample_graph_for_blueprint,
    summarize_blueprint_requests,
)
from causalds.question_generation import generate_tasks
from causalds.utils import atomic_write_json, ensure_plain
from exp.generate_questions import load_generation_config_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def build_placeholder_data(blueprint: SceneBlueprint, sg) -> pd.DataFrame:
    """Create a tiny placeholder dataset for task-yield estimation."""
    observed_nodes = list(sg.observed_nodes or list(sg.graph.nodes()))
    if not observed_nodes:
        observed_nodes = list(sg.graph.nodes())

    frame = pd.DataFrame(
        {str(node): np.array([0.0, 1.0], dtype=float) for node in observed_nodes}
    )
    if sg.treatment in frame.columns:
        if blueprint.treatment_type == "binary":
            frame[sg.treatment] = np.array([0.0, 1.0], dtype=float)
        else:
            frame[sg.treatment] = np.array([0.25, 0.75], dtype=float)
    if sg.outcome in frame.columns:
        if blueprint.outcome_type == "binary":
            frame[sg.outcome] = np.array([0.0, 1.0], dtype=float)
        else:
            frame[sg.outcome] = np.array([0.2, 0.8], dtype=float)
    return frame


def estimate_task_yield(
    *,
    blueprint: SceneBlueprint,
    sg: Any,
    sections: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate supported task families for one realized graph."""
    questions_cfg = dict(ensure_plain(sections["questions"]) or {})
    placeholder = build_placeholder_data(blueprint, sg)
    mapping = {str(node): str(node) for node in sg.graph.nodes()}
    tasks = generate_tasks(
        scene_id=blueprint.scene_id,
        story="Placeholder story for structure-only preflight.",
        mapping=mapping,
        sg=sg,
        columns=list(placeholder.columns),
        data=placeholder,
        include_r1=bool(questions_cfg.get("include_r1", True)),
        include_r2=bool(questions_cfg.get("include_r2", True)),
        include_r3=bool(
            questions_cfg.get("include_r3", False)
            or questions_cfg.get("include_r3_effects", False)
            or questions_cfg.get("include_r3_identification", False)
        ),
    )
    by_task_type = Counter(str(task.task_type) for task in tasks)
    by_rung = Counter(str(task.rung) for task in tasks)
    return {
        "n_tasks": int(len(tasks)),
        "task_types": dict(sorted(by_task_type.items())),
        "rungs": dict(sorted(by_rung.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structure-only preflight on a scene blueprint manifest."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Path to the blueprint manifest (.jsonl or .json).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Override config YAML layered on top of exp/configs/generation_default.yaml.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional summary output path. Defaults next to the blueprint manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of blueprint rows to simulate.",
    )
    return parser.parse_args()


def summarize_requested(rows: List[SceneBlueprint]) -> Dict[str, Dict[str, int]]:
    """Requested counts straight from blueprint rows."""
    return summarize_blueprint_requests(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.inputs)
    if not input_path.exists():
        raise FileNotFoundError(f"Blueprint manifest not found: {input_path}")

    rows = load_blueprint_manifest(input_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No blueprint rows to preflight in {input_path}")

    sections = load_generation_config_sections(
        Path(args.config) if args.config else None
    )
    composition = CompositionConfig.from_config(
        cfg=sections["cfg"],
        override_cfg=sections.get("override_cfg"),
        benchmark_cfg=sections["benchmark"],
        graph_cfg=sections["graph"],
        data_cfg=sections["data"],
    )

    requested = summarize_requested(rows)
    realized_counts: Dict[str, Counter] = {
        "main_motif": Counter(),
        "graft_count": Counter(),
        "identifiability_regime": Counter(),
        "treatment_type": Counter(),
        "outcome_type": Counter(),
        "continuous_scm_profile": Counter(),
        "binary_scm_profile": Counter(),
        "scm_profile": Counter(),
        "n_nodes": Counter(),
    }
    task_type_counts = Counter()
    rung_counts = Counter()
    task_count_values: List[int] = []
    failures: List[Dict[str, Any]] = []

    for index, blueprint in enumerate(rows, start=1):
        try:
            sg = sample_graph_for_blueprint(
                blueprint=blueprint,
                composition=composition,
                graph_cfg=sections["graph"],
            )
            yield_info = estimate_task_yield(
                blueprint=blueprint,
                sg=sg,
                sections=sections,
            )
        except Exception as exc:
            failures.append(
                {
                    "scene_id": blueprint.scene_id,
                    "scene_index": blueprint.scene_index,
                    "reason": str(exc),
                }
            )
            continue

        applied_grafts = int(
            ((sg.meta or {}).get("augmentation") or {}).get("applied_grafts", 0)
        )
        identifiability_regime = (
            IDENTIFIABILITY_IDENTIFIABLE
            if sg.is_identifiable
            else IDENTIFIABILITY_NONIDENTIFIABLE
        )

        realized_counts["main_motif"][str(sg.motif)] += 1
        realized_counts["graft_count"][str(applied_grafts)] += 1
        realized_counts["identifiability_regime"][identifiability_regime] += 1
        realized_counts["treatment_type"][str(blueprint.treatment_type)] += 1
        realized_counts["outcome_type"][str(blueprint.outcome_type)] += 1
        realized_counts["continuous_scm_profile"][
            str(blueprint.continuous_scm_profile)
        ] += 1
        realized_counts["binary_scm_profile"][str(blueprint.binary_scm_profile)] += 1
        realized_counts["scm_profile"][str(blueprint.scm_profile)] += 1
        realized_counts["n_nodes"][str(len(sg.graph.nodes()))] += 1

        task_count_values.append(int(yield_info["n_tasks"]))
        task_type_counts.update(yield_info["task_types"])
        rung_counts.update(yield_info["rungs"])

        if index % 50 == 0 or index == len(rows):
            logger.info(
                "Preflight progress: %d/%d rows (failures=%d)",
                index,
                len(rows),
                len(failures),
            )

    summary = {
        "input_manifest": str(input_path),
        "config_path": str(sections["config_path"]),
        "n_rows": int(len(rows)),
        "n_success": int(len(rows) - len(failures)),
        "n_failed": int(len(failures)),
        "requested": requested,
        "realized": {
            key: dict(sorted(counter.items()))
            for key, counter in realized_counts.items()
        },
        "task_yield": {
            "n_scenes_counted": int(len(task_count_values)),
            "total_tasks": int(sum(task_count_values)),
            "min_tasks_per_scene": (
                int(min(task_count_values)) if task_count_values else 0
            ),
            "max_tasks_per_scene": (
                int(max(task_count_values)) if task_count_values else 0
            ),
            "mean_tasks_per_scene": (
                float(np.mean(task_count_values)) if task_count_values else 0.0
            ),
            "task_type_totals": dict(sorted(task_type_counts.items())),
            "rung_totals": dict(sorted(rung_counts.items())),
        },
        "observation_variants": {
            "released_per_scene": list(composition.released_observation_variants),
            "view_counts_if_released_for_all_scenes": {
                view: int(len(rows))
                for view in composition.released_observation_variants
            },
        },
        "failures": failures[:100],
    }

    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else input_path.with_suffix(input_path.suffix + ".preflight_summary.json")
    )
    atomic_write_json(summary_out, summary)


if __name__ == "__main__":
    main()
