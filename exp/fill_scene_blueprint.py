#!/usr/bin/env python3
"""
Fill a quota-driven scene blueprint manifest into a runnable batch manifest.

This stage turns desired blueprint rows into fully realized `sampled_graph` +
`datagen_spec` rows while preserving the existing batch worker contract.
"""

import argparse
import copy
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "causalds"))

from causalds.blueprint import (
    CompositionConfig,
    SceneBlueprint,
    build_blueprint_realization_record,
    build_datagen_spec_for_blueprint,
    load_blueprint_manifest,
    resolve_released_observation_configs,
    sample_graph_for_blueprint,
    summarize_blueprint_requests,
    write_resolved_blueprint_config,
)
from causalds.utils import atomic_write_json
from exp.generate_questions import load_generation_config_sections
from exp.prepare_question_inputs import (
    ManifestWriter,
    build_manifest_row,
    manifest_resolved_config_path,
    normalize_manifest_output_path,
    resolve_generation_settings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def summary_path(output_path: Path) -> Path:
    """Default sidecar path for fill summary metadata."""
    return output_path.with_suffix(output_path.suffix + ".summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill a scene blueprint manifest into a runnable batch manifest."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Input blueprint manifest (.jsonl or .json).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output runnable manifest path (.jsonl, .json, or .parquet).",
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
        help="Optional summary output path. Defaults next to the runnable manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of blueprint rows to fill.",
    )
    parser.add_argument(
        "--realization-max-seed-restarts",
        type=int,
        default=None,
        help=(
            "Optional override for composition.realization.max_seed_restarts. "
            "Successful rows still use their original scene seed; this only "
            "extends retry blocks for rows that fail the original seed."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def summarize_requested(rows: List[SceneBlueprint]) -> Dict[str, Dict[str, int]]:
    """Requested counts straight from blueprint rows."""
    return summarize_blueprint_requests(rows)


def build_fill_summary(
    *,
    input_path: Path,
    output_path: Path,
    config_path: Path,
    requested_rows: List[SceneBlueprint],
    emitted_count: int,
    realized_counts: Dict[str, Counter],
    fill_attempts: List[int],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact requested-vs-realized summary for the filler stage."""
    mean_fill_attempts = (
        float(sum(fill_attempts) / len(fill_attempts)) if fill_attempts else 0.0
    )
    return {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "config_path": str(config_path),
        "n_requested": int(len(requested_rows)),
        "n_emitted": int(emitted_count),
        "n_failed": int(len(failures)),
        "requested": summarize_requested(requested_rows),
        "realized": {
            key: dict(sorted(counter.items()))
            for key, counter in realized_counts.items()
        },
        "fill_attempts": {
            "n_rows_counted": int(len(fill_attempts)),
            "min": int(min(fill_attempts)) if fill_attempts else 0,
            "max": int(max(fill_attempts)) if fill_attempts else 0,
            "mean": mean_fill_attempts,
        },
        "failures": failures[:100],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.inputs)
    output_path = normalize_manifest_output_path(args.output)
    resolved_config_path = manifest_resolved_config_path(output_path)
    summary_out_path = (
        Path(args.summary_out) if args.summary_out else summary_path(output_path)
    )

    if not args.overwrite and (
        output_path.exists()
        or resolved_config_path.exists()
        or summary_out_path.exists()
    ):
        raise FileExistsError(
            "One or more fill outputs already exist. Use --overwrite to replace them."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Blueprint manifest not found: {input_path}")

    rows = load_blueprint_manifest(input_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError(f"No blueprint rows to fill from {input_path}")

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
    if args.realization_max_seed_restarts is not None:
        if args.realization_max_seed_restarts < 0:
            raise ValueError("--realization-max-seed-restarts must be >= 0.")
        composition.realization_max_seed_restarts = int(
            args.realization_max_seed_restarts
        )
    generation_settings = resolve_generation_settings(
        sections=sections,
        args=SimpleNamespace(),
    )

    logger.info("Filling %d blueprint rows into %s", len(rows), output_path)
    logger.info("Blueprint manifest: %s", input_path)
    logger.info("Config: %s", sections["config_path"])

    realized_counts: Dict[str, Counter] = {
        "main_motif": Counter(),
        "graft_count": Counter(),
        "identifiability_regime": Counter(),
        "treatment_type": Counter(),
        "outcome_type": Counter(),
        "continuous_scm_profile": Counter(),
        "binary_scm_profile": Counter(),
        "scm_profile": Counter(),
        "observation_variant": Counter(),
        "n_nodes": Counter(),
    }
    fill_attempts: List[int] = []
    failures: List[Dict[str, Any]] = []

    writer = ManifestWriter(output_path)
    compile_succeeded = False
    emitted_count = 0
    try:
        for blueprint in rows:
            try:
                sg = sample_graph_for_blueprint(
                    blueprint=blueprint,
                    composition=composition,
                    graph_cfg=sections["graph"],
                )
                realized_scene_seed = int(
                    ((sg.meta or {}).get("blueprint_realization") or {}).get(
                        "scene_seed",
                        blueprint.scene_seed,
                    )
                )
                datagen_spec = build_datagen_spec_for_blueprint(
                    blueprint=blueprint,
                    sg=sg,
                    data_cfg=sections["data"],
                    seed=realized_scene_seed,
                )
                observation_config, observation_variants = (
                    resolve_released_observation_configs(
                        blueprint=blueprint,
                        observation_config=generation_settings["observation_config"],
                        observation_variants=generation_settings[
                            "observation_variants"
                        ],
                    )
                )

                row_generation_settings = copy.deepcopy(generation_settings)
                row_generation_settings["observation_config"] = observation_config
                row_generation_settings["observation_variants"] = observation_variants

                realized = build_blueprint_realization_record(
                    blueprint=blueprint,
                    sg=sg,
                    datagen_spec=datagen_spec,
                )
                row = build_manifest_row(
                    scene_index=blueprint.scene_index,
                    scene_id=blueprint.scene_id,
                    scene_seed=realized_scene_seed,
                    motif_request=blueprint.main_motif,
                    motif_choice=str(sg.motif),
                    sg=sg,
                    generation_settings=row_generation_settings,
                    datagen_spec=datagen_spec,
                    extra_row_fields={
                        "blueprint": blueprint.to_dict(),
                        "realized": realized,
                    },
                )
                writer.write_row(row)
            except Exception as exc:
                failures.append(
                    {
                        "scene_id": blueprint.scene_id,
                        "scene_index": blueprint.scene_index,
                        "reason": str(exc),
                    }
                )
                continue

            emitted_count += 1
            realized_counts["main_motif"][str(realized["main_motif"])] += 1
            realized_counts["graft_count"][str(realized["applied_grafts"])] += 1
            realized_counts["identifiability_regime"][
                str(realized["identifiability_regime"])
            ] += 1
            realized_counts["treatment_type"][str(realized["treatment_type"])] += 1
            realized_counts["outcome_type"][str(realized["outcome_type"])] += 1
            realized_counts["continuous_scm_profile"][
                str(realized["continuous_scm_profile"])
            ] += 1
            realized_counts["binary_scm_profile"][
                str(realized["binary_scm_profile"])
            ] += 1
            realized_counts["scm_profile"][str(realized["scm_profile"])] += 1
            realized_counts["n_nodes"][str(realized["n_nodes"])] += 1
            for view_name in realized["released_observation_variants"]:
                realized_counts["observation_variant"][str(view_name)] += 1
            fill_attempts.append(int(realized["fill_attempts"]))

            if emitted_count % 25 == 0 or emitted_count == len(rows):
                logger.info(
                    "Filled %d/%d rows (failures=%d)",
                    emitted_count,
                    len(rows),
                    len(failures),
                )

        compile_succeeded = True
        writer.close()
    finally:
        if not compile_succeeded:
            writer.abort()

    write_resolved_blueprint_config(
        path=resolved_config_path,
        cfg=sections["cfg"],
        composition=composition,
    )
    atomic_write_json(
        summary_out_path,
        build_fill_summary(
            input_path=input_path,
            output_path=output_path,
            config_path=sections["config_path"],
            requested_rows=rows,
            emitted_count=emitted_count,
            realized_counts=realized_counts,
            fill_attempts=fill_attempts,
            failures=failures,
        ),
    )

    logger.info(
        "Blueprint fill complete: emitted %d/%d rows to %s",
        emitted_count,
        len(rows),
        output_path,
    )
    if failures:
        logger.warning("Blueprint fill recorded %d failed rows", len(failures))


if __name__ == "__main__":
    main()
