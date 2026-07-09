import copy
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from omegaconf import OmegaConf

from . import data as cd
from . import graph as cg
from .graph import MOTIFS
from .utils import (
    atomic_write_json,
    coerce_list,
    deep_merge_dicts,
    ensure_plain,
    ensure_plain_dict,
    json_safe,
    offset_random_seed,
    write_text_atomic,
)

IDENTIFIABILITY_IDENTIFIABLE = "identifiable"
IDENTIFIABILITY_NONIDENTIFIABLE = "nonidentifiable"
VALID_IDENTIFIABILITY_REGIMES = {
    IDENTIFIABILITY_IDENTIFIABLE,
    IDENTIFIABILITY_NONIDENTIFIABLE,
}
VALID_VARIABLE_TYPES = {"binary", "continuous"}
STRUCTURALLY_NONIDENTIFIABLE_MAIN_MOTIFS = {"double_nc"}


def normalize_weights(
    raw: Dict[Any, Any],
    *,
    field_name: str,
    key_transform=None,
) -> Dict[Any, float]:
    """Validate and normalize a weight dict."""
    items = ensure_plain_dict(raw)
    if not items:
        raise ValueError(f"{field_name} must not be empty.")

    normalized: Dict[Any, float] = {}
    total = 0.0
    for raw_key, raw_value in items.items():
        key = key_transform(raw_key) if key_transform is not None else raw_key
        weight = float(raw_value)
        if weight < 0.0:
            raise ValueError(f"{field_name}[{raw_key!r}] must be >= 0, got {weight}.")
        if weight == 0.0:
            continue
        normalized[key] = weight
        total += weight

    if total <= 0.0:
        raise ValueError(f"{field_name} must contain at least one positive weight.")

    return {key: weight / total for key, weight in normalized.items()}


def largest_remainder_counts(weights: Dict[Any, float], total: int) -> Dict[Any, int]:
    """Allocate an exact item count from normalized weights."""
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}.")
    if total == 0:
        return {key: 0 for key in weights}

    raw = {key: float(weight) * total for key, weight in weights.items()}
    counts = {key: int(value) for key, value in raw.items()}
    assigned = sum(counts.values())
    if assigned < total:
        remainders = sorted(
            ((raw[key] - counts[key], str(key), key) for key in weights),
            reverse=True,
        )
        for _, _, key in remainders[: total - assigned]:
            counts[key] += 1
    return counts


def expand_counts(counts: Dict[Any, int], rng: random.Random) -> List[Any]:
    """Expand exact counts into a shuffled list."""
    items: List[Any] = []
    for key, count in counts.items():
        items.extend([key] * int(count))
    rng.shuffle(items)
    return items


def parse_motif_name(value: Any) -> str:
    """Normalize and validate a motif name."""
    motif = str(value).strip().lower()
    if motif not in MOTIFS:
        raise ValueError(f"Unknown motif {value!r}. Available motifs: {sorted(MOTIFS)}")
    return motif


def parse_identifiability_regime(value: Any) -> str:
    """Normalize and validate an identifiability regime."""
    regime = str(value).strip().lower()
    if regime not in VALID_IDENTIFIABILITY_REGIMES:
        raise ValueError(
            f"Unknown identifiability regime {value!r}. "
            f"Expected one of {sorted(VALID_IDENTIFIABILITY_REGIMES)}."
        )
    return regime


def parse_variable_type(value: Any) -> str:
    """Normalize and validate a treatment/outcome type."""
    var_type = str(value).strip().lower()
    if var_type not in VALID_VARIABLE_TYPES:
        raise ValueError(
            f"Unknown variable type {value!r}. Expected one of {sorted(VALID_VARIABLE_TYPES)}."
        )
    return var_type


def parse_graft_count(value: Any) -> int:
    """Normalize a graft count key."""
    graft_count = int(value)
    if graft_count < 0:
        raise ValueError(f"graft_count must be >= 0, got {graft_count}.")
    return graft_count


@lru_cache(maxsize=None)
def motif_has_treatment_outcome_path(motif_name: str) -> bool:
    """Return whether a motif exposes an observed directed path from treatment to outcome."""
    motif_graph = cg.build_motif(motif_name)
    observed_graph = motif_graph.graph.copy()
    observed_graph.remove_nodes_from(list(motif_graph.latent_nodes or []))
    if (
        motif_graph.treatment not in observed_graph
        or motif_graph.outcome not in observed_graph
    ):
        return False
    return nx.has_path(observed_graph, motif_graph.treatment, motif_graph.outcome)


@dataclass
class CompositionConfig:
    """Paper-facing benchmark composition plan."""

    scene_count: int
    main_motif_weights: Dict[str, float]
    graft_count_probs: Dict[int, float]
    identifiability_regime_weights: Dict[str, float]
    treatment_type_weights: Dict[str, float]
    outcome_type_weights: Dict[str, float]
    continuous_scm_profile_weights: Dict[str, float]
    binary_scm_profile_weights: Dict[str, float]
    auxiliary_motif_weights: Dict[str, float]
    released_observation_variants: List[str]
    graft_safe_main_motifs: List[str]
    nonidentifiable_motif_allowlist: List[str]
    nonidentifiable_p_latent_xy: float = 1.0
    nonidentifiable_max_tries: int = 200
    realization_max_seed_restarts: int = 0
    realization_seed_restart_stride: int = 1_000_003
    blueprint_seed_offset: int = 0

    def validate_main_motif_slot_sanity(self) -> None:
        """Fast-fail on obviously impossible aggregate motif/slot marginals."""
        # These are only coarse necessary conditions.
        # Exact slot-level feasibility is still determined later by
        # _assign_motifs_to_blueprint_slots().
        graft_safe_set = set(self.graft_safe_main_motifs or [])
        grafted_prob_mass = sum(
            float(prob)
            for graft_count, prob in self.graft_count_probs.items()
            if int(graft_count) > 0
        )
        if graft_safe_set and grafted_prob_mass > 0.0:
            graft_safe_prob_mass = sum(
                float(weight)
                for motif_name, weight in self.main_motif_weights.items()
                if motif_name in graft_safe_set
            )
            if graft_safe_prob_mass + 1e-12 < grafted_prob_mass:
                raise ValueError(
                    "composition.main_motif_weights and composition.graft_count_probs are "
                    "incompatible with graph.grafting.main_graph_motifs: "
                    f"graft-safe main motifs carry only {graft_safe_prob_mass:.6f} total "
                    f"probability mass, but grafted rows require {grafted_prob_mass:.6f}. "
                    "Increase the mass on graft-safe main motifs, reduce graft_count_probs "
                    "for k>0 rows, or relax graph.grafting.main_graph_motifs."
                )

        nonident_prob_mass = float(
            self.identifiability_regime_weights.get(
                IDENTIFIABILITY_NONIDENTIFIABLE,
                0.0,
            )
        )
        if nonident_prob_mass > 0.0:
            path_capable_prob_mass = sum(
                float(weight)
                for motif_name, weight in self.main_motif_weights.items()
                if motif_has_treatment_outcome_path(str(motif_name))
            )
            if path_capable_prob_mass + 1e-12 < nonident_prob_mass:
                raise ValueError(
                    "composition.main_motif_weights and "
                    "composition.identifiability_regime_weights are incompatible: "
                    f"treatment-outcome-path motifs carry only "
                    f"{path_capable_prob_mass:.6f} total probability mass, but "
                    f"nonidentifiable rows require {nonident_prob_mass:.6f}. "
                    "Increase the mass on path-capable motifs or reduce the "
                    "nonidentifiable row mass."
                )

        identifiable_prob_mass = float(
            self.identifiability_regime_weights.get(
                IDENTIFIABILITY_IDENTIFIABLE,
                0.0,
            )
        )
        if identifiable_prob_mass > 0.0:
            structurally_identifiable_prob_mass = sum(
                float(weight)
                for motif_name, weight in self.main_motif_weights.items()
                if motif_name not in STRUCTURALLY_NONIDENTIFIABLE_MAIN_MOTIFS
            )
            if structurally_identifiable_prob_mass + 1e-12 < identifiable_prob_mass:
                raise ValueError(
                    "composition.main_motif_weights and "
                    "composition.identifiability_regime_weights are incompatible: "
                    f"structurally identifiable motifs carry only "
                    f"{structurally_identifiable_prob_mass:.6f} total probability mass, "
                    f"but identifiable rows require {identifiable_prob_mass:.6f}. "
                    "Increase the mass on structurally identifiable motifs or reduce "
                    "the identifiable row mass."
                )

    @classmethod
    def from_config(
        cls,
        *,
        cfg: Any,
        override_cfg: Any = None,
        benchmark_cfg: Any,
        graph_cfg: Any,
        data_cfg: Any,
    ) -> "CompositionConfig":
        """Resolve composition settings from the layered generation config."""
        cfg_dict = ensure_plain_dict(cfg)
        override_cfg_dict = ensure_plain_dict(override_cfg)
        benchmark_cfg = ensure_plain_dict(benchmark_cfg)
        graph_cfg = ensure_plain_dict(graph_cfg)
        data_cfg = ensure_plain_dict(data_cfg)
        composition_cfg = ensure_plain_dict(cfg_dict.get("composition"))
        override_composition_cfg = ensure_plain_dict(
            override_cfg_dict.get("composition")
        )
        override_data_cfg = ensure_plain_dict(override_cfg_dict.get("data"))

        if not composition_cfg:
            raise ValueError(
                "Config does not contain a composition section. "
                "Add `composition:` to the override YAML before using blueprint tooling."
            )

        def composition_value(key: str, fallback: Any) -> Any:
            if key in override_composition_cfg:
                return override_composition_cfg[key]
            return composition_cfg.get(key, fallback)

        scene_count_raw = composition_value("scene_count", None)
        if scene_count_raw is None:
            scene_count_raw = benchmark_cfg.get("n_scenes", 0)
        scene_count = int(scene_count_raw)
        if scene_count <= 0:
            raise ValueError(
                f"composition.scene_count or benchmark.n_scenes must be > 0, got {scene_count}."
            )

        graph_motif = graph_cfg.get("motif", "random")
        if str(graph_motif).strip().lower() == "random":
            default_motif_weights = {
                motif_name: 1.0 / len(MOTIFS) for motif_name in sorted(MOTIFS)
            }
        else:
            default_motif_weights = {parse_motif_name(graph_motif): 1.0}

        graft_count_defaults = cls._default_graft_count_probs(graph_cfg)
        identifiability_defaults = cls._default_identifiability_regime_weights(
            graph_cfg
        )
        treatment_type_defaults, outcome_type_defaults = cls._default_type_weights(
            data_cfg
        )
        continuous_scm_profile_weights_defaults = (
            cls._default_continuous_scm_profile_weights(
                data_cfg,
                data_override_cfg=override_data_cfg,
            )
        )
        binary_scm_profile_weights_defaults = cls._default_binary_scm_profile_weights(
            data_cfg,
            data_override_cfg=override_data_cfg,
        )
        observation_variants_defaults = cls._default_observation_variants(data_cfg)
        graft_safe_main_motifs = cls._resolve_graft_safe_main_motifs(graph_cfg)
        auxiliary_motif_weights = cls._resolve_auxiliary_motif_weights(
            composition_value("auxiliary_motif_weights", {}),
            graph_cfg=graph_cfg,
        )

        continuous_scm_profiles_defined = cls._resolve_continuous_scm_profile_registry(
            data_cfg,
            data_override_cfg=override_data_cfg,
        )
        continuous_scm_profile_weights_raw = composition_value(
            "continuous_scm_profile_weights",
            composition_value(
                "scm_profile_weights",
                continuous_scm_profile_weights_defaults,
            ),
        )
        continuous_scm_profile_weights = normalize_weights(
            continuous_scm_profile_weights_raw,
            field_name="composition.continuous_scm_profile_weights",
            key_transform=lambda key: str(key).strip(),
        )
        unknown_continuous_scm_profiles = sorted(
            set(continuous_scm_profile_weights) - set(continuous_scm_profiles_defined)
        )
        if unknown_continuous_scm_profiles:
            raise ValueError(
                "composition.continuous_scm_profile_weights references undefined "
                "data.continuous_scm_profiles: "
                f"{unknown_continuous_scm_profiles}"
            )
        binary_scm_profiles_defined = cls._resolve_binary_scm_profile_registry(
            data_cfg,
            data_override_cfg=override_data_cfg,
        )
        binary_scm_profile_weights = normalize_weights(
            composition_value(
                "binary_scm_profile_weights",
                binary_scm_profile_weights_defaults,
            ),
            field_name="composition.binary_scm_profile_weights",
            key_transform=lambda key: str(key).strip(),
        )
        unknown_binary_scm_profiles = sorted(
            set(binary_scm_profile_weights) - set(binary_scm_profiles_defined)
        )
        if unknown_binary_scm_profiles:
            raise ValueError(
                "composition.binary_scm_profile_weights references undefined "
                "data.binary_scm_profiles: "
                f"{unknown_binary_scm_profiles}"
            )

        available_observation_variants = cls._available_observation_variants(data_cfg)
        released_observation_variants = [
            str(name).strip()
            for name in (
                composition_value(
                    "released_observation_variants",
                    observation_variants_defaults,
                )
                or []
            )
            if str(name).strip()
        ]
        unknown_observation_variants = sorted(
            set(released_observation_variants) - set(available_observation_variants)
        )
        if unknown_observation_variants:
            raise ValueError(
                "composition.released_observation_variants references undefined "
                f"data.observation_variants entries: {unknown_observation_variants}"
            )

        nonident_cfg = ensure_plain_dict(composition_value("nonidentifiable", {}))
        realization_cfg = ensure_plain_dict(composition_value("realization", {}))
        nonident_allowlist_raw = nonident_cfg.get(
            "motif_allowlist",
            composition_value("nonidentifiable_motif_allowlist", None),
        )
        nonident_allowlist = [
            parse_motif_name(value) for value in (nonident_allowlist_raw or [])
        ]

        composition = cls(
            scene_count=scene_count,
            main_motif_weights=normalize_weights(
                composition_value("main_motif_weights", default_motif_weights),
                field_name="composition.main_motif_weights",
                key_transform=parse_motif_name,
            ),
            graft_count_probs=normalize_weights(
                composition_value("graft_count_probs", graft_count_defaults),
                field_name="composition.graft_count_probs",
                key_transform=parse_graft_count,
            ),
            identifiability_regime_weights=normalize_weights(
                composition_value(
                    "identifiability_regime_weights",
                    identifiability_defaults,
                ),
                field_name="composition.identifiability_regime_weights",
                key_transform=parse_identifiability_regime,
            ),
            treatment_type_weights=normalize_weights(
                composition_value(
                    "treatment_type_weights",
                    treatment_type_defaults,
                ),
                field_name="composition.treatment_type_weights",
                key_transform=parse_variable_type,
            ),
            outcome_type_weights=normalize_weights(
                composition_value(
                    "outcome_type_weights",
                    outcome_type_defaults,
                ),
                field_name="composition.outcome_type_weights",
                key_transform=parse_variable_type,
            ),
            continuous_scm_profile_weights=continuous_scm_profile_weights,
            binary_scm_profile_weights=binary_scm_profile_weights,
            auxiliary_motif_weights=auxiliary_motif_weights,
            released_observation_variants=released_observation_variants,
            graft_safe_main_motifs=graft_safe_main_motifs,
            nonidentifiable_motif_allowlist=nonident_allowlist,
            nonidentifiable_p_latent_xy=float(
                nonident_cfg.get(
                    "p_latent_xy",
                    composition_value("nonidentifiable_p_latent_xy", 1.0),
                )
            ),
            nonidentifiable_max_tries=int(
                nonident_cfg.get(
                    "max_tries",
                    composition_value("nonidentifiable_max_tries", 200),
                )
            ),
            realization_max_seed_restarts=int(
                realization_cfg.get(
                    "max_seed_restarts",
                    composition_value("realization_max_seed_restarts", 0),
                )
            ),
            realization_seed_restart_stride=int(
                realization_cfg.get(
                    "seed_restart_stride",
                    composition_value("realization_seed_restart_stride", 1_000_003),
                )
            ),
            blueprint_seed_offset=int(composition_value("blueprint_seed_offset", 0)),
        )
        composition.validate_main_motif_slot_sanity()
        return composition

    @staticmethod
    def _default_graft_count_probs(graph_cfg: Dict[str, Any]) -> Dict[int, float]:
        grafting_cfg = ensure_plain_dict(graph_cfg.get("grafting"))
        mode = str(grafting_cfg.get("mode", "none") or "none").strip().lower()
        aux_graft_count = int(grafting_cfg.get("aux_graft_count", 0) or 0)
        if aux_graft_count <= 0 or mode in {"none", "off", "disabled"}:
            return {0: 1.0}
        if mode in {"fixed", "always", "anchor_auxiliary_graft", "anchor_graft"}:
            return {aux_graft_count: 1.0}
        return {k: 1.0 for k in range(0, aux_graft_count + 1)}

    @staticmethod
    def _default_identifiability_regime_weights(
        graph_cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        require_identifiable = bool(graph_cfg.get("require_identifiable", True))
        if require_identifiable:
            return {IDENTIFIABILITY_IDENTIFIABLE: 1.0}
        return {IDENTIFIABILITY_IDENTIFIABLE: 1.0}

    @staticmethod
    def _default_type_weights(
        data_cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        treatment_weights = {"continuous": 1.0}
        outcome_weights = {"continuous": 1.0}
        if data_cfg.get("force_treatment_binary") is True:
            treatment_weights = {"binary": 1.0}
        if data_cfg.get("force_outcome_continuous") is True:
            outcome_weights = {"continuous": 1.0}
        return treatment_weights, outcome_weights

    @staticmethod
    def _resolve_continuous_scm_profile_registry(
        data_cfg: Dict[str, Any],
        data_override_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        data_override_cfg = ensure_plain_dict(data_override_cfg)
        if "continuous_scm_profiles" in data_override_cfg:
            raw_profiles = ensure_plain_dict(
                data_override_cfg.get("continuous_scm_profiles")
            )
        elif "continuous_scm_profiles" in data_cfg:
            raw_profiles = ensure_plain_dict(data_cfg.get("continuous_scm_profiles"))
        elif "scm_profiles" in data_override_cfg:
            raw_profiles = ensure_plain_dict(data_override_cfg.get("scm_profiles"))
        else:
            raw_profiles = ensure_plain_dict(data_cfg.get("scm_profiles"))
        if raw_profiles:
            return {
                str(name).strip(): ensure_plain_dict(value)
                for name, value in raw_profiles.items()
            }

        mechanism_cfg = data_cfg.get("mechanism_config")
        if mechanism_cfg is None:
            return {"default": {"mechanism_config": None}}
        return {"default": {"mechanism_config": ensure_plain_dict(mechanism_cfg)}}

    @classmethod
    def _default_continuous_scm_profile_weights(
        cls,
        data_cfg: Dict[str, Any],
        *,
        data_override_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        profiles = cls._resolve_continuous_scm_profile_registry(
            data_cfg,
            data_override_cfg=data_override_cfg,
        )
        return {name: 1.0 / len(profiles) for name in profiles}

    @staticmethod
    def _resolve_binary_scm_profile_registry(
        data_cfg: Dict[str, Any],
        data_override_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        data_override_cfg = ensure_plain_dict(data_override_cfg)
        if "binary_scm_profiles" in data_override_cfg:
            raw_profiles = ensure_plain_dict(
                data_override_cfg.get("binary_scm_profiles")
            )
        else:
            raw_profiles = ensure_plain_dict(data_cfg.get("binary_scm_profiles"))
        if raw_profiles:
            return {
                str(name).strip(): ensure_plain_dict(value)
                for name, value in raw_profiles.items()
            }

        binary_mechanism_cfg = data_cfg.get("binary_mechanism_config")
        if binary_mechanism_cfg is None:
            return {"default": {"binary_mechanism_config": None}}
        return {
            "default": {
                "binary_mechanism_config": ensure_plain_dict(binary_mechanism_cfg)
            }
        }

    @classmethod
    def _default_binary_scm_profile_weights(
        cls,
        data_cfg: Dict[str, Any],
        *,
        data_override_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        profiles = cls._resolve_binary_scm_profile_registry(
            data_cfg,
            data_override_cfg=data_override_cfg,
        )
        return {name: 1.0 / len(profiles) for name in profiles}

    @staticmethod
    def _default_observation_variants(data_cfg: Dict[str, Any]) -> List[str]:
        variants = CompositionConfig._available_observation_variants(data_cfg)
        if variants:
            return variants
        return ["default"]

    @staticmethod
    def _available_observation_variants(data_cfg: Dict[str, Any]) -> List[str]:
        observation_variants = ensure_plain_dict(data_cfg.get("observation_variants"))
        variants = ensure_plain_dict(observation_variants.get("variants"))
        if variants:
            return [str(name).strip() for name in variants.keys() if str(name).strip()]
        return []

    @staticmethod
    def _resolve_graft_safe_main_motifs(graph_cfg: Dict[str, Any]) -> List[str]:
        grafting_cfg = ensure_plain_dict(graph_cfg.get("grafting"))
        if not bool(grafting_cfg.get("main_graph_restrict_when_grafting", True)):
            return []
        main_graph_motifs = coerce_list(grafting_cfg.get("main_graph_motifs"))
        resolved = cg._resolve_main_graph_motif_pool(
            main_graph_motifs=main_graph_motifs or None
        )
        return [str(motif.value) for motif in resolved]

    @staticmethod
    def _resolve_auxiliary_motif_weights(
        raw: Any,
        *,
        graph_cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        raw = ensure_plain_dict(raw)
        if not raw:
            return {}

        grafting_cfg = ensure_plain_dict(graph_cfg.get("grafting"))
        allowed_pool = cg._resolve_auxiliary_motif_pool(
            restrict_to_basic=bool(
                grafting_cfg.get("auxiliary_restrict_basic_motifs", True)
            ),
            custom_motifs=(
                coerce_list(grafting_cfg.get("auxiliary_custom_motifs")) or None
            ),
        )
        allowed_names = {motif.value for motif in allowed_pool}

        resolved: Dict[str, float] = {}
        for key, value in raw.items():
            motif_name = parse_motif_name(key)
            if motif_name not in allowed_names:
                raise ValueError(
                    "composition.auxiliary_motif_weights references motifs outside the "
                    f"resolved auxiliary motif pool: {motif_name!r}"
                )
            weight = float(value)
            if weight < 0.0:
                raise ValueError(
                    "composition.auxiliary_motif_weights"
                    f"[{motif_name!r}] must be >= 0, got {weight}."
                )
            resolved[motif_name] = weight
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_count": int(self.scene_count),
            "main_motif_weights": json_safe(self.main_motif_weights),
            "graft_count_probs": {
                str(count): float(weight)
                for count, weight in self.graft_count_probs.items()
            },
            "identifiability_regime_weights": json_safe(
                self.identifiability_regime_weights
            ),
            "treatment_type_weights": json_safe(self.treatment_type_weights),
            "outcome_type_weights": json_safe(self.outcome_type_weights),
            "continuous_scm_profile_weights": json_safe(
                self.continuous_scm_profile_weights
            ),
            "binary_scm_profile_weights": json_safe(self.binary_scm_profile_weights),
            "scm_profile_weights": json_safe(self.continuous_scm_profile_weights),
            "auxiliary_motif_weights": json_safe(self.auxiliary_motif_weights),
            "released_observation_variants": list(self.released_observation_variants),
            "graft_safe_main_motifs": list(self.graft_safe_main_motifs),
            "nonidentifiable": {
                "motif_allowlist": list(self.nonidentifiable_motif_allowlist),
                "p_latent_xy": float(self.nonidentifiable_p_latent_xy),
                "max_tries": int(self.nonidentifiable_max_tries),
            },
            "realization": {
                "max_seed_restarts": int(self.realization_max_seed_restarts),
                "seed_restart_stride": int(self.realization_seed_restart_stride),
            },
            "blueprint_seed_offset": int(self.blueprint_seed_offset),
        }


@dataclass
class SceneBlueprint:
    """Quota-driven desired scene instance before graph realization."""

    scene_index: int
    scene_id: str
    scene_seed: int
    main_motif: str
    graft_count: int
    identifiability_regime: str
    treatment_type: str
    outcome_type: str
    continuous_scm_profile: str
    binary_scm_profile: str
    released_observation_variants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scm_profile(self) -> str:
        """Backward-compatible alias for the continuous SCM profile."""
        return self.continuous_scm_profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_index": int(self.scene_index),
            "scene_id": str(self.scene_id),
            "scene_seed": int(self.scene_seed),
            "main_motif": str(self.main_motif),
            "graft_count": int(self.graft_count),
            "identifiability_regime": str(self.identifiability_regime),
            "treatment_type": str(self.treatment_type),
            "outcome_type": str(self.outcome_type),
            "continuous_scm_profile": str(self.continuous_scm_profile),
            "binary_scm_profile": str(self.binary_scm_profile),
            "scm_profile": str(self.continuous_scm_profile),
            "released_observation_variants": list(self.released_observation_variants),
            "metadata": json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SceneBlueprint":
        raw = ensure_plain_dict(raw)
        legacy_profile = raw.get("scm_profile", "default")
        return cls(
            scene_index=int(raw.get("scene_index", 0)),
            scene_id=str(raw.get("scene_id", "")),
            scene_seed=int(raw.get("scene_seed", 0)),
            main_motif=parse_motif_name(raw.get("main_motif")),
            graft_count=parse_graft_count(raw.get("graft_count", 0)),
            identifiability_regime=parse_identifiability_regime(
                raw.get("identifiability_regime", IDENTIFIABILITY_IDENTIFIABLE)
            ),
            treatment_type=parse_variable_type(raw.get("treatment_type", "continuous")),
            outcome_type=parse_variable_type(raw.get("outcome_type", "continuous")),
            continuous_scm_profile=str(
                raw.get("continuous_scm_profile", legacy_profile)
            ),
            binary_scm_profile=str(raw.get("binary_scm_profile", "default")),
            released_observation_variants=[
                str(name).strip()
                for name in (raw.get("released_observation_variants") or [])
                if str(name).strip()
            ],
            metadata=ensure_plain_dict(raw.get("metadata")),
        )


def compile_scene_blueprints(
    *,
    composition: CompositionConfig,
    base_seed: int,
    scene_id_fn,
    scene_seed_fn,
) -> List[SceneBlueprint]:
    """Compile exact-marginal blueprint rows from composition quotas."""
    composition.validate_main_motif_slot_sanity()
    rng = random.Random(int(base_seed) + int(composition.blueprint_seed_offset))
    scene_count = composition.scene_count

    regime_counts = largest_remainder_counts(
        composition.identifiability_regime_weights,
        scene_count,
    )
    regimes = expand_counts(regime_counts, rng)
    graft_counts = expand_counts(
        largest_remainder_counts(composition.graft_count_probs, scene_count),
        rng,
    )

    motif_counts = largest_remainder_counts(composition.main_motif_weights, scene_count)
    motifs = _assign_motifs_to_blueprint_slots(
        motif_counts=motif_counts,
        regimes=regimes,
        graft_counts=graft_counts,
        graft_safe_main_motifs=composition.graft_safe_main_motifs,
        nonidentifiable_allowlist=composition.nonidentifiable_motif_allowlist,
        rng=rng,
    )
    treatment_types = expand_counts(
        largest_remainder_counts(composition.treatment_type_weights, scene_count),
        rng,
    )
    outcome_types = expand_counts(
        largest_remainder_counts(composition.outcome_type_weights, scene_count),
        rng,
    )
    continuous_scm_profiles = expand_counts(
        largest_remainder_counts(
            composition.continuous_scm_profile_weights,
            scene_count,
        ),
        rng,
    )
    binary_scm_profiles = expand_counts(
        largest_remainder_counts(
            composition.binary_scm_profile_weights,
            scene_count,
        ),
        rng,
    )

    blueprints: List[SceneBlueprint] = []
    for scene_index in range(1, scene_count + 1):
        scene_seed = int(scene_seed_fn(base_seed, scene_index))
        blueprints.append(
            SceneBlueprint(
                scene_index=scene_index,
                scene_id=str(scene_id_fn(scene_index)),
                scene_seed=scene_seed,
                main_motif=str(motifs[scene_index - 1]),
                graft_count=int(graft_counts[scene_index - 1]),
                identifiability_regime=str(regimes[scene_index - 1]),
                treatment_type=str(treatment_types[scene_index - 1]),
                outcome_type=str(outcome_types[scene_index - 1]),
                continuous_scm_profile=str(continuous_scm_profiles[scene_index - 1]),
                binary_scm_profile=str(binary_scm_profiles[scene_index - 1]),
                released_observation_variants=list(
                    composition.released_observation_variants
                ),
                metadata={},
            )
        )
    return blueprints


def _assign_motifs_to_blueprint_slots(
    *,
    motif_counts: Dict[str, int],
    regimes: List[str],
    graft_counts: List[int],
    graft_safe_main_motifs: List[str],
    nonidentifiable_allowlist: List[str],
    rng: random.Random,
) -> List[str]:
    """Assign motifs while respecting slot-level structural constraints."""
    remaining = Counter({str(key): int(value) for key, value in motif_counts.items()})
    all_motifs = set(remaining.keys())
    graft_safe_set = set(graft_safe_main_motifs or [])
    nonident_set = set(nonidentifiable_allowlist or [])

    def allowed_base(idx: int) -> List[str]:
        allowed = set(all_motifs)
        if int(graft_counts[idx]) > 0 and graft_safe_set:
            allowed &= graft_safe_set
        if str(regimes[idx]) == IDENTIFIABILITY_NONIDENTIFIABLE:
            allowed = {
                motif_name
                for motif_name in allowed
                if motif_has_treatment_outcome_path(motif_name)
            }
        else:
            allowed -= STRUCTURALLY_NONIDENTIFIABLE_MAIN_MOTIFS
        return sorted(allowed)

    allowed_by_slot = {idx: allowed_base(idx) for idx in range(len(regimes))}
    for idx, allowed in allowed_by_slot.items():
        if not allowed:
            raise RuntimeError(
                "No motifs can satisfy blueprint slot constraints. "
                f"Slot {idx} requires regime={regimes[idx]!r}, graft_count={graft_counts[idx]!r}."
            )

    for motif_name, count in remaining.items():
        compatible_slots = [
            idx for idx, allowed in allowed_by_slot.items() if motif_name in allowed
        ]
        if len(compatible_slots) < int(count):
            raise RuntimeError(
                "Motif counts are incompatible with blueprint slot constraints. "
                f"Motif {motif_name!r} needs {int(count)} slot(s) but only "
                f"{len(compatible_slots)} compatible slot(s) exist."
            )

    flow_graph = nx.DiGraph()
    source = ("source",)
    sink = ("sink",)
    flow_graph.add_node(source, demand=-len(regimes))
    flow_graph.add_node(sink, demand=len(regimes))

    motif_nodes = [("motif", motif_name) for motif_name in remaining]
    rng.shuffle(motif_nodes)
    for motif_node in motif_nodes:
        flow_graph.add_node(motif_node, demand=0)
        flow_graph.add_edge(
            source,
            motif_node,
            capacity=int(remaining[motif_node[1]]),
            weight=0,
        )

    slot_indices = list(range(len(regimes)))
    rng.shuffle(slot_indices)
    for idx in slot_indices:
        slot_node = ("slot", int(idx))
        flow_graph.add_node(slot_node, demand=0)
        flow_graph.add_edge(slot_node, sink, capacity=1, weight=0)
        allowed = list(allowed_by_slot[idx])
        rng.shuffle(allowed)
        for motif_name in allowed:
            cost = 0
            if (
                str(regimes[idx]) == IDENTIFIABILITY_NONIDENTIFIABLE
                and nonident_set
                and motif_name not in nonident_set
            ):
                cost = 1
            flow_graph.add_edge(
                ("motif", motif_name), slot_node, capacity=1, weight=cost
            )

    try:
        flow_dict = nx.min_cost_flow(flow_graph)
    except Exception as exc:
        raise RuntimeError(
            "Failed to find a feasible motif assignment for the requested composition."
        ) from exc

    assignments: List[Optional[str]] = [None] * len(regimes)
    for motif_node in motif_nodes:
        motif_name = motif_node[1]
        for target_node, value in flow_dict.get(motif_node, {}).items():
            if (
                isinstance(target_node, tuple)
                and len(target_node) == 2
                and target_node[0] == "slot"
                and int(value) > 0
            ):
                assignments[int(target_node[1])] = motif_name

    if any(value is None for value in assignments):
        raise RuntimeError("Motif assignment flow left one or more slots unassigned.")
    return [str(value) for value in assignments]


def count_blueprint_axis(
    blueprints: Iterable[SceneBlueprint],
    attr_name: str,
) -> Dict[str, int]:
    """Count blueprint rows by one top-level attribute."""
    counter = Counter()
    for blueprint in blueprints:
        counter[str(getattr(blueprint, attr_name))] += 1
    return dict(sorted(counter.items()))


def summarize_blueprint_requests(
    blueprints: Iterable[SceneBlueprint],
) -> Dict[str, Dict[str, int]]:
    """Summarize requested composition axes from blueprint rows."""
    row_list = list(blueprints)
    counter = Counter()
    for blueprint in row_list:
        for view_name in blueprint.released_observation_variants:
            counter[str(view_name)] += 1

    return {
        "main_motif": count_blueprint_axis(row_list, "main_motif"),
        "graft_count": count_blueprint_axis(row_list, "graft_count"),
        "identifiability_regime": count_blueprint_axis(
            row_list, "identifiability_regime"
        ),
        "treatment_type": count_blueprint_axis(row_list, "treatment_type"),
        "outcome_type": count_blueprint_axis(row_list, "outcome_type"),
        "continuous_scm_profile": count_blueprint_axis(
            row_list, "continuous_scm_profile"
        ),
        "binary_scm_profile": count_blueprint_axis(row_list, "binary_scm_profile"),
        "scm_profile": count_blueprint_axis(row_list, "scm_profile"),
        "observation_variant": dict(sorted(counter.items())),
    }


def build_blueprint_summary(
    blueprints: List[SceneBlueprint],
    *,
    composition: CompositionConfig,
) -> Dict[str, Any]:
    """Build a compact paper-facing summary for blueprint rows."""
    return {
        "scene_count": len(blueprints),
        "requested": summarize_blueprint_requests(blueprints),
        "composition": composition.to_dict(),
    }


def _count_string_values(
    records: Iterable[Dict[str, Any]],
    field_name: str,
) -> Dict[str, int]:
    """Count one scalar string-like field across plain dict records."""
    counter = Counter()
    for record in records:
        value = record.get(field_name)
        if value is None:
            continue
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _count_list_values(
    records: Iterable[Dict[str, Any]],
    field_name: str,
) -> Dict[str, int]:
    """Count one list-like field across plain dict records."""
    counter = Counter()
    for record in records:
        for value in record.get(field_name) or []:
            counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _extract_manifest_payloads(
    rows: Iterable[Dict[str, Any]],
    payload_key: str,
) -> List[Dict[str, Any]]:
    """Collect one nested payload from manifest rows when present."""
    payloads: List[Dict[str, Any]] = []
    for row in rows:
        payload = ensure_plain_dict(ensure_plain(row.get(payload_key)))
        if payload:
            payloads.append(payload)
    return payloads


def _summarize_requested_blueprint_payloads(
    payloads: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Summarize requested blueprint axes from nested blueprint payloads."""
    records = list(payloads)
    return {
        "main_motif": _count_string_values(records, "main_motif"),
        "graft_count": _count_string_values(records, "graft_count"),
        "identifiability_regime": _count_string_values(
            records, "identifiability_regime"
        ),
        "treatment_type": _count_string_values(records, "treatment_type"),
        "outcome_type": _count_string_values(records, "outcome_type"),
        "continuous_scm_profile": _count_string_values(
            records, "continuous_scm_profile"
        ),
        "binary_scm_profile": _count_string_values(records, "binary_scm_profile"),
        "scm_profile": _count_string_values(records, "scm_profile"),
        "observation_variant": _count_list_values(
            records, "released_observation_variants"
        ),
    }


def _summarize_realized_payloads(
    payloads: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Summarize realized axes from nested realized payloads."""
    records = list(payloads)
    return {
        "main_motif": _count_string_values(records, "main_motif"),
        "graft_count": _count_string_values(records, "applied_grafts"),
        "identifiability_regime": _count_string_values(
            records, "identifiability_regime"
        ),
        "treatment_type": _count_string_values(records, "treatment_type"),
        "outcome_type": _count_string_values(records, "outcome_type"),
        "continuous_scm_profile": _count_string_values(
            records, "continuous_scm_profile"
        ),
        "binary_scm_profile": _count_string_values(records, "binary_scm_profile"),
        "scm_profile": _count_string_values(records, "scm_profile"),
        "observation_variant": _count_list_values(
            records, "released_observation_variants"
        ),
        "n_nodes": _count_string_values(records, "n_nodes"),
    }


def summarize_runnable_manifest_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    completed_scene_ids: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Summarize requested/realized composition from runnable manifest rows."""
    row_list = [ensure_plain_dict(row) for row in rows]
    if not row_list:
        return None

    if not any(("blueprint" in row or "realized" in row) for row in row_list):
        return None

    completed_set = {
        str(scene_id).strip()
        for scene_id in (completed_scene_ids or [])
        if str(scene_id).strip()
    }
    completed_rows = [
        row for row in row_list if str(row.get("scene_id", "")).strip() in completed_set
    ]

    blueprint_payloads = _extract_manifest_payloads(row_list, "blueprint")
    realized_payloads = _extract_manifest_payloads(row_list, "realized")
    completed_realized_payloads = _extract_manifest_payloads(completed_rows, "realized")

    return {
        "n_manifest_rows": int(len(row_list)),
        "n_rows_with_blueprint": int(len(blueprint_payloads)),
        "n_rows_with_realized": int(len(realized_payloads)),
        "n_completed_scenes": int(len(completed_set)),
        "requested": _summarize_requested_blueprint_payloads(blueprint_payloads),
        "realized_manifest": _summarize_realized_payloads(realized_payloads),
        "completed": _summarize_realized_payloads(completed_realized_payloads),
    }


def load_blueprint_manifest(path: Path) -> List[SceneBlueprint]:
    """Load a blueprint manifest from JSONL or JSON."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[SceneBlueprint] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(SceneBlueprint.from_dict(json.loads(line)))
        return rows

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [SceneBlueprint.from_dict(row) for row in payload]
        raise ValueError("Blueprint JSON payload must be a list of rows.")

    raise ValueError(f"Unsupported blueprint format: {path}. Use .jsonl or .json.")


def load_runnable_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load a runnable manifest from JSONL, JSON, or parquet."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(ensure_plain_dict(json.loads(line)))
        return rows

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = ensure_plain(json.load(handle))
        if isinstance(payload, list):
            return [ensure_plain_dict(row) for row in payload]
        return [ensure_plain_dict(payload)]

    if suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        return [ensure_plain_dict(row) for row in frame.to_dict(orient="records")]

    raise ValueError(
        f"Unsupported runnable manifest format: {path}. Use .jsonl, .json, or .parquet."
    )


def write_blueprint_manifest(
    *,
    blueprints: List[SceneBlueprint],
    output_path: Path,
) -> None:
    """Write a blueprint manifest as JSONL or JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for blueprint in blueprints:
                handle.write(json.dumps(blueprint.to_dict(), ensure_ascii=True) + "\n")
        tmp_path.replace(output_path)
        return

    if suffix == ".json":
        atomic_write_json(output_path, [bp.to_dict() for bp in blueprints])
        return

    raise ValueError(
        f"Unsupported blueprint format: {output_path}. Use .jsonl or .json."
    )


def write_resolved_blueprint_config(
    *,
    path: Path,
    cfg: Any,
    composition: CompositionConfig,
) -> None:
    """Write the layered config plus resolved composition summary."""
    resolved = copy.deepcopy(json_safe(cfg))
    if not isinstance(resolved, dict):
        resolved = {}
    resolved["composition"] = composition.to_dict()
    write_text_atomic(
        path,
        OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True),
    )


def resolve_scm_profile_registry(data_cfg: Any) -> Dict[str, Dict[str, Any]]:
    """Resolve legacy named SCM profiles from the data config."""
    return resolve_continuous_scm_profile_registry(data_cfg)


def resolve_continuous_scm_profile_registry(data_cfg: Any) -> Dict[str, Dict[str, Any]]:
    """Resolve named continuous SCM profiles from the data config."""
    return CompositionConfig._resolve_continuous_scm_profile_registry(
        ensure_plain_dict(data_cfg)
    )


def resolve_binary_scm_profile_registry(data_cfg: Any) -> Dict[str, Dict[str, Any]]:
    """Resolve named binary SCM profiles from the data config."""
    return CompositionConfig._resolve_binary_scm_profile_registry(
        ensure_plain_dict(data_cfg)
    )


def resolve_continuous_scm_profile_bundle(
    *,
    profile_name: str,
    data_cfg: Any,
) -> Tuple[Dict[str, Any], cd.MechanismConfig]:
    """Resolve one named continuous SCM profile into a mechanism config."""
    data_cfg = ensure_plain_dict(data_cfg)
    registry = resolve_continuous_scm_profile_registry(data_cfg)
    profile_key = str(profile_name).strip()
    if profile_key not in registry:
        raise ValueError(
            "Unknown continuous SCM profile "
            f"{profile_name!r}. Available profiles: {sorted(registry)}"
        )

    profile_payload = ensure_plain_dict(registry[profile_key])

    base_mechanism_raw = ensure_plain(data_cfg.get("mechanism_config"))
    base_mechanism_cfg = (
        ensure_plain_dict(base_mechanism_raw) if base_mechanism_raw is not None else {}
    )
    profile_mechanism_raw = ensure_plain(profile_payload.get("mechanism_config"))
    profile_mechanism_cfg = (
        ensure_plain_dict(profile_mechanism_raw)
        if profile_mechanism_raw is not None
        else {}
    )
    merged_mechanism_cfg = deep_merge_dicts(
        base_mechanism_cfg,
        profile_mechanism_cfg,
    )
    return profile_payload, cd.MechanismConfig(**merged_mechanism_cfg)


def resolve_binary_scm_profile_bundle(
    *,
    profile_name: str,
    data_cfg: Any,
) -> Tuple[Dict[str, Any], cd.BinaryMechanismConfig]:
    """Resolve one named binary SCM profile into a binary mechanism config."""
    data_cfg = ensure_plain_dict(data_cfg)
    registry = resolve_binary_scm_profile_registry(data_cfg)
    profile_key = str(profile_name).strip()
    if profile_key == "logistic" and "default" in registry:
        profile_key = "default"
    if profile_key not in registry:
        raise ValueError(
            f"Unknown binary SCM profile {profile_name!r}. "
            f"Available profiles: {sorted(registry)}"
        )

    profile_payload = ensure_plain_dict(registry[profile_key])
    base_mechanism_raw = ensure_plain(data_cfg.get("binary_mechanism_config"))
    base_mechanism_cfg = (
        ensure_plain_dict(base_mechanism_raw) if base_mechanism_raw is not None else {}
    )
    profile_mechanism_raw = ensure_plain(profile_payload.get("binary_mechanism_config"))
    profile_mechanism_cfg = (
        ensure_plain_dict(profile_mechanism_raw)
        if profile_mechanism_raw is not None
        else {}
    )
    merged_mechanism_cfg = deep_merge_dicts(
        base_mechanism_cfg,
        profile_mechanism_cfg,
    )
    return profile_payload, cd.BinaryMechanismConfig(**merged_mechanism_cfg)


def resolve_blueprint_node_types(
    *,
    blueprint: SceneBlueprint,
    sg: cg.SampledGraph,
    data_cfg: Any,
) -> Dict[str, str]:
    """Resolve explicit node-type overrides for one realized blueprint row."""
    data_cfg = ensure_plain_dict(data_cfg)
    raw_node_types = ensure_plain(data_cfg.get("node_types"))
    node_types = ensure_plain_dict(raw_node_types) if raw_node_types is not None else {}
    resolved = {
        str(node_id): parse_variable_type(node_type)
        for node_id, node_type in node_types.items()
    }
    resolved[str(sg.treatment)] = str(blueprint.treatment_type)
    resolved[str(sg.outcome)] = str(blueprint.outcome_type)
    return resolved


def build_datagen_spec_for_blueprint(
    *,
    blueprint: SceneBlueprint,
    sg: cg.SampledGraph,
    data_cfg: Any,
    seed: Optional[int] = None,
) -> cd.DataGeneratorSpec:
    """Resolve a self-contained data-generator spec for one blueprint row."""
    _, mech_config = resolve_continuous_scm_profile_bundle(
        profile_name=blueprint.continuous_scm_profile,
        data_cfg=data_cfg,
    )
    _, binary_mech_config = resolve_binary_scm_profile_bundle(
        profile_name=blueprint.binary_scm_profile,
        data_cfg=data_cfg,
    )
    return cd.DataGeneratorSpec(
        node_types=resolve_blueprint_node_types(
            blueprint=blueprint,
            sg=sg,
            data_cfg=data_cfg,
        ),
        force_treatment_binary=(blueprint.treatment_type == "binary"),
        force_outcome_continuous=(blueprint.outcome_type == "continuous"),
        seed=int(blueprint.scene_seed if seed is None else seed),
        scm_profile=str(blueprint.continuous_scm_profile),
        continuous_scm_profile=str(blueprint.continuous_scm_profile),
        binary_scm_profile=str(blueprint.binary_scm_profile),
        mech_config=mech_config,
        binary_mech_config=binary_mech_config,
    )


def resolve_released_observation_configs(
    *,
    blueprint: SceneBlueprint,
    observation_config: cd.ObservationConfig,
    observation_variants: Dict[str, cd.ObservationConfig],
) -> Tuple[cd.ObservationConfig, Dict[str, cd.ObservationConfig]]:
    """Filter observation settings down to the released measurement views."""
    base_config = cd.ObservationConfig.from_dict(observation_config.to_dict())
    available_variants = {
        str(name): cd.ObservationConfig.from_dict(cfg.to_dict())
        for name, cfg in dict(observation_variants or {}).items()
    }
    requested_views = [
        str(name).strip() for name in blueprint.released_observation_variants
    ]
    if not requested_views:
        return base_config, available_variants

    selected_variants: Dict[str, cd.ObservationConfig] = {}
    include_default = False
    for view_name in requested_views:
        if not view_name:
            continue
        if view_name == "default":
            include_default = True
            continue
        if view_name not in available_variants:
            raise ValueError(
                "Blueprint released_observation_variants references unknown "
                f"observation variant {view_name!r}. Available variants: {sorted(available_variants)}"
            )
        selected_variants[view_name] = available_variants[view_name]

    if include_default and selected_variants:
        selected_variants = {"default": base_config} | selected_variants
    if include_default and not selected_variants:
        return base_config, {}
    return base_config, selected_variants


def attach_blueprint_metadata(
    *,
    sg: cg.SampledGraph,
    blueprint: SceneBlueprint,
    fill_attempts: int,
    realization_seed: int,
    seed_restart_index: int,
) -> cg.SampledGraph:
    """Annotate a realized graph with requested and realized blueprint metadata."""
    graph_meta = copy.deepcopy(sg.meta) if isinstance(sg.meta, dict) else {}
    augmentation = ensure_plain_dict(graph_meta.get("augmentation"))
    identifiable = sg.is_identifiable
    graph_meta["blueprint"] = blueprint.to_dict()
    graph_meta["blueprint_realization"] = {
        "main_motif": str(sg.motif),
        "n_nodes": int(len(sg.graph.nodes())),
        "applied_grafts": int(augmentation.get("applied_grafts", 0)),
        "identifiable": None if identifiable is None else bool(identifiable),
        "fill_attempts": int(fill_attempts),
        "scene_seed": int(realization_seed),
        "seed_restart_index": int(seed_restart_index),
    }
    sg.meta = graph_meta
    return sg


def build_blueprint_realization_record(
    *,
    blueprint: SceneBlueprint,
    sg: cg.SampledGraph,
    datagen_spec: Optional[cd.DataGeneratorSpec] = None,
) -> Dict[str, Any]:
    """Build a compact requested-vs-realized record for one filled blueprint row."""
    graph_meta = ensure_plain_dict(sg.meta)
    realization_meta = ensure_plain_dict(graph_meta.get("blueprint_realization"))
    augmentation = ensure_plain_dict(graph_meta.get("augmentation"))
    identifiable = sg.is_identifiable

    requested_types = dict(datagen_spec.node_types or {}) if datagen_spec else {}
    treatment_type = requested_types.get(
        str(sg.treatment), str(blueprint.treatment_type)
    )
    outcome_type = requested_types.get(str(sg.outcome), str(blueprint.outcome_type))

    identifiability_regime = "unknown"
    if identifiable is not None:
        identifiability_regime = (
            IDENTIFIABILITY_IDENTIFIABLE
            if bool(identifiable)
            else IDENTIFIABILITY_NONIDENTIFIABLE
        )

    return {
        "main_motif": str(sg.motif),
        "n_nodes": int(len(sg.graph.nodes())),
        "applied_grafts": int(augmentation.get("applied_grafts", 0)),
        "identifiable": None if identifiable is None else bool(identifiable),
        "identifiability_regime": identifiability_regime,
        "treatment": str(sg.treatment),
        "outcome": str(sg.outcome),
        "treatment_type": str(treatment_type),
        "outcome_type": str(outcome_type),
        "continuous_scm_profile": str(
            datagen_spec.continuous_scm_profile
            if datagen_spec and datagen_spec.continuous_scm_profile
            else blueprint.continuous_scm_profile
        ),
        "binary_scm_profile": str(
            datagen_spec.binary_scm_profile
            if datagen_spec and datagen_spec.binary_scm_profile
            else blueprint.binary_scm_profile
        ),
        "scm_profile": str(
            datagen_spec.scm_profile
            if datagen_spec and datagen_spec.scm_profile
            else blueprint.continuous_scm_profile
        ),
        "scene_seed": int(realization_meta.get("scene_seed", blueprint.scene_seed)),
        "seed_restart_index": int(realization_meta.get("seed_restart_index", 0)),
        "released_observation_variants": list(blueprint.released_observation_variants),
        "fill_attempts": int(realization_meta.get("fill_attempts", 1)),
    }


def resolve_blueprint_graph_sampling_kwargs(
    *,
    graph_cfg: Any,
    composition: CompositionConfig,
    blueprint: SceneBlueprint,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Translate one blueprint row plus config defaults into sample_graph kwargs."""
    graph_cfg = ensure_plain_dict(graph_cfg)
    grafting_cfg = ensure_plain_dict(graph_cfg.get("grafting"))
    return {
        "motif": blueprint.main_motif,
        "n_nodes": int(graph_cfg.get("n_nodes", 1) or 1),
        "p_extra_edge": float(graph_cfg.get("p_extra_edge", 0.2)),
        "ensure_connected": bool(graph_cfg.get("ensure_connected", False)),
        "connect_isolates": bool(graph_cfg.get("connect_isolates", True)),
        "seed": int(blueprint.scene_seed if seed is None else seed),
        "augmentation_mode": "none" if blueprint.graft_count <= 0 else "fixed",
        "aux_graft_count": int(blueprint.graft_count),
        "main_graph_restrict_when_grafting": bool(
            grafting_cfg.get("main_graph_restrict_when_grafting", True)
        ),
        "main_graph_motifs": coerce_list(grafting_cfg.get("main_graph_motifs")),
        "aux_restrict_basic_motifs": bool(
            grafting_cfg.get("auxiliary_restrict_basic_motifs", True)
        ),
        "aux_custom_motifs": coerce_list(grafting_cfg.get("auxiliary_custom_motifs")),
        "aux_custom_motif_weights": dict(composition.auxiliary_motif_weights) or None,
        "aux_allow_treatment_outcome_anchor": bool(
            grafting_cfg.get("allow_treatment_outcome_anchor", False)
        ),
        "aux_preserve_treatment_outcome": bool(
            grafting_cfg.get("preserve_treatment_outcome", True)
        ),
        "aux_max_retries_per_graft": int(grafting_cfg.get("max_retries_per_graft", 25)),
        "aux_require_all_grafts": bool(grafting_cfg.get("require_all_grafts", False)),
    }


def sample_graph_for_blueprint(
    *,
    blueprint: SceneBlueprint,
    composition: CompositionConfig,
    graph_cfg: Any,
) -> cg.SampledGraph:
    """Sample one graph that satisfies the blueprint's structural regime."""
    graph_cfg = ensure_plain_dict(graph_cfg)
    restart_count = max(0, int(composition.realization_max_seed_restarts))
    seed_stride = int(composition.realization_seed_restart_stride)
    last_error: Optional[Exception] = None
    total_fill_attempts = 0

    for seed_restart_index in range(restart_count + 1):
        realization_seed = offset_random_seed(
            blueprint.scene_seed,
            int(seed_restart_index) * seed_stride,
        )
        kwargs = resolve_blueprint_graph_sampling_kwargs(
            graph_cfg=graph_cfg,
            composition=composition,
            blueprint=blueprint,
            seed=realization_seed,
        )

        try:
            if blueprint.identifiability_regime == IDENTIFIABILITY_IDENTIFIABLE:
                graph = cg.sample_graph(
                    **kwargs,
                    p_latent_xy=float(graph_cfg.get("p_latent_xy", 0.0)),
                    require_identifiable=True,
                    max_tries=max(50, composition.nonidentifiable_max_tries),
                )
                return attach_blueprint_metadata(
                    sg=graph,
                    blueprint=blueprint,
                    fill_attempts=total_fill_attempts + 1,
                    realization_seed=realization_seed,
                    seed_restart_index=seed_restart_index,
                )

            last_graph: Optional[cg.SampledGraph] = None
            for offset in range(max(1, composition.nonidentifiable_max_tries)):
                candidate = cg.sample_graph(
                    **(kwargs | {"seed": realization_seed + offset}),
                    p_latent_xy=float(composition.nonidentifiable_p_latent_xy),
                    require_identifiable=False,
                )
                last_graph = candidate
                total_fill_attempts += 1
                if candidate.is_identifiable is False:
                    return attach_blueprint_metadata(
                        sg=candidate,
                        blueprint=blueprint,
                        fill_attempts=total_fill_attempts,
                        realization_seed=realization_seed + offset,
                        seed_restart_index=seed_restart_index,
                    )

            last_error = RuntimeError(
                "Failed to realize a nonidentifiable graph within "
                f"{composition.nonidentifiable_max_tries} attempts for seed block "
                f"{seed_restart_index}. Last sampled identifiability="
                f"{None if last_graph is None else last_graph.is_identifiable}."
            )
        except Exception as exc:
            last_error = exc
            if blueprint.identifiability_regime == IDENTIFIABILITY_IDENTIFIABLE:
                total_fill_attempts += 1

    raise RuntimeError(
        "Failed to realize blueprint row "
        f"{blueprint.scene_id!r} after {restart_count + 1} seed block(s). "
        f"Last error: {last_error}"
    )
