"""
Given an identifiable SCM, generate plausible datasets.
- Pure Python/Numpy/Pandas (no external SCM library), so it runs anywhere.
- Generates observational and interventional samples from a randomized SCM consistent with the DAG.
- Supports binary and continuous variables; easy to extend to categorical later.
- Handles latent nodes by dropping them from the returned DataFrame (based on `observed_nodes`).
- Provides a Monte Carlo "true effect" calculator for ATE/ATE-like contrasts via do-interventions.

You can import this as a module in your own repo (e.g., save as `datagen.py`), or tweak inline.

NOTE: This code expects the `SampledGraph` dataclass defined previously (fields: graph, treatment, outcome, motif, observed_nodes, meta).
"""

import copy
import json
import logging
import numbers
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .graph import get_node_roles, toposort
from .utils import (
    deep_merge_dicts,
    ensure_list,
    json_safe,
    normalize_random_seed,
    offset_random_seed,
    random_sign,
    sigmoid,
    softplus,
    softsign,
    tanh,
)

logger = logging.getLogger(__name__)

# Minimal re-declaration (comment out this block if you already have SampledGraph from the sampler)
from .graph import SampledGraph


# ------------------------------
# Mechanism configuration
# ------------------------------
VALID_CONTINUOUS_MECHANISM_FAMILIES = {
    "handcrafted_continuous",
    "linear_additive",
    "interaction_response",
    "symbolic_transform",
    "generic_nn",
}
CONTINUOUS_MECHANISM_ALIASES = {
    "default": "handcrafted_continuous",
    "handcrafted": "handcrafted_continuous",
    "legacy": "handcrafted_continuous",
    "linear": "linear_additive",
    "additive": "linear_additive",
    "interaction": "interaction_response",
    "symbolic": "symbolic_transform",
    "nn": "generic_nn",
    "random_nn": "generic_nn",
    "nonlinear_nn": "generic_nn",
}
VALID_INTERACTION_RESPONSE_SUBFAMILIES = {
    "product",
    "saturating",
    "cooperative",
}
VALID_SYMBOLIC_TRANSFORM_SUBFAMILIES = {
    "rational",
    "power",
    "exp_log",
    "piecewise",
}


def _normalize_named_weights(
    raw: Optional[Dict[str, float]],
    *,
    valid_keys: set,
    default: Dict[str, float],
    field_name: str,
) -> Dict[str, float]:
    """Validate and normalize a nonnegative named weight dictionary."""
    weights = dict(raw or default)
    total = 0.0
    normalized: Dict[str, float] = {}
    for raw_key, raw_weight in weights.items():
        key = str(raw_key).strip().lower()
        if key not in valid_keys:
            raise ValueError(
                f"Unknown {field_name} key {raw_key!r}. "
                f"Expected one of {sorted(valid_keys)}."
            )
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"{field_name} weight for {key!r} must be >= 0.")
        if weight == 0.0:
            continue
        normalized[key] = weight
        total += weight
    if total <= 0.0:
        raise ValueError(f"{field_name} weights must contain positive mass.")
    return {key: weight / total for key, weight in normalized.items()}


def _sample_weighted_key(rng: np.random.RandomState, weights: Dict[str, float]) -> str:
    """Sample one key from an already-normalized weight dictionary."""
    keys = list(weights)
    probs = np.asarray([weights[key] for key in keys], dtype=float)
    idx = int(rng.choice(len(keys), p=probs))
    return keys[idx]


def _canonical_continuous_mechanism_family(raw: Optional[str]) -> str:
    """Normalize aliases for continuous structural mechanism families."""
    key = str(raw or "handcrafted_continuous").strip().lower()
    key = CONTINUOUS_MECHANISM_ALIASES.get(key, key)
    if key not in VALID_CONTINUOUS_MECHANISM_FAMILIES:
        raise ValueError(
            f"Unknown continuous mechanism family {raw!r}. "
            f"Expected one of {sorted(VALID_CONTINUOUS_MECHANISM_FAMILIES)}."
        )
    return key


@dataclass
class MechanismConfig:
    """Controls mechanism generation. Default probabilities are all 0 (handcrafted only)."""

    # Structural mechanism family for endogenous continuous nodes.
    continuous_mechanism_family: str = "handcrafted_continuous"

    # Gaussian mixture noise
    # P(use a Gaussian-mixture noise model for endogenous continuous nodes).
    prob_mixture_noise: float = 0.0
    # Inclusive (min,max) number of mixture components.
    mixture_components_range: Tuple[int, int] = (2, 6)
    # Relative scale for each mixture component std.
    mixture_component_std: float = 0.85
    # P(use a Gaussian-mixture exogenous distribution for root continuous nodes).
    prob_mixture_root: float = 0.0

    # Discretization
    # P(discretize a continuous node after simulation).
    prob_discretized: float = 0.0
    # Inclusive (min,max) discretization bin count.
    discrete_bins_range: Tuple[int, int] = (2, 8)
    # Whether treatment/outcome are allowed to be discretized when enabled.
    discretize_treatment: bool = False
    discretize_outcome: bool = False

    # Random NN mechanisms
    # P(sample a random NN mechanism instead of handcrafted mechanism).
    prob_nn_mechanism: float = 0.0
    # Inclusive (min,max) hidden-layer count and hidden-unit count.
    nn_hidden_layers_range: Tuple[int, int] = (2, 4)
    nn_hidden_units_range: Tuple[int, int] = (4, 64)
    # Spectral-normalized weight multiplier for hidden layers.
    nn_weight_scale: float = 5.0
    # Output calibration range for random NN mechanisms.
    nn_output_value_range: Tuple[float, float] = (-1.0, 1.0)
    # Number of samples for NN/discretization internal calibration pass.
    num_calibration_samples: int = 1000

    # Heteroscedastic noise
    # P(use parent-dependent noise scale for continuous endogenous nodes).
    prob_heteroscedastic: float = 0.0
    # Clamp range for heteroscedastic noise scale.
    heteroscedastic_scale_range: Tuple[float, float] = (0.3, 2.0)
    heteroscedastic_mode: str = "linear"  # "linear" or "nn"

    # Nonlinearity shift range for handcrafted mechanisms.
    # tanh(z - c) / softsign(z - c) with c ~ Uniform(lo, hi).
    # (0, 0) = no shift (default for SCM).
    # Nonzero range (e.g. (-1.5, 1.5)) breaks the systematic derivative
    # cancellation at z=0 for proxy mechanisms.
    nl_shift_range: Tuple[float, float] = (0.0, 0.0)

    # Linear/additive mechanism parameters.
    linear_weight_loc_abs: float = 0.8
    linear_weight_scale: float = 0.4
    linear_bias_scale: float = 0.5

    # Structured interaction-response mechanisms.
    interaction_response_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "product": 1.0,
            "saturating": 1.0,
            "cooperative": 1.0,
        }
    )
    interaction_product_parent_count_range: Tuple[int, int] = (2, 3)
    interaction_response_scale_range: Tuple[float, float] = (0.5, 1.5)
    interaction_response_k_range: Tuple[float, float] = (0.5, 2.0)
    interaction_response_hill_range: Tuple[float, float] = (1.5, 4.0)

    # Symbolic nonlinear-transform mechanisms.
    symbolic_transform_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "rational": 1.0,
            "power": 1.0,
            "exp_log": 1.0,
            "piecewise": 1.0,
        }
    )
    symbolic_transform_scale_range: Tuple[float, float] = (0.4, 1.2)
    symbolic_power_range: Tuple[float, float] = (0.5, 2.0)
    symbolic_clip_range: Tuple[float, float] = (-2.0, 2.0)
    symbolic_threshold_range: Tuple[float, float] = (-1.0, 1.0)
    symbolic_piecewise_step_prob: float = 0.5

    def __post_init__(self) -> None:
        self.continuous_mechanism_family = _canonical_continuous_mechanism_family(
            self.continuous_mechanism_family
        )
        self.mixture_components_range = tuple(
            int(v) for v in self.mixture_components_range
        )
        self.discrete_bins_range = tuple(int(v) for v in self.discrete_bins_range)
        self.nn_hidden_layers_range = tuple(int(v) for v in self.nn_hidden_layers_range)
        self.nn_hidden_units_range = tuple(int(v) for v in self.nn_hidden_units_range)
        self.nn_output_value_range = tuple(float(v) for v in self.nn_output_value_range)
        self.heteroscedastic_scale_range = tuple(
            float(v) for v in self.heteroscedastic_scale_range
        )
        self.nl_shift_range = tuple(float(v) for v in self.nl_shift_range)
        self.interaction_response_weights = _normalize_named_weights(
            self.interaction_response_weights,
            valid_keys=VALID_INTERACTION_RESPONSE_SUBFAMILIES,
            default={"product": 1.0, "saturating": 1.0, "cooperative": 1.0},
            field_name="interaction_response_weights",
        )
        self.interaction_product_parent_count_range = tuple(
            int(v) for v in self.interaction_product_parent_count_range
        )
        self.interaction_response_scale_range = tuple(
            float(v) for v in self.interaction_response_scale_range
        )
        self.interaction_response_k_range = tuple(
            float(v) for v in self.interaction_response_k_range
        )
        self.interaction_response_hill_range = tuple(
            float(v) for v in self.interaction_response_hill_range
        )
        self.symbolic_transform_weights = _normalize_named_weights(
            self.symbolic_transform_weights,
            valid_keys=VALID_SYMBOLIC_TRANSFORM_SUBFAMILIES,
            default={
                "rational": 1.0,
                "power": 1.0,
                "exp_log": 1.0,
                "piecewise": 1.0,
            },
            field_name="symbolic_transform_weights",
        )
        self.symbolic_transform_scale_range = tuple(
            float(v) for v in self.symbolic_transform_scale_range
        )
        self.symbolic_power_range = tuple(float(v) for v in self.symbolic_power_range)
        self.symbolic_clip_range = tuple(float(v) for v in self.symbolic_clip_range)
        self.symbolic_threshold_range = tuple(
            float(v) for v in self.symbolic_threshold_range
        )
        self.symbolic_piecewise_step_prob = float(self.symbolic_piecewise_step_prob)


@dataclass
class BinaryMechanismConfig:
    """Controls binary mechanism-family generation."""

    # Relative weights over binary endogenous mechanism families.
    mechanism_weights: Dict[str, float] = field(
        default_factory=lambda: {"logistic_softsign": 1.0}
    )
    # Root-node Bernoulli support range.
    root_prob_range: Tuple[float, float] = (0.25, 0.75)
    # Per-parent nonlinearity shift for logistic_softsign.
    nl_shift_range: Tuple[float, float] = (0.0, 0.0)
    # Linear-threshold sharpness range.
    threshold_sharpness_range: Tuple[float, float] = (4.0, 8.0)
    # Noisy Boolean gate flip-probability range.
    gate_flip_prob_range: Tuple[float, float] = (0.02, 0.12)
    # Probability that an input is negated in the generic logic_gate family.
    logic_gate_negation_prob: float = 0.25

    def __post_init__(self) -> None:
        self.mechanism_weights = _normalize_binary_mechanism_weights(
            self.mechanism_weights
        )
        self.root_prob_range = tuple(float(v) for v in self.root_prob_range)
        self.nl_shift_range = tuple(float(v) for v in self.nl_shift_range)
        self.threshold_sharpness_range = tuple(
            float(v) for v in self.threshold_sharpness_range
        )
        self.gate_flip_prob_range = tuple(float(v) for v in self.gate_flip_prob_range)
        self.logic_gate_negation_prob = float(self.logic_gate_negation_prob)


VALID_BINARY_MECHANISMS = {
    "logistic_softsign",
    "threshold",
    "noisy_or",
    "noisy_and",
    "logic_gate",
}


def _normalize_binary_mechanism_weights(raw: Dict[str, float]) -> Dict[str, float]:
    """Validate and normalize binary mechanism weights."""
    weights = dict(raw or {"logistic_softsign": 1.0})
    total = 0.0
    normalized: Dict[str, float] = {}
    for raw_key, raw_weight in weights.items():
        key = str(raw_key).strip().lower()
        if key not in VALID_BINARY_MECHANISMS:
            raise ValueError(
                f"Unknown binary mechanism {raw_key!r}. "
                f"Expected one of {sorted(VALID_BINARY_MECHANISMS)}."
            )
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"binary mechanism weight for {key!r} must be >= 0.")
        if weight == 0.0:
            continue
        normalized[key] = weight
        total += weight
    if total <= 0.0:
        raise ValueError("binary mechanism weights must contain positive mass.")
    return {key: weight / total for key, weight in normalized.items()}


def _sample_binary_mechanism_key(
    rng: np.random.RandomState,
    config: Optional[BinaryMechanismConfig],
) -> str:
    """Sample one binary mechanism family from normalized profile weights."""
    cfg = config or BinaryMechanismConfig()
    keys = list(cfg.mechanism_weights)
    probs = np.asarray([cfg.mechanism_weights[key] for key in keys], dtype=float)
    idx = int(rng.choice(len(keys), p=probs))
    return keys[idx]


@dataclass
class CalibrationConfig:
    """Controls the public calibration subset for proxy observation models."""

    # Fraction of train rows targeted for calibration sampling.
    fraction: float = 0.10
    # Minimum number of calibration rows per scene.
    min_rows: int = 64
    # Number of latent-quantile bins used for coverage constraints.
    quantile_bins: int = 6
    # Fraction of calibration rows reserved for private baseline diagnostics.
    # This is separate from the benchmark test split.
    holdout_fraction: float = 0.3
    # Lower bound on held-out rows for diagnostics.
    holdout_min_rows: int = 24


@dataclass
class ProxyInformationThresholds:
    """Lower bounds for keeping a sampled proxy bundle."""

    # Minimum local information at any support point.
    min_information: float = 0.03
    # Tail quantiles used for weak-support checks.
    lower_tail_quantiles: Tuple[float, float] = (0.1, 0.25)
    # Required minimum information at each configured tail quantile.
    lower_tail_minima: Tuple[float, float] = (0.04, 0.08)
    # Required average information across support.
    average_information_min: float = 0.12


@dataclass
class ObservationConfig:
    """Configuration for post-hoc proxy observation models."""

    # Enable/disable proxy observation modeling.
    # Default is True at the object level; generation should still set this
    # explicitly in config YAML to make variant behavior unambiguous.
    enabled: bool = True
    # Number of observed conceptual nodes to proxify.
    proxified_nodes_range: Tuple[int, int] = (1, 3)
    # Proxy bundle dimensionality.
    # Sampled uniformly over the inclusive integer range in _sample_proxy_bundle.
    proxy_dim_range: Tuple[int, int] = (2, 10)
    # Max attempts to resample an informative proxy bundle.
    max_bundle_resamples: int = 40
    # Local neighborhood fraction used for information diagnostics.
    local_info_window_fraction: float = 0.15
    # Number of support grid points for diagnostics.
    info_grid_size: int = 64
    # Noise scale range for continuous proxy mechanisms.
    # Higher values = noisier proxies = harder recovery.
    # Higher values = noisier proxies = harder recovery.
    proxy_noise_scale_range: Tuple[float, float] = (0.35, 0.9)
    # If true, continuous proxies may include heteroscedastic noise.
    # When proxy_mechanism_config is set, its prob_heteroscedastic is used instead.
    allow_heteroscedastic: bool = True
    # Mechanism configuration for proxy mechanisms.  When set, proxies use the
    # same continuous structural families, mixture noise, and heteroscedastic
    # controls as SCM nodes.  When None, proxies use only the
    # handcrafted mechanism with uniform noise kind selection and hardcoded
    # heteroscedastic probability.
    proxy_mechanism_config: Optional[MechanismConfig] = None
    # When True, proxified outcome nodes preserve their latent type
    # (continuous outcomes get only continuous proxies, binary get only binary).
    outcome_preserve_type: bool = True
    # Stricter outcome thresholds via multiplicative factor.
    outcome_info_multiplier: float = 1.75
    # Gradient boosting hyperparameters for calibrated multi-proxy baseline.
    baseline_n_estimators: int = 100
    baseline_max_depth: int = 3
    baseline_learning_rate: float = 0.1
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    info_thresholds: ProxyInformationThresholds = field(
        default_factory=ProxyInformationThresholds
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert this config into a JSON-serializable payload."""
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ObservationConfig":
        """Build a config from canonical OmegaConf/plain dict data."""
        raw = dict(raw or {})

        calibration_raw = raw.pop("calibration", None)
        thresholds_raw = raw.pop("info_thresholds", None)
        proxy_mech_raw = raw.pop("proxy_mechanism_config", None)

        if calibration_raw is not None:
            raw["calibration"] = CalibrationConfig(**dict(calibration_raw))
        if thresholds_raw is not None:
            raw["info_thresholds"] = ProxyInformationThresholds(**dict(thresholds_raw))
        if proxy_mech_raw is not None:
            raw["proxy_mechanism_config"] = MechanismConfig(**dict(proxy_mech_raw))

        for key in ("proxified_nodes_range", "proxy_dim_range"):
            if key in raw and raw[key] is not None:
                raw[key] = tuple(int(v) for v in raw[key])
        if (
            "proxy_noise_scale_range" in raw
            and raw["proxy_noise_scale_range"] is not None
        ):
            raw["proxy_noise_scale_range"] = tuple(
                float(v) for v in raw["proxy_noise_scale_range"]
            )
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate observation-model numeric ranges and constraints."""
        range_specs = (
            ("proxified_nodes_range", self.proxified_nodes_range, 0),
            ("proxy_dim_range", self.proxy_dim_range, 1),
        )
        for name, value, min_allowed in range_specs:
            lo, hi = value
            lo_i, hi_i = int(lo), int(hi)
            if lo_i < min_allowed or hi_i < min_allowed:
                raise ValueError(f"{name} values must be >= {min_allowed}, got {value}")
            if lo_i > hi_i:
                raise ValueError(f"{name} must satisfy low <= high, got {value}")


def merge_observation_config(
    base_config: Optional[ObservationConfig],
    override: Optional[Dict[str, Any]] = None,
) -> ObservationConfig:
    """Merge an observation-config override onto a base config."""
    base = base_config or ObservationConfig()
    if not override:
        return ObservationConfig.from_dict(base.to_dict())
    merged = deep_merge_dicts(base.to_dict(), dict(override))
    return ObservationConfig.from_dict(merged)


def resolve_observation_variant_configs(
    base_config: Optional[ObservationConfig],
    raw_variants: Optional[Dict[str, Any]],
) -> Dict[str, ObservationConfig]:
    """Resolve named observation-variant configs from a plain-dict payload."""
    if not raw_variants:
        return {}

    payload = dict(raw_variants)
    if "enabled" in payload and not bool(payload.get("enabled")):
        return {}

    variants_raw = payload.get("variants", payload)
    if variants_raw is None:
        return {}
    if not isinstance(variants_raw, dict):
        raise ValueError(
            "observation_variants must be a dict or contain a `variants` dict"
        )

    resolved: Dict[str, ObservationConfig] = {}
    for variant_name, override in variants_raw.items():
        resolved[str(variant_name)] = merge_observation_config(
            base_config,
            override=dict(override or {}),
        )
    return resolved


@dataclass
class DataGeneratorSpec:
    """Serializable recipe for constructing a :class:`DataGenerator`."""

    node_types: Optional[Dict[str, str]] = None
    force_treatment_binary: Optional[bool] = None
    force_outcome_continuous: Optional[bool] = None
    seed: Optional[int] = None
    scm_profile: Optional[str] = None
    continuous_scm_profile: Optional[str] = None
    binary_scm_profile: Optional[str] = None
    mech_config: Optional[MechanismConfig] = None
    binary_mech_config: Optional[BinaryMechanismConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert this spec into a JSON-serializable payload."""
        return {
            "node_types": json_safe(self.node_types),
            "force_treatment_binary": self.force_treatment_binary,
            "force_outcome_continuous": self.force_outcome_continuous,
            "seed": self.seed,
            "scm_profile": self.scm_profile,
            "continuous_scm_profile": self.continuous_scm_profile,
            "binary_scm_profile": self.binary_scm_profile,
            "mech_config": (
                json_safe(vars(self.mech_config))
                if self.mech_config is not None
                else None
            ),
            "binary_mech_config": (
                json_safe(vars(self.binary_mech_config))
                if self.binary_mech_config is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "DataGeneratorSpec":
        """Build a data-generator spec from manifest/checkpoint data."""
        raw = dict(raw or {})
        mech_raw = raw.get("mech_config")
        mech_config = None
        if mech_raw is not None:
            mech_config = (
                mech_raw
                if isinstance(mech_raw, MechanismConfig)
                else MechanismConfig(**dict(mech_raw))
            )
        binary_mech_raw = raw.get("binary_mech_config")
        binary_mech_config = None
        if binary_mech_raw is not None:
            binary_mech_config = (
                binary_mech_raw
                if isinstance(binary_mech_raw, BinaryMechanismConfig)
                else BinaryMechanismConfig(**dict(binary_mech_raw))
            )

        node_types = raw.get("node_types")
        if node_types is not None:
            node_types = {
                str(node): str(dtype) for node, dtype in dict(node_types).items()
            }
        legacy_profile = raw.get("scm_profile")
        continuous_profile = raw.get("continuous_scm_profile", legacy_profile)

        return cls(
            node_types=node_types,
            force_treatment_binary=(
                None
                if raw.get("force_treatment_binary") is None
                else bool(raw.get("force_treatment_binary"))
            ),
            force_outcome_continuous=(
                None
                if raw.get("force_outcome_continuous") is None
                else bool(raw.get("force_outcome_continuous"))
            ),
            seed=None if raw.get("seed") is None else int(raw.get("seed")),
            scm_profile=(
                None
                if raw.get("scm_profile") in (None, "")
                else str(raw.get("scm_profile"))
            ),
            continuous_scm_profile=(
                None if continuous_profile in (None, "") else str(continuous_profile)
            ),
            binary_scm_profile=(
                None
                if raw.get("binary_scm_profile") in (None, "")
                else str(raw.get("binary_scm_profile"))
            ),
            mech_config=mech_config,
            binary_mech_config=binary_mech_config,
        )

    def create_generator(
        self,
        sg: SampledGraph,
        *,
        default_node_types: Optional[Dict[str, str]] = None,
        default_force_treatment_binary: bool = False,
        default_force_outcome_continuous: bool = False,
        default_seed: Optional[int] = None,
        default_mech_config: Optional[MechanismConfig] = None,
        default_binary_mech_config: Optional[BinaryMechanismConfig] = None,
    ) -> "DataGenerator":
        """Instantiate a data generator using this spec plus fallback defaults."""
        resolved_node_types = dict(default_node_types or {})
        if self.node_types:
            resolved_node_types.update(dict(self.node_types))

        mech_config = (
            copy.deepcopy(self.mech_config)
            if self.mech_config is not None
            else copy.deepcopy(default_mech_config)
        )
        binary_mech_config = (
            copy.deepcopy(self.binary_mech_config)
            if self.binary_mech_config is not None
            else copy.deepcopy(default_binary_mech_config)
        )
        continuous_scm_profile = self.continuous_scm_profile or self.scm_profile

        return DataGenerator(
            sg=sg,
            node_types=resolved_node_types or None,
            force_treatment_binary=(
                default_force_treatment_binary
                if self.force_treatment_binary is None
                else bool(self.force_treatment_binary)
            ),
            force_outcome_continuous=(
                default_force_outcome_continuous
                if self.force_outcome_continuous is None
                else bool(self.force_outcome_continuous)
            ),
            seed=default_seed if self.seed is None else int(self.seed),
            mech_config=mech_config,
            binary_mech_config=binary_mech_config,
            continuous_scm_profile=continuous_scm_profile,
            binary_scm_profile=self.binary_scm_profile,
        )


@dataclass
class ProxyColumnSpec:
    """Runtime + metadata representation of one proxy column."""

    column_name: str
    latent_name: str
    latent_type: str
    proxy_type: str
    mechanism_kind: str
    noise_kind: str
    base_noise_scale: float
    params: Dict[str, Any]
    mechanism: "Mechanism"

    def to_metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata."""
        params = _serialize_proxy_params(self.params)
        return {
            "column_name": self.column_name,
            "conceptual_node": self.latent_name,
            "conceptual_node_type": self.latent_type,
            "proxy_type": self.proxy_type,
            "mechanism_kind": self.mechanism_kind,
            "noise_kind": self.noise_kind,
            "base_noise_scale": float(self.base_noise_scale),
            "params": params,
        }


@dataclass
class ProxyBundle:
    """A bundle of proxy columns measuring one conceptual observed variable."""

    latent_name: str
    latent_type: str
    proxy_columns: List[ProxyColumnSpec]
    proxy_frame: pd.DataFrame
    information_diagnostics: Dict[str, Any]
    baseline_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def proxy_names(self) -> List[str]:
        """Return proxy column names in order."""
        return [spec.column_name for spec in self.proxy_columns]

    def to_metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata."""
        return {
            "conceptual_node": self.latent_name,
            "conceptual_node_type": self.latent_type,
            "proxy_columns": [spec.to_metadata() for spec in self.proxy_columns],
            "information_diagnostics": self.information_diagnostics,
            "baseline_diagnostics": self.baseline_diagnostics,
        }


@dataclass
class ObservationData:
    """Observed/public view plus calibration data derived from conceptual data."""

    latent_data: pd.DataFrame
    public_data: pd.DataFrame
    calibration_data: Optional[pd.DataFrame]
    proxified_nodes: List[str]
    proxy_bundles: Dict[str, ProxyBundle]
    calibration_indices: List[int]
    metadata: Dict[str, Any]


# ------------------------------
# Mechanism factory
# ------------------------------
# NOTE: inspired by doWhy
class Mechanism:
    """
    Callable mechanism with parameters; given parent values returns a value (before noise for continuous).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        params: Dict[str, Any],
        noise: Optional[Callable[[int], np.ndarray]] = None,
        var_type: str = "continuous",
        noise_scale_fn: Optional[Callable] = None,
    ):
        self.fn = fn
        self.params = params
        self.noise = noise
        self.var_type = var_type  # 'continuous' or 'binary'
        self.noise_scale_fn = (
            noise_scale_fn  # heteroscedastic: (parents, size, rng) -> scale
        )

    def __call__(
        self,
        parents: Dict[str, np.ndarray],
        size: int,
        rng: np.random.RandomState,
    ):
        val = self.fn(parents, self.params, size, rng)
        if self.var_type == "continuous":
            if self.noise is not None:
                n = self.noise(size)
                if self.noise_scale_fn is not None:
                    n = n * self.noise_scale_fn(parents, size, rng)
                val = val + n
            return val
        elif self.var_type == "binary":
            # Interpret val as log-odds; map to probability and sample Bernoulli
            p = sigmoid(val)
            return rng.binomial(1, np.clip(p, 1e-6, 1 - 1e-6)).astype(float)
        else:
            raise ValueError(f"Unknown var_type {self.var_type}")


# ------------------------------
# Mechanism & noise registry
# ------------------------------
# Central catalogue of every mechanism function, noise distribution, and noise
# modifier used by SCM and proxy pipelines.  The notebook diagnostic (section
# 19) imports these registries to iterate over mechanism families without
# hard-coding names or coefficient priors.


@dataclass(frozen=True)
class MechanismFamilyInfo:
    """Descriptor for one mechanism function family."""

    key: str  # short identifier (e.g. "handcrafted_continuous")
    label: str  # human-readable label
    builder: str  # name of the builder function (e.g. "_mk_cont_fn")
    nonlinearity: str  # e.g. "tanh", "softsign", "tanh_multi_layer", "none"
    used_in_scm: bool
    used_in_proxy: bool
    # coefficient priors: list of (name, loc_abs, scale, random_sign) tuples
    coefficient_priors: Tuple[Tuple[str, float, float, bool], ...]
    notes: str = ""


@dataclass(frozen=True)
class NoiseDistInfo:
    """Descriptor for one noise distribution family."""

    key: str  # e.g. "gauss", "laplace", "studentt", "mixture"
    label: str
    used_in_scm: bool
    used_in_proxy: bool
    notes: str = ""


@dataclass(frozen=True)
class NoiseModifierInfo:
    """Descriptor for a noise modifier (e.g. heteroscedastic scale)."""

    key: str
    label: str
    builder: str
    used_in_scm: bool
    used_in_proxy: bool
    notes: str = ""


MECHANISM_FAMILIES: Dict[str, MechanismFamilyInfo] = {
    "handcrafted_continuous": MechanismFamilyInfo(
        key="handcrafted_continuous",
        label="Handcrafted continuous (linear + tanh)",
        builder="_mk_cont_fn",
        nonlinearity="tanh",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(
            ("w", 0.8, 0.4, True),
            ("b", 0.0, 0.5, False),
            ("w_nl", 0.5, 0.3, True),
            ("w_pair", 0.3, 0.2, True),
        ),
        notes=(
            "Single-parent (proxy): h(z) = w*z + b + w_nl*tanh(z). "
            "Multi-parent (SCM): adds pairwise interaction terms."
        ),
    ),
    "linear_additive": MechanismFamilyInfo(
        key="linear_additive",
        label="Linear additive continuous",
        builder="_mk_linear_additive_fn",
        nonlinearity="none",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(
            ("w", 0.8, 0.4, True),
            ("b", 0.0, 0.5, False),
        ),
        notes="Additive affine mechanism: h(x) = b + sum_i w_i x_i.",
    ),
    "interaction_response": MechanismFamilyInfo(
        key="interaction_response",
        label="Interaction response continuous",
        builder="_mk_interaction_response_fn",
        nonlinearity="product_saturation_hill",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(
            ("a", 1.0, 0.4, True),
            ("b", 0.0, 0.5, False),
        ),
        notes=(
            "Small internal mixture over product-like, saturating, and "
            "cooperative response shapes."
        ),
    ),
    "symbolic_transform": MechanismFamilyInfo(
        key="symbolic_transform",
        label="Symbolic nonlinear transform continuous",
        builder="_mk_symbolic_transform_fn",
        nonlinearity="rational_power_exp_log_piecewise",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(
            ("a", 0.8, 0.3, True),
            ("b", 0.0, 0.5, False),
        ),
        notes=(
            "Small internal mixture over safe rational, power, exp/log, "
            "and piecewise/threshold transforms."
        ),
    ),
    "handcrafted_binary": MechanismFamilyInfo(
        key="handcrafted_binary",
        label="Handcrafted binary (logistic + softsign)",
        builder="_mk_bin_fn",
        nonlinearity="softsign",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(
            ("w", 1.0, 0.6, True),
            ("b", 0.0, 0.7, False),
            ("w_nl", 0.6, 0.3, True),
        ),
        notes=(
            "Outputs log-odds through sigmoid + Bernoulli. "
            "eta(z) = w*z + b + w_nl*softsign(z)."
        ),
    ),
    "binary_threshold": MechanismFamilyInfo(
        key="binary_threshold",
        label="Binary threshold",
        builder="_mk_binary_threshold_fn",
        nonlinearity="hard_threshold_logit",
        used_in_scm=True,
        used_in_proxy=False,
        coefficient_priors=(
            ("w", 1.0, 0.5, True),
            ("threshold", 0.0, 0.5, False),
        ),
        notes="Sharp logistic threshold over weighted parent inputs.",
    ),
    "binary_noisy_gate": MechanismFamilyInfo(
        key="binary_noisy_gate",
        label="Binary noisy Boolean gate",
        builder="_mk_binary_gate_fn",
        nonlinearity="boolean_gate",
        used_in_scm=True,
        used_in_proxy=False,
        coefficient_priors=(),
        notes="Noisy OR/AND-style gate over binarized parent inputs.",
    ),
    "random_nn": MechanismFamilyInfo(
        key="random_nn",
        label="Random neural network (tanh activations)",
        builder="_mk_nn_fn",
        nonlinearity="tanh_multi_layer",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(),  # weights are spectral-normalized, not from simple priors
        notes=(
            "2-4 hidden layers, 4-64 units, spectral-normalized weights "
            "scaled by weight_scale (default 5.0). Output calibrated to [-1,1]. "
            "Proxy use requires proxy_mechanism_config with prob_nn_mechanism > 0."
        ),
    ),
    "generic_nn": MechanismFamilyInfo(
        key="generic_nn",
        label="Generic neural continuous",
        builder="_mk_nn_fn",
        nonlinearity="tanh_multi_layer",
        used_in_scm=True,
        used_in_proxy=True,
        coefficient_priors=(),
        notes=(
            "Alias profile family for the calibrated random neural mechanism. "
            "Use continuous_mechanism_family: generic_nn to select it directly."
        ),
    ),
    "root_continuous": MechanismFamilyInfo(
        key="root_continuous",
        label="Root continuous (noise only)",
        builder="(inline lambda)",
        nonlinearity="none",
        used_in_scm=True,
        used_in_proxy=False,
        coefficient_priors=(),
        notes="Z = epsilon, scale ~ Uniform(0.8, 1.5).",
    ),
    "root_binary": MechanismFamilyInfo(
        key="root_binary",
        label="Root binary (constant log-odds)",
        builder="(inline lambda)",
        nonlinearity="none",
        used_in_scm=True,
        used_in_proxy=False,
        coefficient_priors=(),
        notes="p ~ Uniform(0.25, 0.75), constant log-odds through sigmoid + Bernoulli.",
    ),
}

NOISE_DISTRIBUTIONS: Dict[str, NoiseDistInfo] = {
    "gauss": NoiseDistInfo(
        key="gauss",
        label="Gaussian",
        used_in_scm=True,
        used_in_proxy=True,
        notes="N(0, scale). Default noise.",
    ),
    "laplace": NoiseDistInfo(
        key="laplace",
        label="Laplace",
        used_in_scm=True,
        used_in_proxy=True,
        notes="Laplace(0, scale). Heavier tails than Gaussian.",
    ),
    "studentt": NoiseDistInfo(
        key="studentt",
        label="Student-t",
        used_in_scm=True,
        used_in_proxy=True,
        notes="t(df) * scale, df in [3, 10]. Heavy tails, possible outliers.",
    ),
    "mixture": NoiseDistInfo(
        key="mixture",
        label="Gaussian mixture",
        used_in_scm=True,
        used_in_proxy=True,
        notes=(
            "K components (K in [2,6]), centered means ~ U(-1,1), "
            "component std = 0.85 * scale. Requires prob_mixture_noise > 0 "
            "(in MechanismConfig for SCM or proxy_mechanism_config for proxies)."
        ),
    ),
}

NOISE_MODIFIERS: Dict[str, NoiseModifierInfo] = {
    "heteroscedastic_linear": NoiseModifierInfo(
        key="heteroscedastic_linear",
        label="Heteroscedastic (softplus linear)",
        builder="_mk_heteroscedastic_scale_linear",
        used_in_scm=True,
        used_in_proxy=True,
        notes=(
            "sigma(z) = softplus(w_sigma*z + b_sigma), clamped. "
            "w_sigma ~ N(0, 0.5), b_sigma ~ N(0, 0.3). "
            "SCM: prob_heteroscedastic. Proxy: 35% when allow_heteroscedastic."
        ),
    ),
}


def _make_noise_dist(
    rng: np.random.RandomState,
    kind: str = "gauss",
    scale: float = 1.0,
    config: Optional[MechanismConfig] = None,
):
    logger.debug("Creating noise distribution: kind=%s, scale=%.3f", kind, scale)
    if kind == "gauss":
        return lambda n: rng.normal(0.0, scale, size=n)
    if kind == "laplace":
        return lambda n: rng.laplace(0.0, scale, size=n)
    if kind == "studentt":
        # df between 3 and 10
        df = rng.randint(3, 11)
        # NOTE: Capture df and scale by value via default args to avoid late-binding closure bug
        return lambda n, df=df, scale=scale: rng.standard_t(df, size=n) * scale
    if kind == "mixture":
        cfg = config or MechanismConfig()
        lo, hi = cfg.mixture_components_range
        k = rng.randint(lo, hi + 1)
        means = rng.uniform(-1, 1, size=k)
        means = means - means.mean()  # center around 0
        stds = np.full(k, cfg.mixture_component_std * scale)
        weights = np.ones(k) / k

        # Capture by value
        def _mixture(n, means=means, stds=stds, weights=weights):
            ids = rng.choice(len(means), size=n, p=weights)
            return np.array([rng.normal(means[i], stds[i]) for i in ids])

        return _mixture
    return lambda n: rng.normal(0.0, scale, size=n)


def _sample_signed_normal(
    rng: np.random.RandomState,
    *,
    loc_abs: float,
    scale: float,
) -> float:
    """Sample a normal coefficient with random sign and positive-magnitude center."""
    return float(rng.normal(loc=loc_abs * random_sign(rng), scale=scale))


def _bounded_range(
    values: Tuple[float, float],
    *,
    positive: bool = False,
) -> Tuple[float, float]:
    """Normalize a range tuple and enforce positive bounds when requested."""
    lo, hi = (float(values[0]), float(values[1]))
    if hi < lo:
        lo, hi = hi, lo
    if positive:
        lo = max(lo, 1e-6)
        hi = max(hi, lo)
    return lo, hi


def _positive_parent_input(values: np.ndarray) -> np.ndarray:
    """Stable positive transform used by interaction-response mechanisms."""
    return softplus(np.clip(values, -20.0, 20.0))


def _pick_parent_subset(
    parents: List[str],
    rng: np.random.RandomState,
    count_range: Tuple[int, int],
) -> List[str]:
    """Select a bounded random subset of parents."""
    if not parents:
        return []
    lo, hi = (int(count_range[0]), int(count_range[1]))
    if hi < lo:
        lo, hi = hi, lo
    lo = max(1, min(lo, len(parents)))
    hi = max(lo, min(hi, len(parents)))
    count = int(rng.randint(lo, hi + 1))
    indices = list(range(len(parents)))
    rng.shuffle(indices)
    return [parents[i] for i in indices[:count]]


def _sample_pairs(
    parents: List[str],
    rng: np.random.RandomState,
    *,
    max_pairs: int = 2,
) -> List[Tuple[str, str]]:
    """Sample disjoint adjacent parent pairs after shuffling."""
    if len(parents) < 2:
        return []
    indices = list(range(len(parents)))
    rng.shuffle(indices)
    return [
        (parents[indices[i]], parents[indices[i + 1]])
        for i in range(0, min(len(parents) - 1, max_pairs * 2), 2)
    ]


def _mk_linear_additive_fn(
    parents: List[str],
    rng: np.random.RandomState,
    config: MechanismConfig,
):
    """Create f(x) = b + sum_i w_i x_i."""
    logger.debug("Creating linear_additive mechanism for %d parents", len(parents))
    w = {
        p: _sample_signed_normal(
            rng,
            loc_abs=config.linear_weight_loc_abs,
            scale=config.linear_weight_scale,
        )
        for p in parents
    }
    b = float(rng.normal(0.0, config.linear_bias_scale))

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        s = b
        for p, coef in w.items():
            s = s + coef * par_vals[p]
        return s

    return fn, {"w": w, "b": b, "continuous_mechanism_family": "linear_additive"}


def _mk_cont_fn(
    parents: List[str],
    rng: np.random.RandomState,
    nl_shift_range: Tuple[float, float] = (0.0, 0.0),
):
    # NOTE: Linear + interaction + mild nonlinearity mix
    logger.debug("Creating continuous mechanism for %d parents", len(parents))

    # linear terms
    w = {p: rng.normal(loc=0.8 * random_sign(rng), scale=0.4) for p in parents}
    b = rng.normal(0.0, 0.5)

    # pairwise interactions for up to 3 pairs
    pairs = []
    if len(parents) >= 2:
        idx = list(range(len(parents)))
        rng.shuffle(idx)
        pairs = [
            (parents[idx[i]], parents[idx[i + 1]])
            for i in range(0, min(len(parents) - 1, 3), 2)
        ]
    w_pair = {pair: rng.normal(loc=0.3 * random_sign(rng), scale=0.2) for pair in pairs}

    # nonlinear terms
    w_nl = {p: rng.normal(loc=0.5 * random_sign(rng), scale=0.3) for p in parents}

    # per-parent nonlinearity shift: tanh(z - c) instead of tanh(z)
    # When nl_shift_range=(0, 0) (default for SCM), no shift is applied.
    nl_shift = {
        p: float(rng.uniform(*nl_shift_range)) if nl_shift_range != (0.0, 0.0) else 0.0
        for p in parents
    }

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        s = b
        for p, coef in w.items():
            s = s + coef * par_vals[p]
        for (p1, p2), coef in w_pair.items():
            s = s + coef * par_vals[p1] * par_vals[p2]
        for p, coef in w_nl.items():
            s = s + coef * tanh(par_vals[p] - nl_shift[p])
        return s

    return fn, {"w": w, "b": b, "w_pair": w_pair, "w_nl": w_nl, "nl_shift": nl_shift}


def _mk_interaction_response_fn(
    parents: List[str],
    rng: np.random.RandomState,
    config: MechanismConfig,
):
    """Create a product, saturating, or cooperative response mechanism."""
    logger.debug("Creating interaction_response mechanism for %d parents", len(parents))
    subfamily = _sample_weighted_key(rng, config.interaction_response_weights)
    scale_lo, scale_hi = _bounded_range(config.interaction_response_scale_range)
    k_lo, k_hi = _bounded_range(config.interaction_response_k_range, positive=True)
    h_lo, h_hi = _bounded_range(config.interaction_response_hill_range, positive=True)
    b = float(rng.normal(0.0, config.linear_bias_scale))

    if subfamily == "product":
        product_parents = _pick_parent_subset(
            parents,
            rng,
            config.interaction_product_parent_count_range,
        )
        a = float(rng.uniform(scale_lo, scale_hi) * random_sign(rng))

        def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
            response = np.ones(size, dtype=float)
            for p in product_parents:
                response = response * _positive_parent_input(par_vals[p])
            return b + a * response

        params = {
            "continuous_mechanism_family": "interaction_response",
            "interaction_subfamily": subfamily,
            "product_parents": product_parents,
            "a": a,
            "b": b,
        }
        return fn, params

    a = {p: float(rng.uniform(scale_lo, scale_hi) * random_sign(rng)) for p in parents}
    k = {p: float(rng.uniform(k_lo, k_hi)) for p in parents}

    if subfamily == "saturating":

        def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
            response = np.full(size, b, dtype=float)
            for p in parents:
                u = _positive_parent_input(par_vals[p])
                response = response + a[p] * (u / (k[p] + u + 1e-8))
            return response

        params = {
            "continuous_mechanism_family": "interaction_response",
            "interaction_subfamily": subfamily,
            "a": a,
            "k": k,
            "b": b,
        }
        return fn, params

    hill = {p: float(rng.uniform(h_lo, h_hi)) for p in parents}

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        response = np.full(size, b, dtype=float)
        for p in parents:
            u = _positive_parent_input(par_vals[p])
            u_h = np.power(u, hill[p])
            k_h = k[p] ** hill[p]
            response = response + a[p] * (u_h / (k_h + u_h + 1e-8))
        return response

    params = {
        "continuous_mechanism_family": "interaction_response",
        "interaction_subfamily": subfamily,
        "a": a,
        "k": k,
        "hill": hill,
        "b": b,
    }
    return fn, params


def _mk_symbolic_transform_fn(
    parents: List[str],
    rng: np.random.RandomState,
    config: MechanismConfig,
):
    """Create a rational, power, exp/log, or piecewise transform mechanism."""
    logger.debug("Creating symbolic_transform mechanism for %d parents", len(parents))
    subfamily = _sample_weighted_key(rng, config.symbolic_transform_weights)
    scale_lo, scale_hi = _bounded_range(config.symbolic_transform_scale_range)
    power_lo, power_hi = _bounded_range(config.symbolic_power_range, positive=True)
    clip_lo, clip_hi = _bounded_range(config.symbolic_clip_range)
    threshold_lo, threshold_hi = _bounded_range(config.symbolic_threshold_range)
    b = float(rng.normal(0.0, config.linear_bias_scale))

    if subfamily == "rational":
        numerator_w = {
            p: _sample_signed_normal(
                rng,
                loc_abs=config.linear_weight_loc_abs,
                scale=config.linear_weight_scale,
            )
            for p in parents
        }
        denominator_w = {p: float(rng.normal(0.0, 0.4)) for p in parents}
        pairs = _sample_pairs(parents, rng, max_pairs=2)
        pair_w = {
            pair: float(rng.uniform(scale_lo, scale_hi) * random_sign(rng))
            for pair in pairs
        }
        numerator_b = float(rng.normal(0.0, 0.4))
        denominator_b = float(rng.normal(0.0, 0.4))

        def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
            numerator = np.full(size, numerator_b, dtype=float)
            denominator = np.full(size, denominator_b, dtype=float)
            for p, coef in numerator_w.items():
                numerator = numerator + coef * par_vals[p]
            for (p1, p2), coef in pair_w.items():
                numerator = numerator + coef * par_vals[p1] * par_vals[p2]
            for p, coef in denominator_w.items():
                denominator = denominator + coef * par_vals[p]
            return b + numerator / (1.0 + np.abs(denominator))

        params = {
            "continuous_mechanism_family": "symbolic_transform",
            "symbolic_subfamily": subfamily,
            "numerator_w": numerator_w,
            "denominator_w": denominator_w,
            "pair_w": pair_w,
            "numerator_b": numerator_b,
            "denominator_b": denominator_b,
            "b": b,
        }
        return fn, params

    a = {p: float(rng.uniform(scale_lo, scale_hi) * random_sign(rng)) for p in parents}

    if subfamily == "power":
        rho = {p: float(rng.uniform(power_lo, power_hi)) for p in parents}

        def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
            response = np.full(size, b, dtype=float)
            for p in parents:
                x = np.clip(par_vals[p], clip_lo, clip_hi)
                response = response + a[p] * np.sign(x) * np.power(np.abs(x), rho[p])
            return response

        params = {
            "continuous_mechanism_family": "symbolic_transform",
            "symbolic_subfamily": subfamily,
            "a": a,
            "rho": rho,
            "clip_range": (clip_lo, clip_hi),
            "b": b,
        }
        return fn, params

    if subfamily == "exp_log":
        transform_kind = str(rng.choice(["exp", "log"]))

        def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
            response = np.full(size, b, dtype=float)
            for p in parents:
                x = np.clip(par_vals[p], clip_lo, clip_hi)
                if transform_kind == "exp":
                    term = np.exp(x)
                else:
                    term = np.log1p(_positive_parent_input(x))
                response = response + a[p] * term
            return response

        params = {
            "continuous_mechanism_family": "symbolic_transform",
            "symbolic_subfamily": subfamily,
            "transform_kind": transform_kind,
            "a": a,
            "clip_range": (clip_lo, clip_hi),
            "b": b,
        }
        return fn, params

    threshold = {p: float(rng.uniform(threshold_lo, threshold_hi)) for p in parents}
    transform_kind = (
        "step" if rng.random() < config.symbolic_piecewise_step_prob else "hinge"
    )

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        response = np.full(size, b, dtype=float)
        for p in parents:
            x = par_vals[p] - threshold[p]
            if transform_kind == "step":
                term = (x > 0.0).astype(float)
            else:
                term = np.maximum(0.0, x)
            response = response + a[p] * term
        return response

    params = {
        "continuous_mechanism_family": "symbolic_transform",
        "symbolic_subfamily": subfamily,
        "transform_kind": transform_kind,
        "a": a,
        "threshold": threshold,
        "b": b,
    }
    return fn, params


def _make_continuous_mechanism(
    parents: List[str],
    rng: np.random.RandomState,
    config: Optional[MechanismConfig] = None,
) -> Tuple[Callable[..., Any], Dict[str, Any], str]:
    """Create one endogenous continuous mechanism from the selected family."""
    cfg = config or MechanismConfig()
    family = cfg.continuous_mechanism_family
    if family == "linear_additive":
        fn, params = _mk_linear_additive_fn(parents, rng, cfg)
    elif family == "interaction_response":
        fn, params = _mk_interaction_response_fn(parents, rng, cfg)
    elif family == "symbolic_transform":
        fn, params = _mk_symbolic_transform_fn(parents, rng, cfg)
    elif family == "generic_nn":
        fn, params = _mk_nn_fn(parents, rng, cfg)
        params["continuous_mechanism_family"] = "generic_nn"
    else:
        if config and rng.random() < cfg.prob_nn_mechanism:
            fn, params = _mk_nn_fn(parents, rng, cfg)
            params["continuous_mechanism_family"] = "generic_nn"
            family = "generic_nn"
        else:
            fn, params = _mk_cont_fn(
                parents,
                rng,
                nl_shift_range=cfg.nl_shift_range,
            )
            params["continuous_mechanism_family"] = "handcrafted_continuous"
            family = "handcrafted_continuous"
    return fn, params, f"endogenous_continuous_{family}"


# NOTE: one could just sample a predetermined table
def _mk_bin_fn(
    parents: List[str],
    rng: np.random.RandomState,
    nl_shift_range: Tuple[float, float] = (0.0, 0.0),
):
    # Logistic link over linear + mild nonlinearity
    logger.debug("Creating binary mechanism for %d parents", len(parents))

    w = {p: rng.normal(loc=1.0 * random_sign(rng), scale=0.6) for p in parents}
    b = rng.normal(0.0, 0.7)
    w_nl = {p: rng.normal(loc=0.6 * random_sign(rng), scale=0.3) for p in parents}

    # per-parent nonlinearity shift: softsign(z - c) instead of softsign(z)
    nl_shift = {
        p: float(rng.uniform(*nl_shift_range)) if nl_shift_range != (0.0, 0.0) else 0.0
        for p in parents
    }

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        z = b
        for p, coef in w.items():
            z = z + coef * par_vals[p]
        for p, coef in w_nl.items():
            z = z + coef * softsign(par_vals[p] - nl_shift[p])
        # z is log-odds for binary variables
        return z

    return fn, {"w": w, "b": b, "w_nl": w_nl, "nl_shift": nl_shift}


def _mk_binary_threshold_fn(
    parents: List[str],
    rng: np.random.RandomState,
    config: Optional[BinaryMechanismConfig] = None,
):
    """Create a sharp binary threshold mechanism over parent values."""
    cfg = config or BinaryMechanismConfig()
    w = {p: rng.normal(loc=1.0 * random_sign(rng), scale=0.5) for p in parents}
    threshold = rng.normal(0.0, 0.5)
    lo, hi = cfg.threshold_sharpness_range
    sharpness = float(rng.uniform(lo, hi))

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        score = -threshold
        for p, coef in w.items():
            score = score + coef * par_vals[p]
        return sharpness * score

    return fn, {"w": w, "threshold": threshold, "sharpness": sharpness}


def _binarize_parent_values(values: np.ndarray) -> np.ndarray:
    """Map binary-looking parents by 0.5 and continuous parents by 0."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if (
        finite.size
        and float(np.nanmin(finite)) >= 0.0
        and float(np.nanmax(finite)) <= 1.0
    ):
        return arr > 0.5
    return arr > 0.0


def _mk_binary_gate_fn(
    parents: List[str],
    rng: np.random.RandomState,
    config: Optional[BinaryMechanismConfig] = None,
    gate_kind: str = "or",
    allow_negations: bool = False,
):
    """Create a noisy Boolean-gate binary mechanism."""
    cfg = config or BinaryMechanismConfig()
    lo, hi = cfg.gate_flip_prob_range
    flip_prob = float(rng.uniform(lo, hi))
    low = np.clip(flip_prob, 1e-4, 1 - 1e-4)
    high = np.clip(1.0 - flip_prob, 1e-4, 1 - 1e-4)
    negated = {
        p: bool(allow_negations and rng.random() < cfg.logic_gate_negation_prob)
        for p in parents
    }

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        if not parents:
            prob = np.full(size, 0.5, dtype=float)
            return logit(prob)
        inputs = []
        for p in parents:
            active = _binarize_parent_values(par_vals[p])
            if negated[p]:
                active = np.logical_not(active)
            inputs.append(active)
        stacked = np.vstack(inputs)
        if gate_kind == "and":
            gate_active = np.all(stacked, axis=0)
        else:
            gate_active = np.any(stacked, axis=0)
        prob = np.where(gate_active, high, low)
        return logit(prob)

    return fn, {"gate_kind": gate_kind, "flip_prob": flip_prob, "negated": negated}


def _make_binary_mechanism(
    parents: List[str],
    rng: np.random.RandomState,
    config: Optional[BinaryMechanismConfig] = None,
) -> Tuple[Callable[..., Any], Dict[str, Any], str]:
    """Create one binary mechanism from the selected binary SCM profile."""
    cfg = config or BinaryMechanismConfig()
    family = _sample_binary_mechanism_key(rng, cfg)
    if family == "threshold":
        fn, params = _mk_binary_threshold_fn(parents, rng, cfg)
    elif family == "noisy_or":
        fn, params = _mk_binary_gate_fn(parents, rng, cfg, gate_kind="or")
    elif family == "noisy_and":
        fn, params = _mk_binary_gate_fn(parents, rng, cfg, gate_kind="and")
    elif family == "logic_gate":
        gate_kind = str(rng.choice(["or", "and"]))
        fn, params = _mk_binary_gate_fn(
            parents,
            rng,
            cfg,
            gate_kind=gate_kind,
            allow_negations=True,
        )
    else:
        fn, params = _mk_bin_fn(parents, rng, nl_shift_range=cfg.nl_shift_range)
    params["binary_mechanism_family"] = family
    return fn, params, f"endogenous_binary_{family}"


# ------------------------------
# Random NN mechanism
# ------------------------------
class _RandomNN:
    """Feed-forward NN with random spectral-normalized weights and tanh activations.

    Hidden-layer weights are spectral-normalized then scaled by ``weight_scale``
    to push tanh activations into their nonlinear regime. A calibration pass
    maps raw outputs to a bounded target range.
    """

    def __init__(
        self,
        n_inputs: int,
        rng: np.random.RandomState,
        config: MechanismConfig,
    ):
        lo_h, hi_h = config.nn_hidden_units_range
        lo_l, hi_l = config.nn_hidden_layers_range
        n_hidden = rng.randint(lo_l, hi_l + 1)

        dims = [n_inputs]
        for _ in range(n_hidden):
            dims.append(rng.randint(lo_h, hi_h + 1))
        dims.append(1)

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            w = rng.randn(d_in, d_out)
            if i < len(dims) - 2:  # spectral-normalize hidden layers only
                sigma = np.linalg.svd(w, compute_uv=False)[0]
                w = w / max(sigma, 1e-8) * config.nn_weight_scale
            self.biases.append(rng.uniform(-1, 1, size=(1, d_out)))
            self.weights.append(w)

        self._lo, self._hi = config.nn_output_value_range
        self._shift: float = 0.0
        self._scale: float = 1.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Raw forward pass (before calibration normalization)."""
        h = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
        return (h @ self.weights[-1] + self.biases[-1]).squeeze()

    def calibrate(self, X: np.ndarray) -> None:
        """Capture min/max of raw outputs for normalization."""
        raw = self.forward(X)
        lo, hi = float(np.min(raw)), float(np.max(raw))
        span = hi - lo if hi - lo > 1e-8 else 1.0
        self._shift = lo
        self._scale = span

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calibrated forward pass → outputs in target range."""
        raw = self.forward(X)
        normed = (raw - self._shift) / self._scale
        return normed * (self._hi - self._lo) + self._lo


def _mk_nn_fn(parents: List[str], rng: np.random.RandomState, config: MechanismConfig):
    """Create a random NN mechanism. Returns (fn, params) compatible with Mechanism."""
    logger.debug("Creating NN mechanism for %d parents", len(parents))
    nn = _RandomNN(len(parents), rng, config)
    parent_names = list(parents)

    def fn(par_vals: Dict[str, np.ndarray], params, size, rng_):
        X = np.column_stack([par_vals[p] for p in parent_names])
        return params["nn"].predict(X)

    return fn, {"nn": nn, "parent_order": parent_names}


# ------------------------------
# Heteroscedastic noise scale functions
# ------------------------------
def _mk_heteroscedastic_scale_linear(
    parents: List[str], rng: np.random.RandomState, config: MechanismConfig
):
    """Create sigma(X) = softplus(w^T X + b), clamped to scale_range."""
    w = {p: rng.normal(0, 0.5) for p in parents}
    b = rng.normal(0, 0.3)
    lo, hi = config.heteroscedastic_scale_range

    def scale_fn(par_vals, size, rng_, w=w, b=b, lo=lo, hi=hi):
        z = b
        for p, coef in w.items():
            z = z + coef * par_vals[p]
        sigma = softplus(z)
        return np.clip(sigma, lo, hi)

    return scale_fn


# ------------------------------
# SCM: structural model driven by a DAG
# ------------------------------
class SCM:
    def __init__(
        self,
        G: nx.DiGraph,
        node_types: Optional[Dict[str, str]] = None,  # {'X': 'binary'|'continuous'}
        observed_nodes: Optional[List[str]] = None,
        seed: Optional[int] = None,
        mech_config: Optional[MechanismConfig] = None,
        binary_mech_config: Optional[BinaryMechanismConfig] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        continuous_scm_profile: Optional[str] = None,
        binary_scm_profile: Optional[str] = None,
    ):
        self.G = G.copy()
        self.order = toposort(G)
        self.rng = np.random.RandomState(normalize_random_seed(seed or 0))
        self.mech_config = mech_config
        self.binary_mech_config = binary_mech_config or BinaryMechanismConfig()
        self.treatment = treatment
        self.outcome = outcome
        self.continuous_scm_profile = continuous_scm_profile
        self.binary_scm_profile = binary_scm_profile

        # assign variable types
        self.node_types = {v: "continuous" for v in self.order}
        if node_types:
            self.node_types.update(node_types)

        # default: all observed
        self.observed_nodes = (
            set(ensure_list(observed_nodes))
            if observed_nodes is not None
            else set(self.order)
        )

        # discretization state
        self.discretized_nodes: Dict[str, int] = {}  # node -> n_bins
        self._disc_edges: Dict[str, np.ndarray] = {}  # node -> bin edges

        # mechanisms per node
        self.mechanisms: Dict[str, Mechanism] = {}
        self.mechanism_kind_by_node: Dict[str, str] = {}
        self.noise_kind_by_node: Dict[str, str] = {}
        self.heteroscedastic_nodes: set = set()
        for v in self.order:
            parents = list(G.predecessors(v))
            vtype = self.node_types[v]

            # exogenous distribution
            if len(parents) == 0:
                if vtype == "binary":
                    lo, hi = self.binary_mech_config.root_prob_range
                    p = np.clip(self.rng.uniform(lo, hi), 1e-4, 1 - 1e-4)
                    # Return log-odds; Mechanism.__call__ applies sigmoid + binomial
                    # NOTE: Capture p by value via default arg to avoid late-binding closure bug
                    log_odds = logit(p)
                    self.mechanisms[v] = Mechanism(
                        fn=lambda _, __, size, rng_, lo=log_odds: np.full(size, lo),
                        params={"p": p},
                        noise=None,
                        var_type="binary",
                    )
                    self.mechanism_kind_by_node[v] = "root_binary"
                    self.noise_kind_by_node[v] = "none"
                else:
                    scale = self.rng.uniform(0.8, 1.5)
                    noise_kind = self._pick_noise_kind(root=True)
                    noise = _make_noise_dist(
                        self.rng,
                        kind=noise_kind,
                        scale=scale,
                        config=mech_config,
                    )
                    self.mechanisms[v] = Mechanism(
                        fn=lambda _, __, size, rng_: np.zeros(size),
                        params={"mu": 0.0},
                        noise=noise,
                        var_type="continuous",
                    )
                    self.mechanism_kind_by_node[v] = "root_continuous"
                    self.noise_kind_by_node[v] = noise_kind
            # endogenous
            else:
                if vtype == "binary":
                    fn, params, mechanism_kind = _make_binary_mechanism(
                        parents,
                        self.rng,
                        self.binary_mech_config,
                    )
                    self.mechanisms[v] = Mechanism(
                        fn=fn, params=params, noise=None, var_type="binary"
                    )
                    self.mechanism_kind_by_node[v] = mechanism_kind
                    self.noise_kind_by_node[v] = "none"
                else:
                    fn, params, mechanism_kind = _make_continuous_mechanism(
                        parents,
                        self.rng,
                        mech_config,
                    )

                    scale = self.rng.uniform(0.5, 1.0)
                    noise_kind = self._pick_noise_kind(root=False)
                    noise = _make_noise_dist(
                        self.rng,
                        kind=noise_kind,
                        scale=scale,
                        config=mech_config,
                    )

                    # Heteroscedastic noise scale
                    noise_scale_fn = None
                    if (
                        mech_config
                        and self.rng.random() < mech_config.prob_heteroscedastic
                    ):
                        noise_scale_fn = _mk_heteroscedastic_scale_linear(
                            parents, self.rng, mech_config
                        )
                        self.heteroscedastic_nodes.add(v)

                    self.mechanisms[v] = Mechanism(
                        fn=fn,
                        params=params,
                        noise=noise,
                        var_type="continuous",
                        noise_scale_fn=noise_scale_fn,
                    )
                    self.mechanism_kind_by_node[v] = mechanism_kind
                    self.noise_kind_by_node[v] = noise_kind

        # Decide which nodes to discretize
        if mech_config and mech_config.prob_discretized > 0:
            self._assign_discretization(mech_config)

        # Calibration pass (NN mechanisms + discretization bin edges)
        if self._needs_calibration():
            self._run_calibration_pass()

        logger.info(
            "Initialized SCM: %d nodes, %d observed, %d discretized",
            len(self.order),
            len(self.observed_nodes),
            len(self.discretized_nodes),
        )
        logger.info("SCM mechanism summary: %s", self.mechanism_summary_line())
        logger.debug("SCM node types: %s", self.node_types)
        logger.debug(
            "SCM mechanism details: %s",
            json.dumps(
                self.mechanism_diagnostics(), ensure_ascii=False, sort_keys=True
            ),
        )

    def mechanism_diagnostics(self) -> Dict[str, Any]:
        """Return mechanism/noise diagnostics for logging and trace metadata."""
        mechanism_counts = dict(Counter(self.mechanism_kind_by_node.values()))
        noise_counts = dict(Counter(self.noise_kind_by_node.values()))
        per_node: Dict[str, Dict[str, Any]] = {}
        for v in self.order:
            per_node[v] = {
                "parents": list(self.G.predecessors(v)),
                "var_type": self.node_types.get(v),
                "scm_profile": (
                    self.binary_scm_profile
                    if self.node_types.get(v) == "binary"
                    else self.continuous_scm_profile
                ),
                "mechanism_kind": self.mechanism_kind_by_node.get(v, "unknown"),
                "mechanism_family": self.mechanisms[v].params.get(
                    "binary_mechanism_family",
                    self.mechanisms[v].params.get("continuous_mechanism_family"),
                ),
                "noise_kind": self.noise_kind_by_node.get(v, "unknown"),
                "heteroscedastic": v in self.heteroscedastic_nodes,
                "discretized_bins": self.discretized_nodes.get(v),
            }
        return {
            "continuous_scm_profile": self.continuous_scm_profile,
            "binary_scm_profile": self.binary_scm_profile,
            "mechanism_counts": mechanism_counts,
            "noise_counts": noise_counts,
            "nn_nodes": sorted(
                [v for v, mech in self.mechanisms.items() if "nn" in mech.params]
            ),
            "heteroscedastic_nodes": sorted(self.heteroscedastic_nodes),
            "discretized_nodes": {
                v: int(n_bins) for v, n_bins in self.discretized_nodes.items()
            },
            "per_node": per_node,
        }

    def mechanism_summary_line(self) -> str:
        """Compact one-line mechanism summary for INFO logs."""
        diag = self.mechanism_diagnostics()
        return (
            f"mechanisms={diag['mechanism_counts']}, "
            f"noise={diag['noise_counts']}, "
            f"nn_nodes={len(diag['nn_nodes'])}, "
            f"heteroscedastic_nodes={len(diag['heteroscedastic_nodes'])}, "
            f"discretized_nodes={len(diag['discretized_nodes'])}"
        )

    def _pick_noise_kind(self, root: bool = False) -> str:
        """Choose noise distribution kind, respecting MechanismConfig mixture probs."""
        cfg = self.mech_config
        if cfg:
            prob = cfg.prob_mixture_root if root else cfg.prob_mixture_noise
            if self.rng.random() < prob:
                return "mixture"
        return self.rng.choice(["gauss", "laplace", "studentt"])

    def _assign_discretization(self, config: MechanismConfig) -> None:
        """Decide which nodes to discretize."""
        lo, hi = config.discrete_bins_range
        for v in self.order:
            if self.node_types[v] == "binary":
                continue
            # Skip treatment/outcome unless explicitly allowed
            if v == self.treatment and not config.discretize_treatment:
                continue
            if v == self.outcome and not config.discretize_outcome:
                continue
            if self.rng.random() < config.prob_discretized:
                n_bins = self.rng.randint(lo, hi + 1)
                self.discretized_nodes[v] = n_bins
                logger.debug("Will discretize node %s into %d bins", v, n_bins)

    def _needs_calibration(self) -> bool:
        """Check if any mechanism needs a calibration pass."""
        if self.discretized_nodes:
            return True
        for v, mech in self.mechanisms.items():
            if "nn" in mech.params:
                return True
        return False

    def _run_calibration_pass(self) -> None:
        """Forward-propagate calibration samples to calibrate NNs and compute bin edges."""
        cfg = self.mech_config or MechanismConfig()
        n_cal = cfg.num_calibration_samples
        cal_rng = np.random.RandomState(self.rng.randint(0, 10**9))
        logger.debug("Running calibration pass with %d samples", n_cal)

        cal_values: Dict[str, np.ndarray] = {}
        for v in self.order:
            parents = list(self.G.predecessors(v))
            if len(parents) == 0:
                # Sample from exogenous mechanism
                cal_values[v] = self.mechanisms[v]({}, size=n_cal, rng=cal_rng)
            else:
                par_vals = {p: cal_values[p] for p in parents}
                mech = self.mechanisms[v]

                # Calibrate NN if present
                if "nn" in mech.params:
                    nn = mech.params["nn"]
                    parent_names = mech.params["parent_order"]
                    X = np.column_stack([par_vals[p] for p in parent_names])
                    nn.calibrate(X)

                cal_values[v] = mech(par_vals, size=n_cal, rng=cal_rng)

            # Compute discretization bin edges
            if v in self.discretized_nodes:
                n_bins = self.discretized_nodes[v]
                vals = cal_values[v]
                lo, hi = float(np.min(vals)), float(np.max(vals))
                self._disc_edges[v] = np.linspace(lo - 1e-8, hi + 1e-8, n_bins + 1)
                logger.debug(
                    "Computed bin edges for %s: %d bins over [%.3f, %.3f]",
                    v,
                    n_bins,
                    lo,
                    hi,
                )

    def _draw_base_noise(
        self, n: int, rng: np.random.RandomState
    ) -> Dict[str, np.ndarray]:
        """Draw exogenous noise values for all nodes."""
        noise: Dict[str, np.ndarray] = {}
        for v in self.order:
            mech = self.mechanisms[v]
            if mech.var_type == "continuous":
                if mech.noise is not None:
                    eps = np.asarray(mech.noise(n), dtype=float)
                else:
                    eps = np.zeros(n, dtype=float)
            elif mech.var_type == "binary":
                # Store a base Uniform(0,1) draw to enable deterministic
                # thresholding under different interventions/worlds.
                eps = rng.uniform(0.0, 1.0, size=n).astype(float)
            else:
                raise ValueError(f"Unknown mechanism var_type {mech.var_type!r}")

            if eps.shape[0] != n:
                raise ValueError(
                    f"Noise generator for node {v!r} returned length {eps.shape[0]}, expected {n}"
                )
            noise[v] = eps
        return noise

    def _coerce_noise_map(self, noise_data: Any) -> Dict[str, np.ndarray]:
        """Normalize provided noise into {node -> np.ndarray} map."""
        if isinstance(noise_data, pd.DataFrame):
            data = {c: noise_data[c].to_numpy() for c in noise_data.columns}
        elif isinstance(noise_data, dict):
            data = noise_data
        else:
            raise TypeError(
                "noise_data must be a pandas DataFrame or dict[node, array-like]"
            )

        if not data:
            raise ValueError("noise_data is empty")

        n = None
        noise_map: Dict[str, np.ndarray] = {}
        for v in self.order:
            if v not in data:
                raise KeyError(f"noise_data missing node {v!r}")
            arr = np.asarray(data[v], dtype=float).reshape(-1)
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise ValueError(
                    f"noise_data length mismatch for node {v!r}: got {arr.shape[0]}, expected {n}"
                )
            noise_map[v] = arr

        return noise_map

    @staticmethod
    def _coerce_intervention_array(
        value: Any,
        n: int,
        node: str,
    ) -> np.ndarray:
        """Normalize scalar/array intervention values to length-n arrays."""
        if isinstance(value, numbers.Real):
            return np.full(n, float(value), dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(
                f"Intervention for node {node!r} has length {arr.shape[0]}, expected {n}"
            )
        return arr

    def _simulate_with_base_noise(
        self,
        base_noise: Dict[str, np.ndarray],
        interventions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Simulate one world from fixed exogenous noise."""
        n = len(next(iter(base_noise.values())))
        rng = np.random.RandomState(
            normalize_random_seed(seed or self.rng.randint(0, 10**9))
        )
        interventions = interventions or {}
        values: Dict[str, np.ndarray] = {
            v: np.zeros(n, dtype=float) for v in self.order
        }

        for v in self.order:
            if v in interventions:
                values[v] = self._coerce_intervention_array(interventions[v], n, v)
                continue

            mech = self.mechanisms[v]
            parents = list(self.G.predecessors(v))
            par_vals = {p: values[p] for p in parents}

            if mech.var_type == "continuous":
                mean = np.asarray(mech.fn(par_vals, mech.params, n, rng), dtype=float)
                eps = np.asarray(base_noise[v], dtype=float)
                if mech.noise_scale_fn is not None:
                    eps = eps * np.asarray(
                        mech.noise_scale_fn(par_vals, n, rng), dtype=float
                    )
                values[v] = mean + eps
            elif mech.var_type == "binary":
                log_odds = np.asarray(
                    mech.fn(par_vals, mech.params, n, rng), dtype=float
                )
                prob = np.clip(sigmoid(log_odds), 1e-6, 1 - 1e-6)
                u = np.asarray(base_noise[v], dtype=float)
                values[v] = (u < prob).astype(float)
            else:
                raise ValueError(f"Unknown mechanism var_type {mech.var_type!r}")

        # Apply discretization only to emitted values (same semantics as sample()).
        values_out = {v: np.array(vals, copy=True) for v, vals in values.items()}
        for v, n_bins in self.discretized_nodes.items():
            if v in interventions:
                continue
            if v in self._disc_edges:
                values_out[v] = (
                    np.digitize(values_out[v], self._disc_edges[v]) - 1
                ).astype(float)
                values_out[v] = np.clip(values_out[v], 0, n_bins - 1)

        return values_out

    def sample(
        self,
        n: int,
        interventions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> pd.DataFrame:
        """Draw n samples from the SCM. `interventions` is a dict like {'X': 1.0} for do(X=1)."""
        logger.debug(
            "SCM.sample: n=%d, interventions=%s, seed=%s",
            n,
            list(interventions.keys()) if interventions else None,
            seed,
        )
        rng = np.random.RandomState(
            normalize_random_seed(seed or self.rng.randint(0, 10**9))
        )
        base_noise = self._draw_base_noise(n, rng)
        values = self._simulate_with_base_noise(
            base_noise=base_noise,
            interventions=interventions,
            seed=seed,
        )
        cols = (
            self.order
            if return_all_nodes
            else [v for v in self.order if v in self.observed_nodes]
        )
        return pd.DataFrame({v: values[v] for v in cols})

    def sample_with_noise(
        self,
        n: int,
        interventions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Draw samples and return both observed data and exogenous noise table."""
        logger.debug(
            "SCM.sample_with_noise: n=%d, interventions=%s, seed=%s",
            n,
            list(interventions.keys()) if interventions else None,
            seed,
        )
        rng = np.random.RandomState(
            normalize_random_seed(seed or self.rng.randint(0, 10**9))
        )
        base_noise = self._draw_base_noise(n, rng)
        values = self._simulate_with_base_noise(
            base_noise=base_noise,
            interventions=interventions,
            seed=seed,
        )
        cols = (
            self.order
            if return_all_nodes
            else [v for v in self.order if v in self.observed_nodes]
        )
        obs_df = pd.DataFrame({v: values[v] for v in cols})
        noise_df = pd.DataFrame({v: base_noise[v] for v in self.order})
        return obs_df, noise_df

    def simulate_from_noise(
        self,
        noise_data: Any,
        interventions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> pd.DataFrame:
        """Simulate a world from fixed exogenous noise with optional interventions."""
        base_noise = self._coerce_noise_map(noise_data)
        values = self._simulate_with_base_noise(
            base_noise=base_noise,
            interventions=interventions,
            seed=seed,
        )
        cols = (
            self.order
            if return_all_nodes
            else [v for v in self.order if v in self.observed_nodes]
        )
        return pd.DataFrame({v: values[v] for v in cols})

    def nested_from_noise(
        self,
        noise_data: Any,
        *,
        outer_interventions: Optional[Dict[str, Any]] = None,
        inner_interventions: Optional[Dict[str, Any]] = None,
        inner_nodes: Optional[List[str]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> pd.DataFrame:
        """Simulate nested counterfactuals such as Y_{x, M_{x'}} from fixed noise."""
        inner_nodes = inner_nodes or []
        for node in inner_nodes:
            if node not in self.G:
                raise KeyError(f"Inner intervention node {node!r} not in graph")

        base_noise = self._coerce_noise_map(noise_data)
        inner_values = self._simulate_with_base_noise(
            base_noise=base_noise,
            interventions=inner_interventions,
            seed=seed,
        )

        combined_interventions: Dict[str, Any] = dict(outer_interventions or {})
        for node in inner_nodes:
            combined_interventions[node] = inner_values[node]

        outer_values = self._simulate_with_base_noise(
            base_noise=base_noise,
            interventions=combined_interventions,
            seed=seed,
        )
        cols = (
            self.order
            if return_all_nodes
            else [v for v in self.order if v in self.observed_nodes]
        )
        return pd.DataFrame({v: outer_values[v] for v in cols})

    def copy(self) -> "SCM":
        """Return a deep copy of this SCM."""
        return copy.deepcopy(self)

    # Monte Carlo "true effect" via interventions
    def true_effect(
        self,
        treatment: str,
        outcome: str,
        x0: float = 0.0,
        x1: float = 1.0,
        n_mc: int = 200_000,
        seed: Optional[int] = None,
    ) -> float:

        rng_seed = normalize_random_seed(seed or self.rng.randint(0, 10**9))
        y1 = self.sample(n_mc, interventions={treatment: x1}, seed=rng_seed)[
            outcome
        ].mean()
        y0 = self.sample(
            n_mc,
            interventions={treatment: x0},
            seed=offset_random_seed(rng_seed, 1),
        )[outcome].mean()

        val = float(y1 - y0)
        logger.info(
            "SCM.true_effect: ATE(%s->%s; %s vs %s) = %.6f (n_mc=%d)",
            treatment,
            outcome,
            x1,
            x0,
            val,
            n_mc,
        )
        return val


# ------------------------------
# High-level wrapper: from SampledGraph -> SCM -> DataFrames
# ------------------------------
class DataGenerator:
    def __init__(
        self,
        sg: SampledGraph,
        node_types: Optional[Dict[str, str]] = None,
        force_treatment_binary: bool = False,
        force_outcome_continuous: bool = False,
        seed: Optional[int] = None,
        mech_config: Optional[MechanismConfig] = None,
        binary_mech_config: Optional[BinaryMechanismConfig] = None,
        continuous_scm_profile: Optional[str] = None,
        binary_scm_profile: Optional[str] = None,
    ):
        logger.debug(
            "Initializing DataGenerator (force_treatment_binary=%s, force_outcome_continuous=%s, seed=%s)",
            force_treatment_binary,
            force_outcome_continuous,
            seed,
        )
        self.sg = sg
        self.G = sg.graph
        self.continuous_scm_profile = continuous_scm_profile
        self.binary_scm_profile = binary_scm_profile
        # default types: all continuous unless forced/overridden
        inferred_types = {v: "continuous" for v in self.G.nodes()}
        if force_treatment_binary:
            inferred_types[sg.treatment] = "binary"
        if force_outcome_continuous:
            inferred_types[sg.outcome] = "continuous"
        # allow caller overrides
        if node_types:
            inferred_types.update(node_types)

        self.node_types = inferred_types
        observed_nodes = sg.observed_nodes or list(self.G.nodes())

        # Construct a proper SCM model
        self.scm = SCM(
            self.G,
            node_types=self.node_types,
            observed_nodes=observed_nodes,
            seed=seed,
            mech_config=mech_config,
            binary_mech_config=binary_mech_config,
            treatment=sg.treatment,
            outcome=sg.outcome,
            continuous_scm_profile=continuous_scm_profile,
            binary_scm_profile=binary_scm_profile,
        )
        logger.info(
            "DataGenerator created (motif=%s, treatment=%s, outcome=%s)",
            sg.motif,
            sg.treatment,
            sg.outcome,
        )

    def copy(self) -> "DataGenerator":
        """Return a deep copy of this DataGenerator."""
        return copy.deepcopy(self)

    def sample_observational(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        logger.debug("DataGenerator.sample_observational: n=%d", n)
        return self.scm.sample(n=n, interventions=None, seed=seed)

    def sample_with_noise(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Draw observational samples and return exogenous noise per unit."""
        logger.debug("DataGenerator.sample_with_noise: n=%d", n)
        return self.scm.sample_with_noise(n=n, interventions=None, seed=seed)

    def sample_do(
        self, t_value: float, n: int, seed: Optional[int] = None
    ) -> pd.DataFrame:
        logger.debug(
            "DataGenerator.sample_do: do(%s=%.3f), n=%d",
            self.sg.treatment,
            t_value,
            n,
        )
        return self.scm.sample(
            n=n, interventions={self.sg.treatment: t_value}, seed=seed
        )

    def simulate_from_noise(
        self,
        noise_data: Any,
        interventions: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> pd.DataFrame:
        """Simulate a world under interventions using fixed exogenous noise."""
        return self.scm.simulate_from_noise(
            noise_data=noise_data,
            interventions=interventions,
            seed=seed,
            return_all_nodes=return_all_nodes,
        )

    def nested_from_noise(
        self,
        noise_data: Any,
        *,
        outer_interventions: Optional[Dict[str, Any]] = None,
        inner_interventions: Optional[Dict[str, Any]] = None,
        inner_nodes: Optional[List[str]] = None,
        seed: Optional[int] = None,
        return_all_nodes: bool = False,
    ) -> pd.DataFrame:
        """Simulate nested counterfactual worlds using fixed exogenous noise."""
        return self.scm.nested_from_noise(
            noise_data=noise_data,
            outer_interventions=outer_interventions,
            inner_interventions=inner_interventions,
            inner_nodes=inner_nodes,
            seed=seed,
            return_all_nodes=return_all_nodes,
        )

    def true_ate(
        self,
        x0: float = 0.0,
        x1: float = 1.0,
        n_mc: int = 200_000,
        seed: Optional[int] = None,
    ) -> float:
        logger.debug("DataGenerator.true_ate: x0=%.3f, x1=%.3f, n_mc=%d", x0, x1, n_mc)
        return self.scm.true_effect(
            self.sg.treatment,
            self.sg.outcome,
            x0=x0,
            x1=x1,
            n_mc=n_mc,
            seed=seed,
        )

    def build_observation_data(
        self,
        conceptual_data: pd.DataFrame,
        observation_config: Optional[ObservationConfig] = None,
        *,
        node_types: Optional[Dict[str, str]] = None,
        node_name_map: Optional[Dict[str, str]] = None,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
    ) -> ObservationData:
        """Build public proxy observations from conceptual observed-node data."""
        resolved_node_types = dict(self.node_types)
        if node_types:
            resolved_node_types.update(
                {
                    str(node): str(dtype).strip().lower()
                    for node, dtype in dict(node_types).items()
                }
            )
        model = ObservationModel(
            sg=self.sg,
            node_types=resolved_node_types,
            config=observation_config,
            seed=seed,
            node_name_map=node_name_map,
        )
        return model.build(
            conceptual_data=conceptual_data,
            train_ratio=train_ratio,
        )


# ------------------------------
# Observation-model helpers
# ------------------------------
def _serialize_proxy_params(value: Any) -> Any:
    """Convert proxy mechanism params into JSON-serializable values.

    Kept in this module because the serializer depends on mechanism internals
    (e.g., random-NN parameter objects) that are specific to this data pipeline.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serialize_proxy_params(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_proxy_params(v) for v in value]
    if isinstance(value, _RandomNN):
        return {
            "weights": [list(w.shape) for w in value.weights],
            "biases": [list(b.shape) for b in value.biases],
            "output_range": [float(value._lo), float(value._hi)],
        }
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    return repr(value)


def _eligible_proxified_nodes(sg: SampledGraph) -> List[str]:
    """Return conceptually important observed nodes that may be proxified.

    The basic benchmark variants keep the public outcome directly observed.
    """
    observed = list(sg.observed_nodes or sg.graph.nodes())
    latent = set(sg.latent_nodes or [])
    roles = get_node_roles(sg.graph, sg.treatment, sg.outcome)

    eligible = {sg.treatment}
    for key in (
        "on_causal_path",
        "potential_confounders",
        "ancestors_of_treatment",
        "ancestors_of_outcome",
        "descendants_of_treatment",
    ):
        eligible.update(roles.get(key, []))

    eligible.update(n for n in observed if sg.graph.degree(n) > 0)
    return [
        node
        for node in observed
        if node in eligible and node not in latent and node != sg.outcome
    ]


def _pick_proxy_var_type(
    latent_type: str,
    rng: np.random.RandomState,
    *,
    is_outcome: bool,
    config: ObservationConfig,
) -> str:
    """Choose whether a proxy column is continuous or binary."""
    latent_type = str(latent_type).strip().lower()
    if latent_type == "binary":
        if is_outcome and config.outcome_preserve_type:
            return "binary"
        return "binary" if rng.random() < 0.6 else "continuous"

    if is_outcome and config.outcome_preserve_type:
        return "continuous"
    return "continuous" if rng.random() < 0.75 else "binary"


def _make_proxy_continuous_spec(
    latent_name: str,
    proxy_name: str,
    latent_type: str,
    rng: np.random.RandomState,
    *,
    allow_heteroscedastic: bool,
    noise_scale_range: Tuple[float, float] = (0.35, 0.9),
    mech_config: Optional[MechanismConfig] = None,
    latent_values: Optional[np.ndarray] = None,
) -> ProxyColumnSpec:
    """Create one continuous proxy mechanism.

    When *mech_config* is provided the same mechanism families available to
    SCM nodes are used (structural family, mixture noise, heteroscedastic), with
    probabilities drawn from the config.  When ``None`` the
    handcrafted-only path is used.
    """
    # -- mechanism function ------------------------------------------------
    fn, params, selected_kind = _make_continuous_mechanism(
        [latent_name],
        rng,
        mech_config,
    )
    family = params.get("continuous_mechanism_family", "handcrafted_continuous")
    if family == "generic_nn":
        mechanism_kind = "proxy_nn"
        # NN requires calibration; use latent values when available.
        if latent_values is not None:
            params["nn"].calibrate(latent_values.reshape(-1, 1))
    elif selected_kind == "endogenous_continuous_handcrafted_continuous":
        mechanism_kind = "proxy_continuous"
    else:
        mechanism_kind = f"proxy_continuous_{family}"

    # -- noise distribution ------------------------------------------------
    lo, hi = noise_scale_range
    base_noise_scale = float(rng.uniform(lo, hi))
    if mech_config and rng.random() < mech_config.prob_mixture_noise:
        noise_kind = "mixture"
    else:
        noise_kind = rng.choice(["gauss", "laplace", "studentt"])
    noise = _make_noise_dist(
        rng, kind=noise_kind, scale=base_noise_scale, config=mech_config
    )

    # -- heteroscedastic scale ---------------------------------------------
    noise_scale_fn = None
    het_prob = mech_config.prob_heteroscedastic if mech_config else 0.35
    het_cfg = mech_config or MechanismConfig()
    if allow_heteroscedastic and rng.random() < het_prob:
        noise_scale_fn = _mk_heteroscedastic_scale_linear([latent_name], rng, het_cfg)
        params = dict(params)
        params["heteroscedastic"] = True

    mechanism = Mechanism(
        fn=fn,
        params=params,
        noise=noise,
        var_type="continuous",
        noise_scale_fn=noise_scale_fn,
    )
    return ProxyColumnSpec(
        column_name=proxy_name,
        latent_name=latent_name,
        latent_type=latent_type,
        proxy_type="continuous",
        mechanism_kind=mechanism_kind,
        noise_kind=str(noise_kind),
        base_noise_scale=base_noise_scale,
        params=params,
        mechanism=mechanism,
    )


def _make_proxy_binary_spec(
    latent_name: str,
    proxy_name: str,
    latent_type: str,
    rng: np.random.RandomState,
    mech_config: Optional[MechanismConfig] = None,
) -> ProxyColumnSpec:
    """Create one binary proxy mechanism."""
    shift = mech_config.nl_shift_range if mech_config else (0.0, 0.0)
    fn, params = _mk_bin_fn([latent_name], rng, nl_shift_range=shift)
    mechanism = Mechanism(fn=fn, params=params, noise=None, var_type="binary")
    return ProxyColumnSpec(
        column_name=proxy_name,
        latent_name=latent_name,
        latent_type=latent_type,
        proxy_type="binary",
        mechanism_kind="proxy_binary",
        noise_kind="none",
        base_noise_scale=0.0,
        params=params,
        mechanism=mechanism,
    )


def _sample_proxy_column_spec(
    latent_name: str,
    latent_type: str,
    proxy_index: int,
    rng: np.random.RandomState,
    *,
    is_outcome: bool,
    config: ObservationConfig,
    latent_values: Optional[np.ndarray] = None,
) -> ProxyColumnSpec:
    """Sample one proxy column spec for a latent variable."""
    proxy_name = f"{latent_name}{proxy_index}"
    proxy_type = _pick_proxy_var_type(
        latent_type,
        rng,
        is_outcome=is_outcome,
        config=config,
    )
    if proxy_type == "binary":
        return _make_proxy_binary_spec(
            latent_name,
            proxy_name,
            latent_type,
            rng,
            mech_config=config.proxy_mechanism_config,
        )
    return _make_proxy_continuous_spec(
        latent_name,
        proxy_name,
        latent_type,
        rng,
        allow_heteroscedastic=(config.allow_heteroscedastic and not is_outcome),
        noise_scale_range=config.proxy_noise_scale_range,
        mech_config=config.proxy_mechanism_config,
        latent_values=latent_values,
    )


def _rank_coordinate(values: pd.Series) -> np.ndarray:
    """Map a scalar variable onto [0, 1] rank space."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    if np.nanmax(arr) - np.nanmin(arr) < 1e-10:
        return np.full(arr.shape[0], 0.5, dtype=float)
    return values.rank(method="average", pct=True).to_numpy(dtype=float)


def _quantile_strata(values: pd.Series, n_bins: int) -> np.ndarray:
    """Assign each row to a latent support bin."""
    numeric = pd.to_numeric(values, errors="coerce")
    unique = np.unique(numeric.dropna().to_numpy(dtype=float))
    if unique.size <= max(2, min(n_bins, 4)):
        mapping = {val: i for i, val in enumerate(sorted(unique.tolist()))}
        return numeric.map(mapping).fillna(-1).to_numpy(dtype=int)

    ranks = numeric.rank(method="first", pct=True)
    bins = pd.qcut(
        ranks,
        q=min(int(n_bins), len(ranks.dropna())),
        labels=False,
        duplicates="drop",
    )
    return bins.fillna(-1).to_numpy(dtype=int)


def _pick_space_filling_index(
    candidate_indices: List[int],
    coords: np.ndarray,
    selected_indices: List[int],
    rng: np.random.RandomState,
) -> Optional[int]:
    """Pick the candidate farthest from already-selected latent coordinates."""
    if not candidate_indices:
        return None
    if not selected_indices:
        return int(rng.choice(candidate_indices))

    selected_coords = coords[selected_indices]
    best_idx = None
    best_score = -np.inf
    for idx in candidate_indices:
        diff = selected_coords - coords[idx]
        min_dist = float(np.min(np.linalg.norm(diff, axis=1)))
        if min_dist > best_score + 1e-12:
            best_idx = idx
            best_score = min_dist
    return int(best_idx) if best_idx is not None else None


def _select_calibration_indices(
    latent_data: pd.DataFrame,
    proxified_nodes: List[str],
    config: ObservationConfig,
    rng: np.random.RandomState,
) -> List[int]:
    """Select calibration rows with marginal quantile coverage + space filling."""
    n_rows = len(latent_data)
    if n_rows == 0 or not proxified_nodes:
        return []

    cal_cfg = config.calibration
    target = int(round(n_rows * float(cal_cfg.fraction)))
    target = max(int(cal_cfg.min_rows), target)
    target = min(target, n_rows)
    if target <= 0:
        return []

    coord_cols = [_rank_coordinate(latent_data[node]) for node in proxified_nodes]
    coords = np.column_stack(coord_cols)

    selected: List[int] = []
    selected_set = set()

    if len(proxified_nodes) == 1:
        strata = _quantile_strata(
            latent_data[proxified_nodes[0]], cal_cfg.quantile_bins
        )
        unique_bins = [int(b) for b in sorted(set(strata.tolist())) if b >= 0]
        while len(selected) < target:
            added = False
            for bin_id in unique_bins:
                candidates = [
                    idx
                    for idx, row_bin in enumerate(strata.tolist())
                    if row_bin == bin_id and idx not in selected_set
                ]
                chosen = _pick_space_filling_index(candidates, coords, selected, rng)
                if chosen is None:
                    continue
                selected.append(chosen)
                selected_set.add(chosen)
                added = True
                if len(selected) >= target:
                    break
            if not added:
                break
        return selected

    for node in proxified_nodes:
        strata = _quantile_strata(latent_data[node], cal_cfg.quantile_bins)
        unique_bins = [int(b) for b in sorted(set(strata.tolist())) if b >= 0]
        for bin_id in unique_bins:
            candidates = [
                idx
                for idx, row_bin in enumerate(strata.tolist())
                if row_bin == bin_id and idx not in selected_set
            ]
            chosen = _pick_space_filling_index(candidates, coords, selected, rng)
            if chosen is None:
                continue
            selected.append(chosen)
            selected_set.add(chosen)
            if len(selected) >= target:
                return selected

    while len(selected) < target:
        remaining = [idx for idx in range(n_rows) if idx not in selected_set]
        chosen = _pick_space_filling_index(remaining, coords, selected, rng)
        if chosen is None:
            break
        selected.append(chosen)
        selected_set.add(chosen)

    return selected


def _local_window_indices(
    latent_values: np.ndarray,
    center: float,
    window_size: int,
) -> np.ndarray:
    """Indices of the nearest latent values around one support point."""
    distances = np.abs(latent_values - float(center))
    k = min(max(2, int(window_size)), len(latent_values))
    return np.argpartition(distances, kth=k - 1)[:k]


def _grid_over_realized_support(
    latent_values: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """Create a 1-D grid over the realized latent support."""
    unique = np.unique(np.asarray(latent_values, dtype=float))
    if unique.size <= 2:
        return unique.astype(float)
    lo, hi = float(np.min(unique)), float(np.max(unique))
    return np.linspace(lo, hi, num=max(8, int(grid_size)))


def _compute_binary_latent_proxy_diagnostics(
    latent_values: np.ndarray,
    bundle: ProxyBundle,
) -> Dict[str, Any]:
    """Information diagnostics when the latent variable is binary.

    The local-regression approach used for continuous latents degenerates when
    the support is {0, 1}.  Instead we measure class separation per proxy:
      - Continuous proxy: squared standardized mean difference (Cohen's d^2).
      - Binary proxy: mutual information via the 2x2 contingency table.
    The bundle score is the sum over proxies.
    """
    z = np.asarray(latent_values, dtype=float)
    mask0 = z < 0.5
    mask1 = ~mask0
    n0, n1 = int(mask0.sum()), int(mask1.sum())
    if n0 < 2 or n1 < 2:
        return {
            "diagnostic_type": "binary_latent",
            "bundle_score": 0.0,
            "min_information": 0.0,
            "average_information": 0.0,
            "per_proxy": {},
            "grid": [0.0, 1.0],
            "bundle_information_grid": [0.0, 0.0],
            "lower_tail_quantiles": {},
        }

    bundle_score = 0.0
    proxy_details: Dict[str, Any] = {}

    for spec in bundle.proxy_columns:
        vals = bundle.proxy_frame[spec.column_name].to_numpy(dtype=float)
        v0, v1 = vals[mask0], vals[mask1]

        if spec.proxy_type == "binary":
            # Mutual information from 2x2 table
            p_z1 = n1 / len(z)
            p_z0 = 1.0 - p_z1
            p_w1_z0 = float(np.clip(np.mean(v0), 1e-6, 1 - 1e-6))
            p_w1_z1 = float(np.clip(np.mean(v1), 1e-6, 1 - 1e-6))
            p_w1 = p_z0 * p_w1_z0 + p_z1 * p_w1_z1
            p_w0 = 1.0 - p_w1

            mi = 0.0
            for p_wz, p_w, p_zz in [
                (p_z0 * p_w1_z0, p_w1, p_z0),
                (p_z0 * (1 - p_w1_z0), p_w0, p_z0),
                (p_z1 * p_w1_z1, p_w1, p_z1),
                (p_z1 * (1 - p_w1_z1), p_w0, p_z1),
            ]:
                if p_wz > 1e-12 and p_w > 1e-12 and p_zz > 1e-12:
                    mi += p_wz * np.log(p_wz / (p_w * p_zz))

            proxy_details[spec.column_name] = {
                "proxy_type": "binary",
                "mutual_information": float(mi),
                "p_w1_z0": p_w1_z0,
                "p_w1_z1": p_w1_z1,
            }
            bundle_score += mi
        else:
            # Squared standardized mean difference (Cohen's d^2)
            mu0, mu1 = float(np.mean(v0)), float(np.mean(v1))
            var0, var1 = float(np.var(v0, ddof=1)), float(np.var(v1, ddof=1))
            pooled_var = ((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1)
            d_sq = (mu1 - mu0) ** 2 / max(pooled_var, 1e-8)

            proxy_details[spec.column_name] = {
                "proxy_type": "continuous",
                "cohen_d_squared": float(d_sq),
                "mean_z0": mu0,
                "mean_z1": mu1,
                "pooled_var": float(pooled_var),
            }
            bundle_score += d_sq

    # Map to the same output schema so thresholds work uniformly.
    # Use bundle_score at both grid points (binary latent has 2-point support).
    return {
        "diagnostic_type": "binary_latent",
        "bundle_score": float(bundle_score),
        "grid": [0.0, 1.0],
        "bundle_information_grid": [bundle_score, bundle_score],
        "min_information": float(bundle_score),
        "average_information": float(bundle_score),
        "lower_tail_quantiles": {
            f"q{int(round(q * 100)):02d}": float(bundle_score) for q in (0.1, 0.25)
        },
        "per_proxy": proxy_details,
    }


def _compute_proxy_information_diagnostics(
    latent_values: np.ndarray,
    bundle: ProxyBundle,
    config: ObservationConfig,
) -> Dict[str, Any]:
    """Estimate local information diagnostics from realized latent/proxy pairs.

    Dispatches to a binary-latent-specific path when the latent support has
    at most 2 unique values (the local-regression approach degenerates there).
    """
    latent_values = np.asarray(latent_values, dtype=float)
    unique_latent = np.unique(latent_values)

    # Binary latent: class-separation diagnostics instead of local regression.
    if unique_latent.size <= 2:
        return _compute_binary_latent_proxy_diagnostics(latent_values, bundle)

    grid = _grid_over_realized_support(latent_values, config.info_grid_size)
    window_size = max(
        12,
        int(np.ceil(len(latent_values) * float(config.local_info_window_fraction))),
    )

    bundle_info = np.zeros_like(grid, dtype=float)
    proxy_details: Dict[str, Any] = {}

    for spec in bundle.proxy_columns:
        values = bundle.proxy_frame[spec.column_name].to_numpy(dtype=float)
        if spec.proxy_type == "binary":
            probs = []
            local_slopes = []
            for center in grid:
                idx = _local_window_indices(latent_values, center, window_size)
                x_local = latent_values[idx].reshape(-1, 1)
                y_local = values[idx]
                if np.unique(y_local).size < 2 or np.std(x_local) < 1e-8:
                    probs.append(float(np.clip(np.mean(y_local), 1e-4, 1 - 1e-4)))
                    local_slopes.append(0.0)
                    continue

                model = LogisticRegression(C=1.0, max_iter=1000)
                model.fit(x_local, y_local.astype(int))
                prob_center = float(
                    model.predict_proba(np.array([[float(center)]], dtype=float))[0, 1]
                )
                probs.append(float(np.clip(prob_center, 1e-4, 1 - 1e-4)))
                local_slopes.append(float(model.coef_[0, 0]))

            prob_grid = np.asarray(probs, dtype=float)
            deriv = np.asarray(local_slopes, dtype=float)
            info = prob_grid * (1.0 - prob_grid) * (deriv**2)
            proxy_details[spec.column_name] = {
                "proxy_type": spec.proxy_type,
                "probability_grid": prob_grid.tolist(),
                "logit_derivative_grid": deriv.tolist(),
                "information_grid": info.tolist(),
            }
        else:
            mean_grid = []
            var_grid = []
            deriv_grid = []
            for center in grid:
                idx = _local_window_indices(latent_values, center, window_size)
                x_local = latent_values[idx]
                local_vals = values[idx]
                if np.std(x_local) < 1e-8:
                    local_mean = float(np.mean(local_vals))
                    slope = 0.0
                    residual_var = float(np.var(local_vals - local_mean)) + 1e-6
                else:
                    x_centered = x_local - np.mean(x_local)
                    y_centered = local_vals - np.mean(local_vals)
                    slope = float(
                        np.dot(x_centered, y_centered)
                        / max(np.dot(x_centered, x_centered), 1e-8)
                    )
                    intercept = float(np.mean(local_vals) - slope * np.mean(x_local))
                    local_mean = intercept + slope * float(center)
                    residual = local_vals - (intercept + slope * x_local)
                    residual_var = float(np.var(residual)) + 1e-6
                mean_grid.append(local_mean)
                var_grid.append(residual_var)
                deriv_grid.append(slope)
            mean_grid_arr = np.asarray(mean_grid, dtype=float)
            var_grid_arr = np.asarray(var_grid, dtype=float)
            deriv = np.asarray(deriv_grid, dtype=float)
            info = (deriv**2) / np.maximum(var_grid_arr, 1e-6)
            proxy_details[spec.column_name] = {
                "proxy_type": spec.proxy_type,
                "mean_grid": mean_grid_arr.tolist(),
                "variance_grid": var_grid_arr.tolist(),
                "derivative_grid": deriv.tolist(),
                "information_grid": info.tolist(),
            }

        bundle_info += info

    thresholds = config.info_thresholds
    lower_tail = {
        f"q{int(round(q * 100)):02d}": float(np.quantile(bundle_info, q))
        for q in thresholds.lower_tail_quantiles
    }
    return {
        "diagnostic_type": "continuous_latent",
        "grid": grid.tolist(),
        "bundle_information_grid": bundle_info.tolist(),
        "min_information": float(np.min(bundle_info)),
        "average_information": float(np.mean(bundle_info)),
        "lower_tail_quantiles": lower_tail,
        "per_proxy": proxy_details,
    }


def _passes_information_thresholds(
    diagnostics: Dict[str, Any],
    config: ObservationConfig,
    *,
    is_outcome: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """Check whether a proxy bundle is informative enough to keep."""
    thresholds = config.info_thresholds
    multiplier = float(config.outcome_info_multiplier) if is_outcome else 1.0
    reasons: Dict[str, Any] = {}

    min_info = float(diagnostics.get("min_information", 0.0))
    avg_info = float(diagnostics.get("average_information", 0.0))
    lower_tail = diagnostics.get("lower_tail_quantiles", {})

    if min_info < thresholds.min_information * multiplier:
        reasons["min_information"] = {
            "value": min_info,
            "required": thresholds.min_information * multiplier,
        }
    if avg_info < thresholds.average_information_min * multiplier:
        reasons["average_information"] = {
            "value": avg_info,
            "required": thresholds.average_information_min * multiplier,
        }

    for q, required in zip(
        thresholds.lower_tail_quantiles,
        thresholds.lower_tail_minima,
    ):
        key = f"q{int(round(q * 100)):02d}"
        value = float(lower_tail.get(key, 0.0))
        if value < float(required) * multiplier:
            reasons[key] = {
                "value": value,
                "required": float(required) * multiplier,
            }

    return len(reasons) == 0, reasons


def _sample_proxy_bundle(
    latent_name: str,
    latent_values: np.ndarray,
    latent_type: str,
    rng: np.random.RandomState,
    *,
    is_outcome: bool,
    config: ObservationConfig,
) -> ProxyBundle:
    """Sample and validate one proxy bundle for a latent node."""
    dim_lo, dim_hi = config.proxy_dim_range
    dim_lo, dim_hi = int(dim_lo), int(dim_hi)

    for attempt in range(1, int(config.max_bundle_resamples) + 1):
        n_dims = dim_hi if dim_lo == dim_hi else rng.randint(dim_lo, dim_hi + 1)
        specs: List[ProxyColumnSpec] = []
        proxy_data: Dict[str, np.ndarray] = {}

        for proxy_idx in range(1, n_dims + 1):
            spec = _sample_proxy_column_spec(
                latent_name,
                latent_type,
                proxy_idx,
                rng,
                is_outcome=is_outcome,
                config=config,
                latent_values=np.asarray(latent_values, dtype=float),
            )
            parent_values = {latent_name: np.asarray(latent_values, dtype=float)}
            proxy_data[spec.column_name] = np.asarray(
                spec.mechanism(parent_values, size=len(latent_values), rng=rng),
                dtype=float,
            )
            specs.append(spec)

        proxy_frame = pd.DataFrame(proxy_data)
        bundle = ProxyBundle(
            latent_name=latent_name,
            latent_type=str(latent_type),
            proxy_columns=specs,
            proxy_frame=proxy_frame,
            information_diagnostics={},
        )
        diagnostics = _compute_proxy_information_diagnostics(
            np.asarray(latent_values, dtype=float),
            bundle,
            config,
        )
        ok, failures = _passes_information_thresholds(
            diagnostics,
            config,
            is_outcome=is_outcome,
        )
        diagnostics["rejected_thresholds"] = failures
        diagnostics["sampling_attempt"] = attempt
        bundle.information_diagnostics = diagnostics
        if ok:
            return bundle

    raise ValueError(
        f"Failed to sample an informative proxy bundle for {latent_name!r} after "
        f"{config.max_bundle_resamples} attempts"
    )


def _ensure_unique_proxy_column_names(
    bundle: ProxyBundle,
    protected_names: set,
) -> None:
    """Rename proxy columns in-place to avoid collisions with existing columns."""
    rename_map: Dict[str, str] = {}

    for spec in bundle.proxy_columns:
        original_name = str(spec.column_name)
        candidate = original_name
        if candidate in protected_names:
            stem = str(bundle.latent_name)
            suffix = (
                original_name[len(stem) :] if original_name.startswith(stem) else ""
            )
            start_idx = int(suffix) if suffix.isdigit() else 1
            idx = max(1, start_idx)
            candidate = f"{stem}{idx}"
            while candidate in protected_names:
                idx += 1
                candidate = f"{stem}{idx}"

        rename_map[original_name] = candidate
        spec.column_name = candidate
        protected_names.add(candidate)

    bundle.proxy_frame = bundle.proxy_frame.rename(columns=rename_map)

    per_proxy = bundle.information_diagnostics.get("per_proxy")
    if isinstance(per_proxy, dict) and per_proxy:
        bundle.information_diagnostics["per_proxy"] = {
            rename_map.get(name, name): value for name, value in per_proxy.items()
        }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute AUC only when both classes are present."""
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _compute_proxy_baseline_diagnostics(
    bundle: ProxyBundle,
    latent_values: np.ndarray,
    proxy_data: pd.DataFrame,
    calibration_data: Optional[pd.DataFrame],
    config: ObservationConfig,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Compute naive vs calibrated proxy recoverability diagnostics.

    The two baselines use different training data but share a common holdout
    from the full dataset so their scores are directly comparable:

    - **Naive** (lower bound): best single-proxy linear model trained on
      the small public calibration set.  This is what a lazy agent would get.
    - **Calibrated** (upper bound): multi-proxy GBM trained on the full
      latent+proxy dataset.  This is the best-case recoverability that no
      agent can exceed (they never see the full labels).
    """
    proxy_names = bundle.proxy_names()
    missing = [c for c in proxy_names if c not in proxy_data.columns]
    if missing or len(latent_values) < 8:
        return {"status": "skipped", "reason": "insufficient_data"}

    X_full = proxy_data[proxy_names].to_numpy(dtype=float)
    y_full = np.asarray(latent_values, dtype=float)
    row_ids = np.arange(len(y_full))

    # Holdout from full data — shared evaluation set for both baselines.
    test_size = max(50, int(round(len(y_full) * 0.2)))
    test_size = min(test_size, len(y_full) - 2)
    if test_size <= 1:
        return {"status": "skipped", "reason": "too_few_rows_for_holdout"}

    stratify = None
    is_binary = str(bundle.latent_type).strip().lower() == "binary"
    if is_binary and np.unique(y_full).size >= 2:
        stratify = y_full

    try:
        full_train_idx, test_idx = train_test_split(
            row_ids,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        full_train_idx, test_idx = train_test_split(
            row_ids,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    X_test = X_full[test_idx]
    y_test = y_full[test_idx]

    # --- Calibrated upper bound: GBM on full training data ---
    X_full_train = X_full[full_train_idx]
    y_full_train = y_full[full_train_idx]

    # --- Naive lower bound: linear on calibration data only ---
    # Use calibration rows that are NOT in the test set.
    has_cal = (
        calibration_data is not None
        and bundle.latent_name in calibration_data.columns
        and len(calibration_data) >= 4
    )
    if has_cal:
        X_cal = calibration_data[proxy_names].to_numpy(dtype=float)
        y_cal = calibration_data[bundle.latent_name].to_numpy(dtype=float)
    else:
        # Fallback: use full train data for naive too (degrades to old behavior).
        X_cal = X_full_train
        y_cal = y_full_train

    if is_binary:
        if np.unique(y_full_train).size < 2 or np.unique(y_test).size < 2:
            return {
                "status": "skipped",
                "reason": "binary_holdout_missing_class",
                "holdout_row_positions": test_idx.tolist(),
            }

        # Naive: best single proxy, logistic, trained on calibration
        naive_models = []
        for i, proxy_name in enumerate(proxy_names):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=1.0, max_iter=1000)),
                ]
            )
            model.fit(X_cal[:, [i]], y_cal)
            prob = model.predict_proba(X_test[:, [i]])[:, 1]
            ll = float(log_loss(y_test, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]))
            auc = _safe_auc(y_test, prob)
            naive_models.append({"proxy_name": proxy_name, "log_loss": ll, "auc": auc})
        naive_best = min(naive_models, key=lambda row: row["log_loss"])

        # Calibrated: GBM on all proxies, trained on full data
        multi = GradientBoostingClassifier(
            n_estimators=config.baseline_n_estimators,
            max_depth=config.baseline_max_depth,
            learning_rate=config.baseline_learning_rate,
            subsample=0.8,
            random_state=seed,
        )
        multi.fit(X_full_train, y_full_train)
        prob_multi = multi.predict_proba(X_test)[:, 1]
        calibrated = {
            "log_loss": float(
                log_loss(y_test, np.clip(prob_multi, 1e-6, 1 - 1e-6), labels=[0, 1])
            ),
            "auc": _safe_auc(y_test, prob_multi),
        }
        return {
            "status": "ok",
            "metric_type": "binary",
            "naive_data_source": "calibration" if has_cal else "full",
            "naive_train_rows": len(y_cal),
            "calibrated_train_rows": len(y_full_train),
            "holdout_row_positions": test_idx.tolist(),
            "naive": naive_best,
            "calibrated": calibrated,
            "improvement_gap": {
                "log_loss_reduction": naive_best["log_loss"] - calibrated["log_loss"],
                "auc_gain": (
                    None
                    if naive_best["auc"] is None or calibrated["auc"] is None
                    else float(calibrated["auc"] - naive_best["auc"])
                ),
            },
        }

    # --- Continuous ---
    # Naive: best single proxy, OLS, trained on calibration
    naive_models = []
    for i, proxy_name in enumerate(proxy_names):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
        model.fit(X_cal[:, [i]], y_cal)
        pred = model.predict(X_test[:, [i]])
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        naive_models.append(
            {
                "proxy_name": proxy_name,
                "rmse": rmse,
                "r2": float(r2_score(y_test, pred)),
            }
        )
    naive_best = min(naive_models, key=lambda row: row["rmse"])

    # Calibrated: GBM on all proxies, trained on full data
    multi = GradientBoostingRegressor(
        n_estimators=config.baseline_n_estimators,
        max_depth=config.baseline_max_depth,
        learning_rate=config.baseline_learning_rate,
        subsample=0.8,
        random_state=seed,
    )
    multi.fit(X_full_train, y_full_train)
    pred_multi = multi.predict(X_test)
    calibrated = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_multi))),
        "r2": float(r2_score(y_test, pred_multi)),
    }
    return {
        "status": "ok",
        "metric_type": "continuous",
        "naive_data_source": "calibration" if has_cal else "full",
        "naive_train_rows": len(y_cal),
        "calibrated_train_rows": len(y_full_train),
        "holdout_row_positions": test_idx.tolist(),
        "naive": naive_best,
        "calibrated": calibrated,
        "improvement_gap": {
            "rmse_reduction": naive_best["rmse"] - calibrated["rmse"],
            "r2_gain": float(calibrated["r2"] - naive_best["r2"]),
        },
    }


class ObservationModel:
    """Object-oriented proxy observation layer over conceptual observed variables.

    Important distinction:
    - `sg.latent_nodes` are true latent/unobserved variables (e.g., hidden confounders).
    - proxified nodes in this model are observed conceptual variables where full-fidelity
      measurement is expensive, so public data exposes proxy bundles instead.
    """

    def __init__(
        self,
        *,
        sg: SampledGraph,
        node_types: Optional[Dict[str, str]] = None,
        config: Optional[ObservationConfig] = None,
        seed: Optional[int] = None,
        node_name_map: Optional[Dict[str, str]] = None,
    ):
        self.sg = sg
        self.node_types = {
            str(node): str(dtype).strip().lower()
            for node, dtype in dict(node_types or {}).items()
        }
        self.node_name_map = {
            str(node): str(mapped) for node, mapped in dict(node_name_map or {}).items()
        }
        self.config = config or ObservationConfig()
        self.seed = normalize_random_seed(seed or 0)
        self.rng = np.random.RandomState(self.seed)
        self.true_latent_nodes = {
            self._to_conceptual_name(str(node)) for node in (sg.latent_nodes or [])
        }

    def _to_conceptual_name(self, node: str) -> str:
        """Map graph node IDs to conceptual/public column names when available."""
        key = str(node)
        return self.node_name_map.get(key, key)

    def _eligible_conceptual_nodes(self, columns: List[str]) -> List[str]:
        """Return observed conceptual nodes eligible for proxification."""
        colset = {str(col) for col in columns}
        eligible = [
            self._to_conceptual_name(str(node))
            for node in _eligible_proxified_nodes(self.sg)
        ]
        eligible = [node for node in eligible if node in colset]
        eligible = [node for node in eligible if node not in self.true_latent_nodes]
        return sorted(set(eligible))

    def _select_proxified_nodes(self, eligible_nodes: List[str]) -> List[str]:
        """Sample how many/which conceptual nodes to proxify."""
        prox_lo, prox_hi = self.config.proxified_nodes_range
        prox_hi = min(int(prox_hi), len(eligible_nodes))
        prox_lo = min(int(prox_lo), prox_hi)
        if prox_hi <= 0:
            return []
        if prox_lo == prox_hi:
            chosen = self.rng.choice(eligible_nodes, size=prox_hi, replace=False)
            return sorted([str(node) for node in chosen.tolist()])
        n_proxified = self.rng.randint(prox_lo, prox_hi + 1)
        chosen = self.rng.choice(eligible_nodes, size=n_proxified, replace=False)
        return sorted([str(node) for node in chosen.tolist()])

    def _resolve_conceptual_var_type(self, node: str, values: pd.Series) -> str:
        """Resolve conceptual variable type with fallback inference from data."""
        declared = self.node_types.get(str(node))
        if declared in {"binary", "continuous"}:
            return declared

        numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        unique_vals = set(np.unique(numeric).tolist()) if numeric.size else set()
        if unique_vals and unique_vals <= {0.0, 1.0}:
            return "binary"
        return "continuous"

    def build(
        self,
        conceptual_data: pd.DataFrame,
        *,
        train_ratio: float = 0.8,
    ) -> ObservationData:
        """Build proxy-observed public data and calibration data."""
        conceptual_copy = conceptual_data.copy()

        if not self.config.enabled:
            return ObservationData(
                latent_data=conceptual_copy,
                public_data=conceptual_copy.copy(),
                calibration_data=None,
                proxified_nodes=[],
                proxy_bundles={},
                calibration_indices=[],
                metadata={
                    "enabled": False,
                    "measurement_scope": "observed_conceptual_nodes_only",
                    "true_latent_nodes": sorted(self.true_latent_nodes),
                    "proxified_nodes": [],
                    "proxy_groups": {},
                    "calibration_indices": [],
                },
            )

        eligible_nodes = self._eligible_conceptual_nodes(list(conceptual_copy.columns))
        if not eligible_nodes:
            return ObservationData(
                latent_data=conceptual_copy,
                public_data=conceptual_copy.copy(),
                calibration_data=None,
                proxified_nodes=[],
                proxy_bundles={},
                calibration_indices=[],
                metadata={
                    "enabled": True,
                    "measurement_scope": "observed_conceptual_nodes_only",
                    "true_latent_nodes": sorted(self.true_latent_nodes),
                    "proxified_nodes": [],
                    "proxy_groups": {},
                    "calibration_indices": [],
                    "warning": "No eligible conceptual nodes available for proxification.",
                },
            )

        proxified_nodes = self._select_proxified_nodes(eligible_nodes)
        invalid = [node for node in proxified_nodes if node in self.true_latent_nodes]
        if invalid:
            raise ValueError(
                f"Proxy observation nodes must be observed conceptual variables, got true latent nodes: {invalid}"
            )

        public_data = conceptual_copy.copy()
        proxy_bundles: Dict[str, ProxyBundle] = {}
        for node in proxified_nodes:
            node_values = pd.to_numeric(conceptual_copy[node], errors="coerce")
            conceptual_type = self._resolve_conceptual_var_type(node, node_values)
            bundle = _sample_proxy_bundle(
                node,
                node_values.to_numpy(dtype=float),
                conceptual_type,
                self.rng,
                is_outcome=(node == self._to_conceptual_name(self.sg.outcome)),
                config=self.config,
            )
            protected_names = set(public_data.columns) - {node}
            _ensure_unique_proxy_column_names(bundle, protected_names)
            public_data = public_data.drop(columns=[node]).join(bundle.proxy_frame)
            proxy_bundles[node] = bundle

        n_train = int(len(conceptual_copy) * float(train_ratio))
        conceptual_train = conceptual_copy.iloc[:n_train].reset_index(drop=True)
        calibration_indices = _select_calibration_indices(
            conceptual_train,
            proxified_nodes,
            self.config,
            self.rng,
        )

        calibration_data = None
        if calibration_indices:
            public_subset = public_data.iloc[calibration_indices].reset_index(drop=True)
            conceptual_subset = conceptual_copy.iloc[calibration_indices][
                proxified_nodes
            ].reset_index(drop=True)
            calibration_data = pd.concat([public_subset, conceptual_subset], axis=1)

        # Baseline diagnostics:
        #   naive  = single-proxy linear trained on calibration (what a lazy agent gets)
        #   calibrated = multi-proxy GBM trained on full data (upper bound)
        for offset, node in enumerate(proxified_nodes, start=1):
            bundle = proxy_bundles[node]
            bundle.baseline_diagnostics = _compute_proxy_baseline_diagnostics(
                bundle,
                latent_values=conceptual_copy[node].to_numpy(dtype=float),
                proxy_data=public_data,
                calibration_data=calibration_data,
                config=self.config,
                seed=offset_random_seed(self.seed, 97 * offset),
            )

        metadata = {
            "enabled": True,
            "measurement_scope": "observed_conceptual_nodes_only",
            "true_latent_nodes": sorted(self.true_latent_nodes),
            "proxified_nodes": proxified_nodes,
            "conceptual_nodes_with_proxy_observation": proxified_nodes,
            "proxy_groups": {
                conceptual_node: bundle.proxy_names()
                for conceptual_node, bundle in proxy_bundles.items()
            },
            "calibration_indices": calibration_indices,
            "calibration_source": "train_split",
            "train_ratio": float(train_ratio),
            "bundles": {
                conceptual_node: bundle.to_metadata()
                for conceptual_node, bundle in proxy_bundles.items()
            },
        }
        return ObservationData(
            latent_data=conceptual_copy,
            public_data=public_data,
            calibration_data=calibration_data,
            proxified_nodes=proxified_nodes,
            proxy_bundles=proxy_bundles,
            calibration_indices=calibration_indices,
            metadata=metadata,
        )
