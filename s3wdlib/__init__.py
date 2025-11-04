"""
s3wdlib: Minimal scaffolding for S3WD experiments.
"""
from .data_io import load_table_auto, minmax_scale_fit_transform
from .features import rank_features_mi, make_levels
from .dyn_threshold import (
    adapt_thresholds_rule_based,
    adapt_thresholds_windowed_pso,
    DynamicAdaptConfig,
    IncrementalUpdateConfig,
    run_dynamic_thresholds,
)
from .drift import DriftDetector, DriftEvent, DriftConfig

__all__ = [
    "load_table_auto",
    "minmax_scale_fit_transform",
    "rank_features_mi",
    "make_levels",
    "adapt_thresholds_rule_based",
    "adapt_thresholds_windowed_pso",
    "run_dynamic_thresholds",
    "DynamicAdaptConfig",
    "IncrementalUpdateConfig",
    "DriftDetector",
    "DriftEvent",
    "DriftConfig",
]
