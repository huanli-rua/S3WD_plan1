"""
s3wdlib: Minimal scaffolding for S3WD experiments.
"""
from .data_io import load_table_auto, minmax_scale_fit_transform
from .features import rank_features_mi, make_levels
from .dyn_threshold import adapt_thresholds_rule_based, adapt_thresholds_windowed_pso
from .drift import DriftDetector, DriftEvent
from .incremental import PosteriorUpdater, latest_estimator_for_flow
from .streaming import (
    DynamicLoopConfig,
    DriftDetectorConfig,
    PosteriorUpdaterConfig,
    build_dynamic_components,
)
from .evalx import classification_metrics, batch_metrics, layer_stats
from .viz import probability_histogram, threshold_trajectory, drift_timeline

__all__ = [
    "load_table_auto",
    "minmax_scale_fit_transform",
    "rank_features_mi",
    "make_levels",
    "adapt_thresholds_rule_based",
    "adapt_thresholds_windowed_pso",
    "DriftDetector",
    "DriftEvent",
    "PosteriorUpdater",
    "latest_estimator_for_flow",
    "DynamicLoopConfig",
    "DriftDetectorConfig",
    "PosteriorUpdaterConfig",
    "build_dynamic_components",
    "classification_metrics",
    "batch_metrics",
    "layer_stats",
    "probability_histogram",
    "threshold_trajectory",
    "drift_timeline",
]
