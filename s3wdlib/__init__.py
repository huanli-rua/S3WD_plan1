"""
s3wdlib: Minimal scaffolding for S3WD experiments.
"""
from .data_io import (
    load_table_auto,
    minmax_scale_fit_transform,
    augment_airline_features,
    assign_year_from_month_sequence,
)
from .baselines import (
    BaselineFeatureSet,
    enrich_airline_dataframe,
    split_by_year,
    prepare_baseline_features,
    evaluate_gwb_baseline_by_year,
    evaluate_xgb_baseline_by_year,
)
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
from .bucketizer import assign_buckets
from .ref_tuple import build_ref_tuples, combine_history
from .similarity import corr_to_set
from .batch_measure import to_trisect_probs, expected_cost, expected_fbeta, compute_region_masks
from .threshold_selector import select_alpha_beta
from .smoothing import ema_clip
from .drift_controller import detect_drift, apply_actions
from .v02_flow import run_streaming_flow

__all__ = [
    "load_table_auto",
    "minmax_scale_fit_transform",
    "augment_airline_features",
    "assign_year_from_month_sequence",
    "BaselineFeatureSet",
    "enrich_airline_dataframe",
    "split_by_year",
    "prepare_baseline_features",
    "evaluate_gwb_baseline_by_year",
    "evaluate_xgb_baseline_by_year",
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
    "assign_buckets",
    "build_ref_tuples",
    "combine_history",
    "corr_to_set",
    "to_trisect_probs",
    "expected_cost",
    "expected_fbeta",
    "compute_region_masks",
    "select_alpha_beta",
    "ema_clip",
    "detect_drift",
    "apply_actions",
    "run_streaming_flow",
]
