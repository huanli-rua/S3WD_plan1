from __future__ import annotations

"""Utility functions for training simple binary classification baselines."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_io import augment_airline_features
from .evalx import classification_metrics
from .gwb import GWBProbEstimator

try:  # pragma: no cover - optional dependency used in notebooks
    import xgboost as xgb
except ImportError:  # pragma: no cover - fallback for environments without XGBoost
    xgb = None

_DEFAULT_WARMUP_YEARS = list(range(1987, 2000))
_DEFAULT_CATEGORICAL = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month", "dep_block"]
_DEFAULT_NUMERIC = [
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "dep_hour",
    "arr_hour",
    "block_time_min",
    "Year",
]


@dataclass
class BaselineFeatureSet:
    """Container for baseline-ready feature matrices."""

    matrix: pd.DataFrame
    y: np.ndarray
    encoders: Dict[str, Mapping[str, int]]
    numeric_columns: List[str]
    categorical_columns: List[str]


def enrich_airline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared airline feature augmentation in a copy-safe manner."""

    if df is None:
        raise ValueError("输入 DataFrame 不能为空。")
    enriched = augment_airline_features(df)
    if "Year" in enriched.columns:
        enriched["Year"] = pd.to_numeric(enriched["Year"], errors="coerce").astype(int)
    if "Month" in enriched.columns:
        enriched["Month"] = pd.to_numeric(enriched["Month"], errors="coerce").astype(int)
    return enriched


def _ensure_year_column(df: pd.DataFrame, year_col: str) -> pd.Series:
    if year_col not in df.columns:
        raise KeyError(f"数据集中缺少年份列 '{year_col}'，无法按年份拆分。")
    years = pd.to_numeric(df[year_col], errors="coerce")
    if years.isna().any():
        raise ValueError("年份列包含无法转换为整数的值。")
    return years.astype(int)


def split_by_year(
    df: pd.DataFrame,
    *,
    year_col: str = "Year",
    warmup_years: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, MutableMapping[int, pd.DataFrame]]:
    """Split the dataframe into warmup (train) and per-year stream sets."""

    years = _ensure_year_column(df, year_col)
    warmup_years = list(warmup_years or _DEFAULT_WARMUP_YEARS)
    warm_mask = years.isin(warmup_years)
    train_df = df.loc[warm_mask].copy()
    if train_df.empty:
        raise RuntimeError("暖启动年份拆分后训练集为空，请检查年份配置。")
    stream_years = sorted(int(y) for y in years.unique() if int(y) not in warmup_years)
    test_splits: MutableMapping[int, pd.DataFrame] = {}
    for year in stream_years:
        subset = df.loc[years == year].copy()
        if not subset.empty:
            test_splits[year] = subset
    if not test_splits:
        raise RuntimeError("未找到任何 stream 年份数据，请确认 warmup/stream 划分。")
    return train_df, test_splits


def _resolve_columns(df: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    present = [col for col in candidates if col in df.columns]
    return present


def _encode_categorical(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
    encoders: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Mapping[str, int]]]:
    encoded = pd.DataFrame(index=df.index)
    fitted: Dict[str, Mapping[str, int]] = {}
    if encoders:
        for key, mapping in encoders.items():
            fitted[key] = dict(mapping)
    for col in categorical_columns:
        series = df[col].astype(str).fillna("__MISSING__")
        mapping = fitted.get(col)
        if mapping is None:
            categories = pd.Index(pd.unique(series))
            mapping = {val: idx for idx, val in enumerate(categories)}
            fitted[col] = mapping
        encoded[col] = series.map(mapping).fillna(-1).astype(int)
    encoded.reset_index(drop=True, inplace=True)
    return encoded, fitted


def _binarize_labels(values: pd.Series, positive_label=1) -> np.ndarray:
    series = pd.Series(values)
    if series.dtype == object:
        series_norm = series.astype(str)
        y = (series_norm == str(positive_label)).astype(int)
    else:
        y = (series == positive_label).astype(int)
    return y.to_numpy(dtype=int, copy=False)


def prepare_baseline_features(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_label=1,
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
    encoders: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> BaselineFeatureSet:
    """Construct baseline feature matrix, binary labels, and categorical encoders."""

    if label_col not in df.columns:
        raise KeyError(f"数据集中缺少标签列 '{label_col}'。")
    categorical_cols = _resolve_columns(df, categorical_features or _DEFAULT_CATEGORICAL)
    numeric_cols = _resolve_columns(df, numeric_features or _DEFAULT_NUMERIC)
    if not numeric_cols:
        raise ValueError("未找到可用于建模的数值特征列。")

    numeric_part = (
        df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    encoded_cats, fitted_encoders = _encode_categorical(df, categorical_cols, encoders=encoders)
    if encoded_cats.empty:
        matrix = numeric_part.reset_index(drop=True)
    else:
        matrix = pd.concat([numeric_part.reset_index(drop=True), encoded_cats], axis=1)
    y = _binarize_labels(df[label_col], positive_label)
    return BaselineFeatureSet(
        matrix=matrix,
        y=y,
        encoders=fitted_encoders,
        numeric_columns=list(numeric_cols),
        categorical_columns=list(categorical_cols),
    )


def _metrics_with_counts(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Mapping[str, float]:
    metrics = classification_metrics(y_true, y_pred, y_score)
    return {
        "Precision": metrics.get("Prec", np.nan),
        "Recall": metrics.get("Rec", np.nan),
        "F1": metrics.get("F1", np.nan),
        "BAC": metrics.get("BAC", np.nan),
        "MCC": metrics.get("MCC", np.nan),
        "Kappa": metrics.get("Kappa", np.nan),
        "AUC": metrics.get("AUC", np.nan),
    }


def evaluate_gwb_baseline_by_year(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_label=1,
    year_col: str = "Year",
    warmup_years: Optional[Sequence[int]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    gwb_params: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Train a GWB binary classifier on warmup years and evaluate yearly streams."""

    enriched = enrich_airline_dataframe(df)
    warmup_df, test_splits = split_by_year(
        enriched,
        year_col=year_col,
        warmup_years=warmup_years,
    )
    warm_features = prepare_baseline_features(
        warmup_df,
        label_col=label_col,
        positive_label=positive_label,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )
    gwb_kwargs = dict(gwb_params or {})
    if "categorical_features" not in gwb_kwargs:
        gwb_kwargs["categorical_features"] = warm_features.categorical_columns
    estimator = GWBProbEstimator(**gwb_kwargs)
    cat_train = (
        warmup_df[warm_features.categorical_columns].reset_index(drop=True)
        if warm_features.categorical_columns
        else None
    )
    estimator.fit(
        warm_features.matrix[warm_features.numeric_columns],
        warm_features.y,
        categorical_values=cat_train,
    )

    records: List[Mapping[str, float]] = []
    for year, subset in sorted(test_splits.items()):
        feat_year = prepare_baseline_features(
            subset,
            label_col=label_col,
            positive_label=positive_label,
            categorical_features=warm_features.categorical_columns,
            numeric_features=warm_features.numeric_columns,
            encoders=warm_features.encoders,
        )
        cat_values = (
            subset[feat_year.categorical_columns].reset_index(drop=True)
            if feat_year.categorical_columns
            else None
        )
        probs = estimator.predict_proba(
            feat_year.matrix[feat_year.numeric_columns],
            categorical_values=cat_values,
        )
        preds = (probs >= threshold).astype(int)
        metrics = _metrics_with_counts(feat_year.y, preds, probs)
        record = {
            "Year": year,
            "n_samples": int(len(feat_year.y)),
            "POS_rate": float(np.mean(feat_year.y)) if len(feat_year.y) else np.nan,
        }
        record.update(metrics)
        records.append(record)
    result = pd.DataFrame(records).set_index("Year").sort_index()
    return result


def evaluate_xgb_baseline_by_year(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_label=1,
    year_col: str = "Year",
    warmup_years: Optional[Sequence[int]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    numeric_features: Optional[Sequence[str]] = None,
    threshold: float = 0.5,
    random_state: int = 42,
    xgb_params: Optional[Mapping[str, object]] = None,
) -> pd.DataFrame:
    """Train an XGBoost classifier on warmup data and evaluate each stream year."""

    if xgb is None:
        raise ImportError("xgboost 未安装，无法训练 XGBoost baseline。")

    enriched = enrich_airline_dataframe(df)
    warmup_df, test_splits = split_by_year(
        enriched,
        year_col=year_col,
        warmup_years=warmup_years,
    )
    warm_features = prepare_baseline_features(
        warmup_df,
        label_col=label_col,
        positive_label=positive_label,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "use_label_encoder": False,
    }
    if xgb_params:
        base_params.update(xgb_params)
    model = xgb.XGBClassifier(**base_params)
    model.fit(warm_features.matrix, warm_features.y)

    records: List[Mapping[str, float]] = []
    for year, subset in sorted(test_splits.items()):
        feat_year = prepare_baseline_features(
            subset,
            label_col=label_col,
            positive_label=positive_label,
            categorical_features=warm_features.categorical_columns,
            numeric_features=warm_features.numeric_columns,
            encoders=warm_features.encoders,
        )
        probs = model.predict_proba(feat_year.matrix)[:, 1]
        preds = (probs >= threshold).astype(int)
        metrics = _metrics_with_counts(feat_year.y, preds, probs)
        record = {
            "Year": year,
            "n_samples": int(len(feat_year.y)),
            "POS_rate": float(np.mean(feat_year.y)) if len(feat_year.y) else np.nan,
        }
        record.update(metrics)
        records.append(record)
    result = pd.DataFrame(records).set_index("Year").sort_index()
    return result


__all__ = [
    "BaselineFeatureSet",
    "enrich_airline_dataframe",
    "split_by_year",
    "prepare_baseline_features",
    "evaluate_gwb_baseline_by_year",
    "evaluate_xgb_baseline_by_year",
]
