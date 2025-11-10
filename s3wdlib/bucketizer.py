# -*- coding: utf-8 -*-
"""分桶与回退逻辑模块。"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

__all__ = ["configure", "assign_buckets", "current_config"]

# 模块级配置，默认与 YAML 约定一致，外部可通过 configure() 注入。
_CONFIG: dict = {
    "keys": ["UniqueCarrier", "Origin", "Dest", "dep_hour"],
    "min_bucket": 500,
    "backoff": [["UniqueCarrier", "Origin", "Dest"], ["Origin", "Dest"], ["Origin"]],
}


def _safe_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """返回 DataFrame 中存在的列名，缺失列会被剔除。"""

    valid: List[str] = []
    for col in cols:
        if col in df.columns:
            valid.append(col)
    return valid


def configure(bucket_cfg: dict | None) -> None:
    """更新模块级配置。未提供的键保持默认值。"""

    global _CONFIG
    if not bucket_cfg:
        return
    cfg = dict(_CONFIG)
    for key in ["keys", "min_bucket", "backoff"]:
        if key in bucket_cfg and bucket_cfg[key] is not None:
            cfg[key] = bucket_cfg[key]
    cfg["keys"] = list(cfg.get("keys", []))
    cfg["backoff"] = [list(x) for x in cfg.get("backoff", [])]
    cfg["min_bucket"] = int(cfg.get("min_bucket", 0))
    _CONFIG = cfg


def current_config() -> dict:
    """返回当前桶配置的副本，供调试打印使用。"""

    return dict(_CONFIG)


def _combine_values(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """将指定列拼接为字符串ID。缺失列时返回固定标识。"""

    if not cols:
        return pd.Series(["__MISSING__"] * len(df), index=df.index, dtype="object")
    part = df[cols].astype(str)
    combined = part.apply(lambda row: "|".join(row.values.tolist()), axis=1)
    return combined.astype("object")


def assign_buckets(df: pd.DataFrame) -> np.ndarray:
    """
    根据 BUCKET.keys 生成分桶ID（字符串）；若样本不足 min_bucket，按 backoff 阶梯回退。
    返回与 df 同长度的一维数组。
    """

    if df.empty:
        return np.asarray([], dtype="object")

    combos: List[List[str]] = [list(_CONFIG.get("keys", []))]
    for back in _CONFIG.get("backoff", []):
        combos.append(list(back))

    bucket_series: List[pd.Series] = []
    counts_cache: List[dict[str, int]] = []
    for cols in combos:
        cols_valid = _safe_columns(df, cols)
        series = _combine_values(df, cols_valid)
        bucket_series.append(series)
        counts = series.value_counts().to_dict()
        counts_cache.append(counts)

    min_bucket = int(max(0, _CONFIG.get("min_bucket", 0)))
    final = pd.Series([None] * len(df), index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index, dtype="bool")

    for series, counts in zip(bucket_series, counts_cache):
        if assigned.all():
            break
        mask = ~assigned
        series_sub = series[mask]
        sufficient = series_sub.map(lambda v: counts.get(v, 0) >= min_bucket)
        to_assign = mask.copy()
        to_assign.loc[mask] = sufficient
        final[to_assign] = series[to_assign]
        assigned |= to_assign

    # 最后一层兜底：仍未分配的直接使用最后一级桶ID
    if not assigned.all():
        fallback_series = bucket_series[-1]
        final[~assigned] = fallback_series[~assigned]

    return final.to_numpy(dtype="object")
