# -*- coding: utf-8 -*-
"""分桶与回退逻辑模块。"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

__all__ = ["configure", "assign_buckets", "current_config", "Bucketizer"]

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


class Bucketizer:
    """V3 bucketizer with fine (Origin, dep_slot) and coarse airport buckets."""

    def __init__(
        self,
        *,
        fine_keys: list[str],
        coarse_keys_level1: list[str],
        min_fine_size: int,
        min_level1_size: int,
        airport_col: str,
        dep_slot_col: str,
        feature_cols: list[str],
    ) -> None:
        self.fine_keys = list(fine_keys or [])
        self.coarse_keys_level1 = list(coarse_keys_level1 or [])
        self.min_fine_size = int(max(0, min_fine_size))
        self.min_level1_size = int(max(0, min_level1_size))
        self.airport_col = airport_col
        self.dep_slot_col = dep_slot_col
        self.feature_cols = list(feature_cols or [])

        self.fine_mapping: dict[tuple, str] = {}
        self.coarse_mapping: dict[str, str] = {}
        self.bucket_sizes: dict[str, int] = {}

    def _ensure_columns(self, df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise KeyError(f"数据缺少必要列: {missing}")

    @staticmethod
    def _as_tuple(key_values) -> tuple:
        if isinstance(key_values, tuple):
            return key_values
        if isinstance(key_values, list):
            return tuple(key_values)
        return (key_values,)

    def _format_fine_bucket_id(self, fine_key: tuple) -> str:
        bucket_id = None
        try:
            airport_idx = self.fine_keys.index(self.airport_col)
            slot_idx = self.fine_keys.index(self.dep_slot_col)
            airport = fine_key[airport_idx]
            dep_slot = fine_key[slot_idx]
            bucket_id = f"{airport}__slot{dep_slot}"
        except ValueError:
            pass
        if bucket_id is None:
            joined = "__".join(str(v) for v in fine_key)
            bucket_id = f"FINE__{joined}"
        return bucket_id

    def _format_coarse_bucket_id(self, airport: str) -> str:
        return f"{airport}__COARSE"

    def build_from_train0(self, df_train0: pd.DataFrame) -> None:
        required_cols = set(self.fine_keys + self.coarse_keys_level1 + [self.airport_col, self.dep_slot_col])
        required_cols.update(self.feature_cols)
        self._ensure_columns(df_train0, list(required_cols))

        if not self.fine_keys:
            raise ValueError("fine_keys 不能为空。")
        if not self.coarse_keys_level1:
            raise ValueError("coarse_keys_level1 不能为空。")

        self.bucket_sizes = {}
        grouped = df_train0.groupby(self.fine_keys).size()
        self.fine_mapping.clear()
        for key_values, count in grouped.items():
            key_tuple = self._as_tuple(key_values)
            if count >= self.min_fine_size:
                bucket_id = self._format_fine_bucket_id(key_tuple)
                self.fine_mapping[key_tuple] = bucket_id
                self.bucket_sizes[bucket_id] = int(count)

        airport_counts = df_train0.groupby(self.airport_col).size()
        self.coarse_mapping.clear()
        for airport, count in airport_counts.items():
            if count >= self.min_level1_size:
                bucket_id = self._format_coarse_bucket_id(str(airport))
                self.coarse_mapping[str(airport)] = bucket_id
                self.bucket_sizes[bucket_id] = int(count)

    def assign_bucket(self, row: pd.Series) -> str:
        key_values = tuple(row[col] for col in self.fine_keys)
        bucket_id = self.fine_mapping.get(key_values)
        if bucket_id is not None:
            return bucket_id

        airport = str(row[self.airport_col])
        bucket_id = self.coarse_mapping.get(airport)
        if bucket_id is None:
            bucket_id = self._format_coarse_bucket_id(airport)
            self.coarse_mapping[airport] = bucket_id
            self.bucket_sizes.setdefault(bucket_id, 0)
        return bucket_id

    def _mask_for_bucket(self, df: pd.DataFrame, bucket_id: str) -> pd.Series:
        if bucket_id.endswith("__COARSE"):
            airport = bucket_id[: -len("__COARSE")]
            return df[self.airport_col].astype(str) == airport
        if "__slot" in bucket_id:
            airport, slot_part = bucket_id.split("__slot", 1)
            try:
                dep_slot = int(slot_part)
            except ValueError:
                raise ValueError(f"无法解析桶 ID: {bucket_id}") from None
            return (df[self.airport_col].astype(str) == airport) & (df[self.dep_slot_col] == dep_slot)
        raise ValueError(f"不支持的桶 ID 格式: {bucket_id}")

    def get_bucket_data(self, df: pd.DataFrame, bucket_id: str, label_col: str):
        required_cols = [label_col] + self.feature_cols
        self._ensure_columns(df, required_cols)
        mask = self._mask_for_bucket(df, bucket_id)
        bucket_df = df.loc[mask]
        X = bucket_df[self.feature_cols].to_numpy(dtype=float, copy=False)
        y = bucket_df[label_col].to_numpy(copy=False)
        return X, y

    def extract_features(self, row: pd.Series) -> np.ndarray:
        values = []
        for col in self.feature_cols:
            if col not in row.index:
                raise KeyError(f"样本缺少特征列 {col}")
            values.append(row[col])
        return np.asarray(values, dtype=float)
