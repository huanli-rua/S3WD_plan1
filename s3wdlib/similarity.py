# -*- coding: utf-8 -*-
"""混合相似度计算模块。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .ref_tuple import RefItem

__all__ = ["rbf_num_sim", "cat_match", "mixed_similarity", "corr_to_set", "configure", "current_config"]

_CONFIG: dict = {
    "sigma": 0.5,
    "cat_weights": {"carrier": 0.4, "origin": 0.2, "dest": 0.2, "dow": 0.1, "month": 0.1},
    "combine": "product",
    "mix_alpha": 0.7,
}

_CAT_FIELD_MAP = {
    "carrier": "UniqueCarrier",
    "origin": "Origin",
    "dest": "Dest",
    "dow": "DayOfWeek",
    "month": "Month",
}


def configure(sim_cfg: dict | None) -> None:
    """更新模块默认配置。"""

    global _CONFIG
    if not sim_cfg:
        return
    cfg = dict(_CONFIG)
    for key in ["sigma", "combine", "mix_alpha"]:
        if key in sim_cfg and sim_cfg[key] is not None:
            cfg[key] = sim_cfg[key]
    if "cat_weights" in sim_cfg and sim_cfg["cat_weights"] is not None:
        cfg["cat_weights"] = dict(sim_cfg["cat_weights"])
    _CONFIG = cfg


def current_config() -> dict:
    """返回当前相似度配置。"""

    return dict(_CONFIG)


def rbf_num_sim(x_num: np.ndarray, tau_num: np.ndarray, sigma: float) -> float:
    """计算数值特征的 RBF（径向基函数）相似度。"""

    if x_num.size == 0 or tau_num.size == 0:
        return 1.0
    sigma = max(float(sigma), 1e-6)
    diff = x_num - tau_num
    dist2 = float(np.dot(diff, diff))
    return float(np.exp(-dist2 / (2 * sigma ** 2)))


def cat_match(x_cat: Dict[str, object], tau_cat: Dict[str, object], weights: Dict[str, float]) -> float:
    """计算类别字段匹配得分，按权重归一。"""

    if not weights:
        return 1.0
    total = float(sum(max(0.0, w) for w in weights.values()))
    if total <= 0:
        return 1.0
    score = 0.0
    for key, weight in weights.items():
        col = _CAT_FIELD_MAP.get(key, key)
        if str(x_cat.get(col, "__NA__")) == str(tau_cat.get(col, "__NA__")):
            score += max(0.0, float(weight))
    return float(score / total)


def mixed_similarity(
    x_row,
    tau_item: RefItem,
    sigma: float,
    cat_weights: Dict[str, float],
    combine: str = "product",
    mix_alpha: float = 0.7,
) -> float:
    """综合数值+类别相似度，支持乘积或加权和策略。"""

    if isinstance(x_row, pd.Series):
        x_num = x_row[[c for c in ["dep_hour", "arr_hour", "block_time_min", "Distance"] if c in x_row.index]].to_numpy(dtype=float)
        x_cat = {col: x_row.get(col) for col in ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month"]}
    elif isinstance(x_row, dict):
        x_num = np.asarray([x_row.get(c, 0.0) for c in ["dep_hour", "arr_hour", "block_time_min", "Distance"]], dtype=float)
        x_cat = {col: x_row.get(col) for col in ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month"]}
    else:
        x_arr = np.asarray(x_row, dtype=float)
        x_num = x_arr
        x_cat = {}

    num_sim = rbf_num_sim(x_num, tau_item.num_vec, sigma)
    cat_sim = cat_match(x_cat, tau_item.cat_fields, cat_weights)

    strategy = str(combine).lower()
    if strategy == "weighted_sum":
        mix_alpha = float(np.clip(mix_alpha, 0.0, 1.0))
        return float(mix_alpha * num_sim + (1 - mix_alpha) * cat_sim)
    return float(num_sim * cat_sim)


def _standardize_block(X_block: pd.DataFrame) -> pd.DataFrame:
    """按窗口自身统计量对数值列标准化，避免尺度影响。"""

    num_cols = [c for c in ["dep_hour", "arr_hour", "block_time_min", "Distance"] if c in X_block.columns]
    if not num_cols:
        return X_block
    X_std = X_block.copy()
    for col in num_cols:
        values = pd.to_numeric(X_std[col], errors="coerce").astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0) or 1.0)
        X_std[col] = (values - mean) / (std if std != 0 else 1.0)
    return X_std


def corr_to_set(
    X_block: pd.DataFrame,
    ref_set: List[RefItem],
    sigma,
    cat_weights,
    combine: str = "product",
    mix_alpha: float = 0.7,
) -> np.ndarray:
    """
    计算每个样本对 ref_set 的相关度 E（长度=|X_block|）。
    数值特征列：dep_hour, arr_hour, block_time_min, Distance（标准化后）
    类别字段：UniqueCarrier, Origin, Dest, DayOfWeek, Month
    """

    if not ref_set:
        return np.zeros(len(X_block), dtype=float)
    X_std = _standardize_block(X_block)
    cat_weights = dict(cat_weights or {})
    sims = np.zeros(len(X_std), dtype=float)
    weight_sum = np.zeros(len(X_std), dtype=float)
    records: List[dict] = X_std.to_dict(orient="records")

    for item in ref_set:
        for pos, row in enumerate(records):
            sim = mixed_similarity(row, item, sigma=sigma, cat_weights=cat_weights, combine=combine, mix_alpha=mix_alpha)
            sims[pos] += sim * float(item.weight)
            weight_sum[pos] += float(item.weight)
    weight_sum = np.where(weight_sum <= 0, 1.0, weight_sum)
    return sims / weight_sum
