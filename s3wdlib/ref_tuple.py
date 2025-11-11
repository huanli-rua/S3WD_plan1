# -*- coding: utf-8 -*-
"""参考元组 Γ/Ψ 的构建与重建逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["RefItem", "build_ref_tuples", "combine_history"]


@dataclass
class RefItem:
    """参考元组元素的数据结构。"""

    num_vec: np.ndarray
    cat_fields: dict
    weight: float


def _select_top_indices(values: np.ndarray, topk: int, quantile: float, reverse: bool = True) -> np.ndarray:
    """按权重选择 Top-K 元素，支持分位数筛选。"""

    if values.size == 0:
        return np.asarray([], dtype=int)
    threshold = np.quantile(values, quantile) if values.size > 1 else values[0]
    if reverse:
        mask = values >= threshold
    else:
        mask = values <= threshold
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.arange(values.size)
    order = np.argsort(values[idx])
    if reverse:
        order = order[::-1]
    sorted_idx = idx[order]
    return sorted_idx[: min(topk, sorted_idx.size)]


def _prepare_numeric(X: pd.DataFrame, cols: Iterable[str]) -> Tuple[pd.DataFrame, dict]:
    """对数值列做标准化，返回变换后的 DataFrame 与统计量。"""

    stats = {"mean": {}, "std": {}}
    if not cols:
        return pd.DataFrame(index=X.index), stats
    X_num = X.loc[:, cols].copy()
    for col in cols:
        values = pd.to_numeric(X_num[col], errors="coerce").astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0) or 1.0)
        stats["mean"][col] = mean
        stats["std"][col] = std
        X_num[col] = (values - mean) / (std if std != 0 else 1.0)
    return X_num, stats


def combine_history(new_refs: Dict[str, dict], old_refs: Dict[str, dict] | None, keep_ratio: float) -> Dict[str, dict]:
    """将历史参考元组与新构建结果按比例合并。"""

    if not old_refs or keep_ratio <= 0:
        return new_refs
    merged: Dict[str, dict] = {}
    for bucket, ref in new_refs.items():
        history = old_refs.get(bucket, {"pos": [], "neg": []})
        merged[bucket] = {
            "pos": _blend_items(ref.get("pos", []), history.get("pos", []), keep_ratio),
            "neg": _blend_items(ref.get("neg", []), history.get("neg", []), keep_ratio),
        }
    return merged


def _blend_items(new_items: List[RefItem], old_items: List[RefItem], keep_ratio: float) -> List[RefItem]:
    """按照历史比例截取旧元素，并与新元素拼接。"""

    if not old_items or keep_ratio <= 0:
        return new_items
    keep_cnt = int(np.ceil(len(new_items) * keep_ratio))
    keep_cnt = min(len(old_items), max(0, keep_cnt))
    selected_old = old_items[:keep_cnt]
    return selected_old + new_items


def build_ref_tuples(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    buckets: np.ndarray,
    topk_per_class: int,
    pos_quantile: float,
    keep_history_ratio: float = 0.3,
    gwb_prob: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    weight_clip: float | None = None,
) -> Dict[str, dict]:
    """
    在每个桶内构建参考元组：
    - Γ_pos：y=1 且延误强度/置信度高的前 topk_per_class
    - Ψ_neg：y=0 且延误为负或很小的前 topk_per_class
    - gwb_prob（可选）：作为置信度权重
    - sample_weight（可选）：统一的样本重要性权重（结合时间/季节/漂移/置信）
    - weight_clip（可选）：截断样本权重的上限
    返回：dict[bucket_id] -> {'pos': list[RefItem], 'neg': list[RefItem]}
    RefItem = {'num_vec': np.ndarray, 'cat_fields': dict, 'weight': float}
    """

    if len(X) != len(y):
        raise ValueError("X 与 y 的行数不一致，无法构建参考元组。")
    if len(X) != len(buckets):
        raise ValueError("buckets 长度必须与 X 对齐。")

    y_arr = np.asarray(y, dtype=int)
    prob = (
        np.asarray(gwb_prob, dtype=float).reshape(-1)
        if gwb_prob is not None
        else np.full(len(X), 0.5, dtype=float)
    )
    base_weight = (
        np.asarray(sample_weight, dtype=float).reshape(-1)
        if sample_weight is not None
        else np.ones(len(X), dtype=float)
    )
    if base_weight.size != len(X):
        raise ValueError("sample_weight 长度必须与 X 行数一致。")
    if weight_clip is not None and weight_clip > 0:
        base_weight = np.clip(base_weight, 1e-6, float(weight_clip))
    else:
        base_weight = np.clip(base_weight, 1e-6, None)

    pos_score = base_weight * np.clip(prob, 1e-6, None)
    neg_score = base_weight * np.clip(1.0 - prob, 1e-6, None)

    num_cols = [c for c in ["dep_hour", "arr_hour", "block_time_min", "Distance"] if c in X.columns]
    X_num, stats = _prepare_numeric(X, num_cols)

    cat_cols = [c for c in ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month"] if c in X.columns]
    cat_frame = X[cat_cols] if cat_cols else pd.DataFrame(index=X.index)

    ref_map: Dict[str, dict] = {}
    df_idx = pd.Index(buckets, name="bucket")
    grouped = pd.Series(np.arange(len(X)), index=df_idx)

    for bucket_id, idx_values in grouped.groupby(level=0):
        indices = idx_values.to_numpy()
        if indices.size == 0:
            continue
        pos_mask = y_arr[indices] == 1
        neg_mask = ~pos_mask
        pos_idx = indices[pos_mask]
        neg_idx = indices[neg_mask]

        bucket_refs = {"pos": [], "neg": []}

        if pos_idx.size:
            pos_scores = pos_score[pos_idx]
            selected = _select_top_indices(pos_scores, topk_per_class, pos_quantile, reverse=True)
            for order_idx in pos_idx[selected]:
                bucket_refs["pos"].append(
                    RefItem(
                        num_vec=X_num.iloc[order_idx].to_numpy(dtype=float) if num_cols else np.asarray([], dtype=float),
                        cat_fields={col: str(cat_frame.iloc[order_idx][col]) for col in cat_cols},
                        weight=float(base_weight[order_idx]),
                    )
                )

        if neg_idx.size:
            neg_scores = neg_score[neg_idx]
            selected_neg = _select_top_indices(neg_scores, topk_per_class, 1 - pos_quantile, reverse=True)
            for order_idx in neg_idx[selected_neg]:
                bucket_refs["neg"].append(
                    RefItem(
                        num_vec=X_num.iloc[order_idx].to_numpy(dtype=float) if num_cols else np.asarray([], dtype=float),
                        cat_fields={col: str(cat_frame.iloc[order_idx][col]) for col in cat_cols},
                        weight=float(base_weight[order_idx]),
                    )
                )

        ref_map[str(bucket_id)] = bucket_refs

    X.attrs.setdefault("num_stats", stats)
    return ref_map
