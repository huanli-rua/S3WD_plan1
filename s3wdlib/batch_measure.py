# -*- coding: utf-8 -*-
"""批级三域概率与整体测度计算模块。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["to_trisect_probs", "expected_cost", "expected_fbeta", "compute_region_masks"]


def to_trisect_probs(E_pos: np.ndarray, E_neg: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """返回 {'p_pos':..., 'p_neg':..., 'p_bnd':...}，逐元素归一化。"""

    E_pos = np.asarray(E_pos, dtype=float)
    E_neg = np.asarray(E_neg, dtype=float)
    total = np.clip(E_pos + E_neg, eps, None)
    p_pos = E_pos / total
    p_neg = E_neg / total
    p_bnd = np.clip(1.0 - p_pos - p_neg, 0.0, 1.0)
    return {"p_pos": p_pos, "p_neg": p_neg, "p_bnd": p_bnd}


def compute_region_masks(P: Dict[str, np.ndarray], alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据 α/β 计算正域、负域、边界域掩码。"""

    p_pos = P["p_pos"]
    positive = p_pos >= alpha
    negative = p_pos <= beta
    boundary = ~(positive | negative)
    return positive, negative, boundary


def expected_cost(
    P: Dict[str, np.ndarray],
    alpha: float,
    beta: float,
    c_fn: float,
    c_fp: float,
    c_bnd: float,
) -> float:
    """用三域概率近似 TP/FP/FN/BND 的期望并返回期望成本。"""

    positive, negative, boundary = compute_region_masks(P, alpha, beta)
    p_pos = P["p_pos"]
    p_neg = P["p_neg"]
    p_bnd = P["p_bnd"]

    exp_fn = float(np.sum(p_pos[negative]))
    exp_fp = float(np.sum(p_neg[positive]))
    exp_bnd = float(np.sum(p_bnd[boundary]))
    total = max(len(p_pos), 1)
    cost = (c_fn * exp_fn + c_fp * exp_fp + c_bnd * exp_bnd) / total
    return float(cost)


def expected_fbeta(
    P: Dict[str, np.ndarray],
    alpha: float,
    beta: float,
    beta_weight: float = 1.0,
) -> float:
    """返回期望 F_beta（用三域概率近似 TP/FP/FN 的期望计数）。"""

    beta_weight = max(float(beta_weight), 1e-6)
    positive, negative, _ = compute_region_masks(P, alpha, beta)
    p_pos = P["p_pos"]
    p_neg = P["p_neg"]

    exp_tp = float(np.sum(p_pos[positive]))
    exp_fp = float(np.sum(p_neg[positive]))
    exp_fn = float(np.sum(p_pos[negative]))

    precision = exp_tp / max(exp_tp + exp_fp, 1e-6)
    recall = exp_tp / max(exp_tp + exp_fn, 1e-6)
    beta_sq = beta_weight ** 2
    denom = beta_sq * precision + recall
    if denom <= 0:
        return 0.0
    f_beta = (1 + beta_sq) * precision * recall / denom
    return float(f_beta)
