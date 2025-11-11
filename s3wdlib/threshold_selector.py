# -*- coding: utf-8 -*-
"""小网格阈值搜索模块。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .batch_measure import expected_cost, expected_fbeta, compute_region_masks

__all__ = ["select_alpha_beta"]


def _frange(start: float, end: float, step: float) -> np.ndarray:
    """生成闭区间的浮点网格。"""

    if step <= 0:
        raise ValueError("step 必须为正数。")
    values = np.arange(start, end + step * 0.5, step, dtype=float)
    return np.clip(values, 0.0, 1.0)


def _calc_metrics(P: Dict[str, np.ndarray], alpha: float, beta: float) -> Tuple[float, float]:
    """返回 (pos_coverage, bnd_ratio)。"""

    positive, negative, boundary = compute_region_masks(P, alpha, beta)
    total = max(len(P["p_pos"]), 1)
    pos_cov = float(np.sum(positive)) / total
    bnd_ratio = float(np.sum(boundary)) / total
    return pos_cov, bnd_ratio


def select_alpha_beta(
    P: Dict[str, np.ndarray],
    grid: Dict[str, list | tuple],
    constraints: Dict[str, float],
    objective: str = "expected_cost",
    costs: Dict[str, float] | None = None,
    beta_weight: float = 1.0,
):
    """
    网格：
      alpha: [a0, a1, step]，beta: [b0, b1, step]
    约束：
      alpha >= beta + keep_gap
      POS_coverage(alpha, beta) >= min_pos_coverage
      BND_ratio(alpha, beta) <= bnd_cap
    返回：(alpha_star, beta_star, best_score)
    """

    alpha_grid = _frange(*grid.get("alpha", [0.5, 0.9, 0.02]))
    beta_grid = _frange(*grid.get("beta", [0.05, 0.45, 0.02]))

    keep_gap = float(constraints.get("keep_gap", 0.0))
    min_pos_coverage = float(constraints.get("min_pos_coverage", 0.0))
    bnd_cap = float(constraints.get("bnd_cap", 1.0))

    best_alpha = float(alpha_grid[0])
    best_beta = float(beta_grid[0])
    best_val = np.inf if objective == "expected_cost" else -np.inf

    c_fn = float(costs.get("c_fn", 1.0)) if costs else 1.0
    c_fp = float(costs.get("c_fp", 1.0)) if costs else 1.0
    c_bnd = float(costs.get("c_bnd", 0.5)) if costs else 0.5

    feasible_found = False
    candidates: List[Tuple[float, float, float]] = []

    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha < beta + keep_gap:
                continue
            pos_cov, bnd_ratio = _calc_metrics(P, alpha, beta)
            if pos_cov < min_pos_coverage or bnd_ratio > bnd_cap:
                continue
            feasible_found = True
            if objective == "expected_cost":
                score = expected_cost(P, alpha, beta, c_fn, c_fp, c_bnd)
                if score < best_val:
                    best_val = score
                    best_alpha, best_beta = float(alpha), float(beta)
            else:
                score = expected_fbeta(P, alpha, beta, beta_weight=beta_weight)
                if score > best_val:
                    best_val = score
                    best_alpha, best_beta = float(alpha), float(beta)
            candidates.append((float(alpha), float(beta), float(score)))

    if not feasible_found:
        # 若无可行解，则退回到约束最接近的组合
        fallback_alpha = float(alpha_grid[-1])
        fallback_beta = float(min(beta_grid[-1], fallback_alpha - keep_gap))
        best_alpha, best_beta = fallback_alpha, fallback_beta
        if objective == "expected_cost":
            best_val = expected_cost(P, best_alpha, best_beta, c_fn, c_fp, c_bnd)
        else:
            best_val = expected_fbeta(P, best_alpha, best_beta, beta_weight=beta_weight)

    if objective == "expected_cost":
        sort_reverse = False
    else:
        sort_reverse = True

    if candidates:
        ordered = sorted(candidates, key=lambda item: item[2], reverse=sort_reverse)
        best_tuple = ordered[0]
        second_tuple = ordered[1] if len(ordered) > 1 else ordered[0]
        best_score = float(best_tuple[2])
        second_score = float(second_tuple[2])
        if objective == "expected_cost":
            delta = float(abs(second_score - best_score))
        else:
            delta = float(abs(best_score - second_score))
        best_point = (float(best_tuple[0]), float(best_tuple[1]))
        second_point = (float(second_tuple[0]), float(second_tuple[1]))
        feasible_count = len(candidates)
    else:
        best_score = float(best_val)
        second_score = float("nan")
        delta = float("nan")
        best_point = (best_alpha, best_beta)
        second_point = (float("nan"), float("nan"))
        feasible_count = 0

    info = {
        "best_score": best_score,
        "second_score": second_score,
        "delta": delta,
        "feasible_count": feasible_count,
        "best_point": best_point,
        "second_point": second_point,
        "objective": objective,
    }

    return best_alpha, best_beta, info
