# -*- coding: utf-8 -*-
"""动态阈值自适应模块。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import logging
import math

import numpy as np

from .objective import S3WDParams, s3wd_objective


_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


@dataclass
class ThresholdAdaptResult:
    """阈值搜索结果容器。"""

    alphas: np.ndarray
    betas: np.ndarray
    gamma: float
    fitness: float
    details: Dict[str, float]
    feasible: bool
    history: MutableMapping[str, List[np.ndarray]]
    method: str


def _enforce_gap(alpha: np.ndarray, beta: np.ndarray, gap: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(alpha, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy()
    for i in range(len(alpha)):
        if alpha[i] < beta[i] + gap:
            mid = 0.5 * (alpha[i] + beta[i])
            beta[i] = max(0.0, min(mid - 0.5 * gap, beta[i]))
            alpha[i] = min(1.0, max(beta[i] + gap, mid + 0.5 * gap))
        alpha[i] = float(np.clip(alpha[i], 0.0, 1.0))
        beta[i] = float(np.clip(beta[i], 0.0, 1.0))
        if alpha[i] < beta[i] + gap:
            alpha[i] = min(0.999, beta[i] + gap)
    return alpha, beta


def _calc_boundary_ratio(prob_levels: Sequence[np.ndarray], alphas: np.ndarray, betas: np.ndarray) -> float:
    ratios: List[float] = []
    for lvl, a, b in zip(prob_levels, alphas, betas):
        lvl_arr = np.asarray(lvl, dtype=float).ravel()
        if lvl_arr.size == 0:
            continue
        mask_bnd = (lvl_arr < a) & (lvl_arr > b)
        ratios.append(float(np.mean(mask_bnd)))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def _ensure_history(history: Optional[MutableMapping[str, List[np.ndarray]]]) -> MutableMapping[str, List[np.ndarray]]:
    if history is None:
        history = {}
    history.setdefault("alpha_raw", [])
    history.setdefault("beta_raw", [])
    return history


def _smooth_thresholds(alpha: np.ndarray, beta: np.ndarray, history: MutableMapping[str, List[np.ndarray]],
                       ema_alpha: float, median_window: int, gap: float) -> Tuple[np.ndarray, np.ndarray]:
    history["alpha_raw"].append(np.asarray(alpha, dtype=float))
    history["beta_raw"].append(np.asarray(beta, dtype=float))

    if len(history["alpha_raw"]) > median_window and median_window > 0:
        history["alpha_raw"] = history["alpha_raw"][-median_window:]
        history["beta_raw"] = history["beta_raw"][-median_window:]

    med_alpha = np.median(np.stack(history["alpha_raw"]), axis=0)
    med_beta = np.median(np.stack(history["beta_raw"]), axis=0)

    ema_alpha_prev = history.setdefault("alpha_ema", np.asarray(alpha, dtype=float))
    ema_beta_prev = history.setdefault("beta_ema", np.asarray(beta, dtype=float))

    ema_alpha_cur = ema_alpha * np.asarray(alpha, dtype=float) + (1.0 - ema_alpha) * ema_alpha_prev
    ema_beta_cur = ema_alpha * np.asarray(beta, dtype=float) + (1.0 - ema_alpha) * ema_beta_prev

    history["alpha_ema"] = ema_alpha_cur
    history["beta_ema"] = ema_beta_cur

    smooth_alpha = 0.5 * (ema_alpha_cur + med_alpha)
    smooth_beta = 0.5 * (ema_beta_cur + med_beta)

    smooth_alpha, smooth_beta = _enforce_gap(smooth_alpha, smooth_beta, gap)
    return smooth_alpha, smooth_beta


def adapt_thresholds_rule_based(
    prob_levels: Sequence[np.ndarray],
    y: Iterable[int],
    params: S3WDParams,
    *,
    keep_gap: Optional[float] = None,
    history: Optional[MutableMapping[str, List[np.ndarray]]] = None,
    ema_alpha: float = 0.6,
    median_window: int = 3,
    gamma_last: float | None = None,
) -> ThresholdAdaptResult:
    """基于经验规则的阈值自适应。"""

    keep_gap = float(keep_gap if keep_gap is not None else params.gap)
    prob_levels_np = [np.asarray(p, dtype=float).ravel() for p in prob_levels]
    y_arr = (np.asarray(y) > 0).astype(int)
    nL = len(prob_levels_np)

    alphas = np.empty(nL, dtype=float)
    betas = np.empty(nL, dtype=float)
    for idx, lvl in enumerate(prob_levels_np):
        if lvl.size == 0:
            alphas[idx] = 0.7
            betas[idx] = 0.3
            continue
        pos = lvl[y_arr == 1]
        neg = lvl[y_arr == 0]
        if pos.size == 0:
            pos = lvl
        if neg.size == 0:
            neg = lvl
        alphas[idx] = float(np.quantile(pos, 0.7))
        betas[idx] = float(np.quantile(neg, 0.3))

    alphas, betas = _enforce_gap(alphas, betas, keep_gap)
    history = _ensure_history(history)
    smooth_alpha, smooth_beta = _smooth_thresholds(alphas, betas, history, ema_alpha, median_window, keep_gap)

    gamma = 0.5 if gamma_last is None else float(gamma_last)
    fitness, detail = s3wd_objective(prob_levels_np, y_arr, smooth_alpha, smooth_beta, gamma, params)

    bnd_ratio = _calc_boundary_ratio(prob_levels_np, smooth_alpha, smooth_beta)
    feasible = bool(np.all(smooth_alpha >= smooth_beta + keep_gap))
    detail = dict(detail)
    detail.update({
        "bnd_ratio": float(bnd_ratio),
        "fitness": float(fitness),
        "feasible": feasible,
        "penalty_infeasible": 0.0,
        "method": "rule_based",
    })

    _logger.info(
        "Rule-based 阈值平滑后 α=%s, β=%s, 边界占比=%.3f, IG=%.4f, Regret=%.4f, 可行=%s",
        np.round(smooth_alpha, 4).tolist(),
        np.round(smooth_beta, 4).tolist(),
        bnd_ratio,
        detail.get("ig", 0.0),
        detail.get("regret", 0.0),
        "是" if feasible else "否",
    )

    return ThresholdAdaptResult(
        alphas=smooth_alpha,
        betas=smooth_beta,
        gamma=gamma,
        fitness=float(fitness),
        details=detail,
        feasible=feasible,
        history=history,
        method="rule_based",
    )


def adapt_thresholds_windowed_pso(
    prob_levels: Sequence[np.ndarray],
    y: Iterable[int],
    params: S3WDParams,
    *,
    particles: int = 12,
    iters: int = 30,
    seed: Optional[int] = None,
    keep_gap: Optional[float] = None,
    history: Optional[MutableMapping[str, List[np.ndarray]]] = None,
    window_size: Optional[int] = None,
    ema_alpha: float = 0.6,
    median_window: int = 3,
    gamma_last: float | None = None,
    stall_rounds: int = 6,
    fallback_rule: bool = True,
) -> ThresholdAdaptResult:
    """窗口化 PSO 阈值自适应。"""

    keep_gap = float(keep_gap if keep_gap is not None else params.gap)
    prob_levels_np = [np.asarray(p, dtype=float).ravel() for p in prob_levels]
    y_arr = (np.asarray(y) > 0).astype(int)
    nL = len(prob_levels_np)

    effective_samples = prob_levels_np[0].size if prob_levels_np else 0
    if window_size is None:
        window_size = effective_samples

    particles = int(max(4, min(particles, max(6, math.ceil(math.sqrt(max(1, window_size)))) * 2)))
    iters = int(max(12, min(iters, max(18, math.ceil(math.log2(max(2, window_size + 1)))) * 5)))

    rng = np.random.default_rng(seed)

    alpha_low, alpha_high = 0.45, 0.95
    beta_low, beta_high = 0.05, 0.45

    X = np.concatenate([
        rng.uniform(alpha_low, alpha_high, size=(particles, nL)),
        rng.uniform(beta_low, beta_high, size=(particles, nL)),
    ], axis=1)
    V = rng.uniform(-0.08, 0.08, size=X.shape)

    pbest = X.copy()
    pbest_fit = np.full(particles, np.inf, dtype=float)
    pbest_info: List[Dict[str, float]] = [{} for _ in range(particles)]
    pbest_alpha = np.zeros((particles, nL), dtype=float)
    pbest_beta = np.zeros((particles, nL), dtype=float)

    def _evaluate(vec: np.ndarray) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, bool]:
        raw_alpha = np.clip(vec[:nL], 0.0, 1.0)
        raw_beta = np.clip(vec[nL:], 0.0, 1.0)
        feasible_raw = bool(np.all(raw_alpha >= raw_beta + keep_gap))
        adj_alpha, adj_beta = _enforce_gap(raw_alpha, raw_beta, keep_gap)
        gamma_eval = 0.5 if gamma_last is None else float(gamma_last)
        fitness, detail = s3wd_objective(prob_levels_np, y_arr, adj_alpha, adj_beta, gamma_eval, params)
        penalty = 0.0 if feasible_raw else float(params.penalty_large)
        fitness_pen = float(fitness + penalty)
        det_map = dict(detail)
        det_map.update({
            "penalty_infeasible": penalty,
            "fitness": fitness_pen,
            "feasible": bool(np.all(adj_alpha >= adj_beta + keep_gap)),
        })
        return fitness_pen, det_map, adj_alpha, adj_beta, feasible_raw

    gbest_fit = np.inf
    gbest_alpha = np.zeros(nL, dtype=float)
    gbest_beta = np.zeros(nL, dtype=float)
    gbest_info: Dict[str, float] = {}

    for idx in range(particles):
        fit, info, adj_a, adj_b, _ = _evaluate(X[idx])
        pbest_fit[idx] = fit
        pbest[idx] = X[idx].copy()
        pbest_info[idx] = info
        pbest_alpha[idx] = adj_a
        pbest_beta[idx] = adj_b
        if fit < gbest_fit:
            gbest_fit = fit
            gbest_alpha = adj_a
            gbest_beta = adj_b
            gbest_info = info

    stall_count = 0
    for t in range(iters):
        w = 0.7 - 0.3 * (t / max(1, iters - 1))
        r1 = rng.random(size=X.shape)
        r2 = rng.random(size=X.shape)
        V = 0.5 * V + 1.4 * r1 * (pbest - X) + 1.2 * r2 * (np.concatenate([gbest_alpha, gbest_beta]) - X)
        X = np.clip(X + w * V, 0.0, 1.0)

        improved = False
        for idx in range(particles):
            fit, info, adj_a, adj_b, feasible_raw = _evaluate(X[idx])
            if fit < pbest_fit[idx]:
                pbest_fit[idx] = fit
                pbest[idx] = X[idx].copy()
                pbest_info[idx] = info
                pbest_alpha[idx] = adj_a
                pbest_beta[idx] = adj_b
            if fit < gbest_fit:
                gbest_fit = fit
                gbest_alpha = adj_a
                gbest_beta = adj_b
                gbest_info = info
                improved = True
        if improved:
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= stall_rounds:
            _logger.info("窗口化 PSO 在第 %d 轮提前收敛。", t + 1)
            break

    history = _ensure_history(history)
    smooth_alpha, smooth_beta = _smooth_thresholds(gbest_alpha, gbest_beta, history, ema_alpha, median_window, keep_gap)
    gamma = 0.5 if gamma_last is None else float(gamma_last)
    final_fit, final_detail = s3wd_objective(prob_levels_np, y_arr, smooth_alpha, smooth_beta, gamma, params)
    bnd_ratio = _calc_boundary_ratio(prob_levels_np, smooth_alpha, smooth_beta)
    feasible = bool(np.all(smooth_alpha >= smooth_beta + keep_gap))

    detail = dict(final_detail)
    detail.update({
        "bnd_ratio": float(bnd_ratio),
        "fitness": float(final_fit),
        "feasible": feasible,
        "penalty_infeasible": gbest_info.get("penalty_infeasible", 0.0),
        "method": "windowed_pso",
    })

    if (not feasible or not np.isfinite(final_fit)) and fallback_rule:
        _logger.info("窗口化 PSO 找到的解不可行或数值异常，切换到 rule-based 兜底方案。")
        return adapt_thresholds_rule_based(
            prob_levels_np,
            y_arr,
            params,
            keep_gap=keep_gap,
            history=history,
            ema_alpha=ema_alpha,
            median_window=median_window,
            gamma_last=gamma_last,
        )

    _logger.info(
        "窗口化 PSO 平滑后阈值 α=%s, β=%s, 边界占比=%.3f, IG=%.4f, Regret=%.4f, 可行=%s",
        np.round(smooth_alpha, 4).tolist(),
        np.round(smooth_beta, 4).tolist(),
        bnd_ratio,
        detail.get("ig", 0.0),
        detail.get("regret", 0.0),
        "是" if feasible else "否",
    )

    return ThresholdAdaptResult(
        alphas=smooth_alpha,
        betas=smooth_beta,
        gamma=gamma,
        fitness=float(final_fit),
        details=detail,
        feasible=feasible,
        history=history,
        method="windowed_pso",
    )
