# -*- coding: utf-8 -*-
"""动态阈值自适应模块。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import logging
import math

import numpy as np

from collections import deque

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
    history.setdefault("gamma_raw", [])
    return history


def _smooth_thresholds(
    alpha: np.ndarray,
    beta: np.ndarray,
    history: MutableMapping[str, List[np.ndarray]],
    ema_alpha: float,
    median_window: int,
    gap: float,
    *,
    gamma: Optional[float] = None,
    ema_gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    history["alpha_raw"].append(np.asarray(alpha, dtype=float))
    history["beta_raw"].append(np.asarray(beta, dtype=float))
    if gamma is not None:
        history["gamma_raw"].append(np.asarray([gamma], dtype=float))

    if len(history["alpha_raw"]) > median_window and median_window > 0:
        history["alpha_raw"] = history["alpha_raw"][-median_window:]
        history["beta_raw"] = history["beta_raw"][-median_window:]
        if gamma is not None:
            history["gamma_raw"] = history["gamma_raw"][-median_window:]

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

    smooth_gamma: Optional[float] = gamma
    if gamma is not None:
        gamma_values = np.concatenate(history.get("gamma_raw", [])) if history.get("gamma_raw") else np.asarray([gamma])
        med_gamma = float(np.median(gamma_values))
        ema_gamma_coeff = float(ema_gamma if ema_gamma is not None else ema_alpha)
        ema_gamma_prev = float(history.setdefault("gamma_ema", float(gamma)))
        ema_gamma_cur = float(ema_gamma_coeff * float(gamma) + (1.0 - ema_gamma_coeff) * ema_gamma_prev)
        history["gamma_ema"] = np.asarray([ema_gamma_cur], dtype=float)
        smooth_gamma = float(0.5 * (ema_gamma_cur + med_gamma))
        smooth_gamma = float(np.clip(smooth_gamma, 0.01, 0.99))

    return smooth_alpha, smooth_beta, smooth_gamma


def _hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.0, 1.0))
    total = float(np.sum(counts))
    if total <= 0.0:
        return float(np.clip(q, 0.0, 1.0))
    target = q * total
    cumsum = np.cumsum(counts)
    idx = int(np.searchsorted(cumsum, target, side="right"))
    idx = min(max(idx, 0), len(edges) - 2)
    prev_cum = float(cumsum[idx - 1]) if idx > 0 else 0.0
    bin_count = float(counts[idx])
    left = float(edges[idx])
    right = float(edges[idx + 1])
    if bin_count <= 0.0:
        return left
    frac = float(np.clip((target - prev_cum) / max(bin_count, 1e-12), 0.0, 1.0))
    return float(left + frac * (right - left))


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
    ema_gamma: Optional[float] = None,
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
    gamma = 0.5 if gamma_last is None else float(gamma_last)
    smooth_alpha, smooth_beta, smooth_gamma = _smooth_thresholds(
        alphas,
        betas,
        history,
        ema_alpha,
        median_window,
        keep_gap,
        gamma=gamma,
        ema_gamma=ema_gamma,
    )

    gamma = smooth_gamma if smooth_gamma is not None else gamma
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
        gamma=float(gamma),
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
    search_gamma: bool = True,
    gamma_bounds: Tuple[float, float] = (0.3, 0.7),
    ema_gamma: Optional[float] = None,
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

    gamma_low, gamma_high = gamma_bounds
    if search_gamma:
        gamma_low = float(np.clip(gamma_low, 0.01, 0.99))
        gamma_high = float(np.clip(gamma_high, gamma_low + 1e-3, 0.99))

    base_particles = [
        rng.uniform(alpha_low, alpha_high, size=(particles, nL)),
        rng.uniform(beta_low, beta_high, size=(particles, nL)),
    ]
    if search_gamma:
        base_particles.append(rng.uniform(gamma_low, gamma_high, size=(particles, 1)))

    X = np.concatenate(base_particles, axis=1)
    V = rng.uniform(-0.08, 0.08, size=X.shape)

    pbest = X.copy()
    pbest_fit = np.full(particles, np.inf, dtype=float)
    pbest_info: List[Dict[str, float]] = [{} for _ in range(particles)]
    pbest_alpha = np.zeros((particles, nL), dtype=float)
    pbest_beta = np.zeros((particles, nL), dtype=float)

    def _evaluate(vec: np.ndarray) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, float, bool]:
        raw_alpha = np.clip(vec[:nL], 0.0, 1.0)
        raw_beta = np.clip(vec[nL:], 0.0, 1.0)
        if search_gamma:
            raw_gamma = float(np.clip(vec[2 * nL], 0.01, 0.99))
        else:
            raw_gamma = 0.5 if gamma_last is None else float(gamma_last)
        feasible_raw = bool(np.all(raw_alpha >= raw_beta + keep_gap))
        adj_alpha, adj_beta = _enforce_gap(raw_alpha, raw_beta, keep_gap)
        gamma_eval = raw_gamma
        fitness, detail = s3wd_objective(prob_levels_np, y_arr, adj_alpha, adj_beta, gamma_eval, params)
        penalty = 0.0 if feasible_raw else float(params.penalty_large)
        fitness_pen = float(fitness + penalty)
        det_map = dict(detail)
        det_map.update({
            "penalty_infeasible": penalty,
            "fitness": fitness_pen,
            "feasible": bool(np.all(adj_alpha >= adj_beta + keep_gap)),
        })
        return fitness_pen, det_map, adj_alpha, adj_beta, gamma_eval, feasible_raw

    gbest_fit = np.inf
    gbest_alpha = np.zeros(nL, dtype=float)
    gbest_beta = np.zeros(nL, dtype=float)
    gbest_gamma = 0.5 if gamma_last is None else float(gamma_last)
    gbest_info: Dict[str, float] = {}

    for idx in range(particles):
        fit, info, adj_a, adj_b, g_val, _ = _evaluate(X[idx])
        pbest_fit[idx] = fit
        pbest[idx] = X[idx].copy()
        pbest_info[idx] = info
        pbest_alpha[idx] = adj_a
        pbest_beta[idx] = adj_b
        if search_gamma:
            pbest_gamma_val = history.setdefault("pbest_gamma", np.zeros(particles))
            pbest_gamma_val[idx] = g_val
        if fit < gbest_fit:
            gbest_fit = fit
            gbest_alpha = adj_a
            gbest_beta = adj_b
            gbest_gamma = g_val
            gbest_info = info

    stall_count = 0
    for t in range(iters):
        w = 0.7 - 0.3 * (t / max(1, iters - 1))
        r1 = rng.random(size=X.shape)
        r2 = rng.random(size=X.shape)
        gbest_concat = [gbest_alpha, gbest_beta]
        if search_gamma:
            gbest_concat.append(np.asarray([gbest_gamma]))
        V = 0.5 * V + 1.4 * r1 * (pbest - X) + 1.2 * r2 * (np.concatenate(gbest_concat) - X)
        X = np.clip(X + w * V, 0.0, 1.0)

        improved = False
        for idx in range(particles):
            fit, info, adj_a, adj_b, g_val, feasible_raw = _evaluate(X[idx])
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
                gbest_gamma = g_val
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
    gamma_value = gbest_gamma if search_gamma else (0.5 if gamma_last is None else float(gamma_last))
    smooth_alpha, smooth_beta, smooth_gamma = _smooth_thresholds(
        gbest_alpha,
        gbest_beta,
        history,
        ema_alpha,
        median_window,
        keep_gap,
        gamma=gamma_value,
        ema_gamma=ema_gamma,
    )
    gamma = smooth_gamma if smooth_gamma is not None else gamma_value
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
        gamma=float(gamma),
        fitness=float(final_fit),
        details=detail,
        feasible=feasible,
        history=history,
        method="windowed_pso",
    )


def adapt_thresholds_s_lite(
    prob_levels: Sequence[np.ndarray],
    y: Iterable[int],
    params: S3WDParams,
    *,
    history: Optional[MutableMapping[str, List[np.ndarray]]] = None,
    keep_gap: Optional[float] = None,
    approx_bins: int = 256,
    ema_alpha: float = 0.6,
    ema_gamma: Optional[float] = None,
    median_window: int = 3,
    step_max: float = 0.05,
    hysteresis: float = 0.01,
    target_bnd: Optional[float] = None,
    safe_mode: bool = False,
    gamma_last: Optional[float] = None,
    replay_size: int = 20,
) -> ThresholdAdaptResult:
    """S-lite·Fast 阈值更新：量化桶 + EMA + 限幅 + 滞回 + 成本感知。"""

    keep_gap = float(keep_gap if keep_gap is not None else params.gap)
    prob_levels_np = [np.asarray(p, dtype=float).ravel() for p in prob_levels]
    y_arr = (np.asarray(y) > 0).astype(int)
    nL = len(prob_levels_np)

    history = _ensure_history(history)
    bins = int(max(32, approx_bins))
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=float)

    hist_pos = history.setdefault("s_hist_pos", np.zeros((nL, bins), dtype=float))
    hist_neg = history.setdefault("s_hist_neg", np.zeros((nL, bins), dtype=float))
    tau_alpha = history.setdefault("s_tau_alpha", np.full(nL, 0.7, dtype=float))
    tau_beta = history.setdefault("s_tau_beta", np.full(nL, 0.3, dtype=float))
    tau_gamma = float(history.setdefault("s_tau_gamma", np.asarray([0.5], dtype=float))[0])

    theta_pos = float(params.theta_pos)
    theta_neg = float(params.theta_neg)
    cost_ratio = theta_pos / max(theta_pos + theta_neg, 1e-6)

    pos_hist_cur = np.zeros_like(hist_pos)
    neg_hist_cur = np.zeros_like(hist_neg)

    for idx, lvl in enumerate(prob_levels_np):
        if lvl.size == 0:
            continue
        pos_vals = lvl[y_arr == 1]
        neg_vals = lvl[y_arr == 0]
        if pos_vals.size > 0:
            pos_hist_cur[idx] = np.histogram(pos_vals, bins=edges)[0]
        if neg_vals.size > 0:
            neg_hist_cur[idx] = np.histogram(neg_vals, bins=edges)[0]

    hist_pos *= ema_alpha
    hist_pos += (1.0 - ema_alpha) * pos_hist_cur
    hist_neg *= ema_alpha
    hist_neg += (1.0 - ema_alpha) * neg_hist_cur

    if safe_mode:
        tau_alpha = np.clip(tau_alpha - 0.05, 0.4, 0.95)
        tau_beta = np.clip(tau_beta + 0.05, 0.05, 0.6)
        tau_gamma = float(np.clip(tau_gamma - 0.05, 0.2, 0.9))
    else:
        adjust = (cost_ratio - 0.5) * 0.1
        tau_alpha = np.clip(tau_alpha + adjust, 0.45, 0.95)
        tau_beta = np.clip(tau_beta - adjust, 0.05, 0.55)
        tau_gamma = float(np.clip(tau_gamma + adjust * 0.5, 0.2, 0.9))

    history["s_tau_alpha"] = tau_alpha
    history["s_tau_beta"] = tau_beta
    history["s_tau_gamma"] = np.asarray([tau_gamma], dtype=float)

    alpha_candidates = np.zeros(nL, dtype=float)
    beta_candidates = np.zeros(nL, dtype=float)

    for idx in range(nL):
        alpha_candidates[idx] = _hist_quantile(hist_pos[idx], edges, float(tau_alpha[idx]))
        beta_candidates[idx] = _hist_quantile(hist_neg[idx], edges, float(tau_beta[idx]))

    prev_alpha = np.asarray(history.get("s_prev_alpha", alpha_candidates), dtype=float)
    prev_beta = np.asarray(history.get("s_prev_beta", beta_candidates), dtype=float)

    delta_alpha = np.clip(alpha_candidates - prev_alpha, -step_max, step_max)
    delta_beta = np.clip(beta_candidates - prev_beta, -step_max, step_max)

    alpha_candidates = prev_alpha + delta_alpha
    beta_candidates = prev_beta + delta_beta

    mask_alpha = np.abs(alpha_candidates - prev_alpha) < hysteresis
    alpha_candidates[mask_alpha] = prev_alpha[mask_alpha]

    mask_beta = np.abs(beta_candidates - prev_beta) < hysteresis
    beta_candidates[mask_beta] = prev_beta[mask_beta]

    alpha_candidates, beta_candidates = _enforce_gap(alpha_candidates, beta_candidates, keep_gap)

    history["s_prev_alpha"] = alpha_candidates.copy()
    history["s_prev_beta"] = beta_candidates.copy()

    gamma_candidate = _hist_quantile(hist_pos[-1] + hist_neg[-1], edges, tau_gamma)
    if not np.isfinite(gamma_candidate):
        gamma_candidate = float(gamma_last if gamma_last is not None else 0.5)

    target_bnd = float(target_bnd) if target_bnd is not None else None
    if target_bnd is not None:
        bnd_ratio = _calc_boundary_ratio(prob_levels_np[:-1], alpha_candidates[:-1], beta_candidates[:-1])
        gap_adjust = np.clip(target_bnd - bnd_ratio, -0.05, 0.05)
        alpha_candidates[:-1] = np.clip(alpha_candidates[:-1] + gap_adjust, keep_gap, 0.999)
        beta_candidates[:-1] = np.clip(beta_candidates[:-1] - gap_adjust, 0.0, 0.999)
        alpha_candidates, beta_candidates = _enforce_gap(alpha_candidates, beta_candidates, keep_gap)

    smooth_alpha, smooth_beta, smooth_gamma = _smooth_thresholds(
        alpha_candidates,
        beta_candidates,
        history,
        ema_alpha,
        median_window,
        keep_gap,
        gamma=gamma_candidate,
        ema_gamma=ema_gamma,
    )

    gamma_value = float(smooth_gamma if smooth_gamma is not None else gamma_candidate)
    fitness, detail = s3wd_objective(prob_levels_np, y_arr, smooth_alpha, smooth_beta, gamma_value, params)
    bnd_ratio = _calc_boundary_ratio(prob_levels_np[:-1], smooth_alpha[:-1], smooth_beta[:-1]) if nL > 1 else 0.0

    feasible = bool(np.all(smooth_alpha >= smooth_beta + keep_gap))
    detail = dict(detail)
    detail.update({
        "bnd_ratio": float(bnd_ratio),
        "fitness": float(fitness),
        "feasible": feasible,
        "safe_mode": bool(safe_mode),
        "method": "s_lite",
        "tau_alpha_mean": float(np.mean(tau_alpha)),
        "tau_beta_mean": float(np.mean(tau_beta)),
        "tau_gamma": float(tau_gamma),
    })

    buffer: Deque[Dict[str, np.ndarray]] = history.setdefault("s_replay", deque(maxlen=max(1, int(replay_size))))  # type: ignore[assignment]
    buffer.append({
        "prob": [lvl.copy() for lvl in prob_levels_np],
        "y": y_arr.copy(),
    })

    return ThresholdAdaptResult(
        alphas=smooth_alpha,
        betas=smooth_beta,
        gamma=gamma_value,
        fitness=float(fitness),
        details=detail,
        feasible=feasible,
        history=history,
        method="s_lite",
    )


def rlite_calibration(
    history: MutableMapping[str, List[np.ndarray]],
    params: S3WDParams,
    *,
    keep_gap: Optional[float] = None,
    grid_radius: int = 2,
    grid_step: float = 0.02,
    ema_decay: float = 0.6,
) -> bool:
    """R-lite：基于最近窗口回放的离线量化校正。"""

    keep_gap = float(keep_gap if keep_gap is not None else params.gap)
    replay: Deque[Dict[str, np.ndarray]] = history.get("s_replay")  # type: ignore[assignment]
    if not replay:
        return False

    tau_alpha = np.asarray(history.get("s_tau_alpha", np.full(3, 0.7)), dtype=float)
    tau_beta = np.asarray(history.get("s_tau_beta", np.full(3, 0.3)), dtype=float)
    tau_gamma = float(np.asarray(history.get("s_tau_gamma", np.asarray([0.5])))[0])

    prob_levels = list(zip(*[chunk["prob"] for chunk in replay]))
    y_concat = np.concatenate([chunk["y"] for chunk in replay])
    prob_levels_concat = [np.concatenate(level) for level in prob_levels]

    def evaluate(tau_a: np.ndarray, tau_b: np.ndarray, tau_g: float) -> Tuple[float, np.ndarray, np.ndarray, float]:
        alphas = np.zeros_like(tau_a)
        betas = np.zeros_like(tau_b)
        for idx, lvl in enumerate(prob_levels_concat):
            pos = lvl[y_concat == 1]
            neg = lvl[y_concat == 0]
            alphas[idx] = float(np.quantile(pos if pos.size > 0 else lvl, float(np.clip(tau_a[idx], 0.01, 0.99))))
            betas[idx] = float(np.quantile(neg if neg.size > 0 else lvl, float(np.clip(tau_b[idx], 0.01, 0.99))))
        alphas, betas = _enforce_gap(alphas, betas, keep_gap)
        gamma_val = float(np.quantile(prob_levels_concat[-1], float(np.clip(tau_g, 0.01, 0.99))))
        fit, _ = s3wd_objective(prob_levels_concat, y_concat, alphas, betas, gamma_val, params)
        return float(fit), alphas, betas, gamma_val

    best_tau_alpha = tau_alpha.copy()
    best_tau_beta = tau_beta.copy()
    best_tau_gamma = tau_gamma
    best_fit, _, _, _ = evaluate(best_tau_alpha, best_tau_beta, best_tau_gamma)

    improved = False
    for idx in range(len(tau_alpha)):
        candidates = [tau_alpha[idx] + step * grid_step for step in range(-grid_radius, grid_radius + 1)]
        candidates = [float(np.clip(c, 0.05, 0.95)) for c in candidates]
        for cand in candidates:
            trial_tau_alpha = best_tau_alpha.copy()
            trial_tau_alpha[idx] = cand
            fit, _, _, _ = evaluate(trial_tau_alpha, best_tau_beta, best_tau_gamma)
            if fit < best_fit:
                best_fit = fit
                best_tau_alpha = trial_tau_alpha
                improved = True

    for idx in range(len(tau_beta)):
        candidates = [tau_beta[idx] + step * grid_step for step in range(-grid_radius, grid_radius + 1)]
        candidates = [float(np.clip(c, 0.02, 0.8)) for c in candidates]
        for cand in candidates:
            trial_tau_beta = best_tau_beta.copy()
            trial_tau_beta[idx] = cand
            fit, _, _, _ = evaluate(best_tau_alpha, trial_tau_beta, best_tau_gamma)
            if fit < best_fit:
                best_fit = fit
                best_tau_beta = trial_tau_beta
                improved = True

    gamma_candidates = [best_tau_gamma + step * grid_step for step in range(-grid_radius, grid_radius + 1)]
    gamma_candidates = [float(np.clip(c, 0.1, 0.95)) for c in gamma_candidates]
    for cand in gamma_candidates:
        fit, _, _, _ = evaluate(best_tau_alpha, best_tau_beta, cand)
        if fit < best_fit:
            best_fit = fit
            best_tau_gamma = cand
            improved = True

    if improved:
        history["s_tau_alpha"] = ema_decay * tau_alpha + (1.0 - ema_decay) * best_tau_alpha
        history["s_tau_beta"] = ema_decay * tau_beta + (1.0 - ema_decay) * best_tau_beta
        history["s_tau_gamma"] = np.asarray([ema_decay * tau_gamma + (1.0 - ema_decay) * best_tau_gamma])
    return improved
