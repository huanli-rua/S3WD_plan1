# -*- coding: utf-8 -*-
"""动态阈值自适应模块。"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import logging
import math

import numpy as np

from .objective import S3WDParams, s3wd_objective

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .trainer import PSOParams


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


@dataclass
class DynamicAdaptConfig:
    """动态阈值策略配置（YAML 的 DYN 分组）。"""

    enabled: bool = True
    strategy: str = "windowed_pso"
    step: int = 1
    target_bnd: float = 0.12
    ema_alpha: float = 0.6
    median_window: int = 3
    keep_gap: Optional[float] = None
    window_size: Optional[int] = None
    stall_rounds: int = 6
    fallback_rule: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DynamicAdaptConfig":
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    @staticmethod
    def ensure(cfg: Optional[Mapping[str, Any] | "DynamicAdaptConfig"]) -> "DynamicAdaptConfig":
        if cfg is None:
            return DynamicAdaptConfig()
        if isinstance(cfg, DynamicAdaptConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return DynamicAdaptConfig.from_mapping(cfg)
        raise TypeError(f"无法识别的动态阈值配置类型: {type(cfg)!r}")

    def resolved_strategy(self) -> str:
        name = (self.strategy or "windowed_pso").lower()
        if name in {"windowed_pso", "windowed-pso", "pso"}:
            return "windowed_pso"
        if name in {"rule_based", "rule-based", "rule"}:
            return "rule_based"
        raise ValueError(f"Unsupported dynamic strategy: {self.strategy}")

    def should_update(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        step = int(self.step) if self.step is not None else 0
        if step <= 0:
            return True
        iteration = max(1, int(iteration))
        return iteration % step == 0

    def apply_to_pso(self, params: "PSOParams") -> "PSOParams":
        params.window_mode = True
        if self.window_size is not None:
            params.window_size = int(self.window_size)
        params.ema_alpha = float(self.ema_alpha)
        params.median_window = int(self.median_window)
        params.keep_gap = self.keep_gap
        params.fallback_rule = bool(self.fallback_rule)
        if hasattr(params, "stall_rounds"):
            params.stall_rounds = int(self.stall_rounds)
        return params


@dataclass
class IncrementalUpdateConfig:
    """增量学习配置（YAML 的 INCR 分组）。"""

    enabled: bool = False
    strategy: str = "rolling"
    step: int = 1
    buffer_size: int = 2048
    max_samples: Optional[int] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "IncrementalUpdateConfig":
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    @staticmethod
    def ensure(cfg: Optional[Mapping[str, Any] | "IncrementalUpdateConfig"]) -> "IncrementalUpdateConfig":
        if cfg is None:
            return IncrementalUpdateConfig()
        if isinstance(cfg, IncrementalUpdateConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return IncrementalUpdateConfig.from_mapping(cfg)
        raise TypeError(f"无法识别的增量配置类型: {type(cfg)!r}")

    def resolved_strategy(self) -> str:
        return (self.strategy or "rolling").lower()

    def should_update(self, iteration: int) -> bool:
        if not self.enabled:
            return True
        step = int(self.step) if self.step is not None else 0
        if step <= 0:
            return True
        iteration = max(1, int(iteration))
        return iteration % step == 0

    def effective_window(self, fallback: Optional[int] = None) -> Optional[int]:
        if not self.enabled:
            return fallback
        return int(self.buffer_size) if self.buffer_size is not None else fallback

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
    target_bnd: Optional[float] = None,
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
    penalty_target = 0.0
    if target_bnd is not None and bnd_ratio < float(target_bnd):
        shortfall = float(target_bnd) - float(bnd_ratio)
        penalty_target = float(params.penalty_large) * shortfall
        fitness = float(fitness + penalty_target)
    detail = dict(detail)
    detail.update({
        "bnd_ratio": float(bnd_ratio),
        "fitness": float(fitness),
        "feasible": feasible,
        "penalty_infeasible": 0.0,
        "penalty_target": float(penalty_target),
        "bnd_target": float(target_bnd) if target_bnd is not None else None,
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
    target_bnd: Optional[float] = None,
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
        penalty_infeasible = 0.0 if feasible_raw else float(params.penalty_large)
        bnd_ratio = _calc_boundary_ratio(prob_levels_np, adj_alpha, adj_beta)
        penalty_target = 0.0
        if target_bnd is not None and bnd_ratio < float(target_bnd):
            shortfall = float(target_bnd) - float(bnd_ratio)
            penalty_target = float(params.penalty_large) * shortfall
        fitness_pen = float(fitness + penalty_infeasible + penalty_target)
        det_map = dict(detail)
        det_map.update({
            "penalty_infeasible": penalty_infeasible,
            "penalty_target": float(penalty_target),
            "fitness": fitness_pen,
            "feasible": bool(np.all(adj_alpha >= adj_beta + keep_gap)),
            "bnd_ratio": float(bnd_ratio),
            "bnd_target": float(target_bnd) if target_bnd is not None else None,
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
    penalty_target = 0.0
    if target_bnd is not None and bnd_ratio < float(target_bnd):
        shortfall = float(target_bnd) - float(bnd_ratio)
        penalty_target = float(params.penalty_large) * shortfall
        final_fit = float(final_fit + penalty_target)

    detail = dict(final_detail)
    detail.update({
        "bnd_ratio": float(bnd_ratio),
        "fitness": float(final_fit),
        "feasible": feasible,
        "penalty_infeasible": gbest_info.get("penalty_infeasible", 0.0),
        "penalty_target": float(penalty_target),
        "bnd_target": float(target_bnd) if target_bnd is not None else None,
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
            target_bnd=target_bnd,
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


def run_dynamic_thresholds(
    prob_levels: Sequence[np.ndarray],
    y: Iterable[int],
    params: S3WDParams,
    *,
    dynamic: Optional[Mapping[str, Any] | DynamicAdaptConfig] = None,
    incremental: Optional[Mapping[str, Any] | IncrementalUpdateConfig] = None,
    iteration: int = 1,
    pso_params: Optional["PSOParams"] = None,
    history: Optional[MutableMapping[str, List[np.ndarray]]] = None,
    gamma_last: float | None = None,
) -> Optional[ThresholdAdaptResult]:
    """根据 DYN/INCR 配置执行一次动态阈值调整。

    若当前迭代无需刷新阈值，则返回 ``None``。
    """

    dyn_cfg = DynamicAdaptConfig.ensure(dynamic)
    incr_cfg = IncrementalUpdateConfig.ensure(incremental)

    if not dyn_cfg.should_update(iteration):
        return None
    if not incr_cfg.should_update(iteration):
        return None

    strategy = dyn_cfg.resolved_strategy()
    keep_gap = dyn_cfg.keep_gap
    if strategy == "rule_based":
        result = adapt_thresholds_rule_based(
            prob_levels,
            y,
            params,
            keep_gap=keep_gap,
            history=history,
            ema_alpha=dyn_cfg.ema_alpha,
            median_window=dyn_cfg.median_window,
            gamma_last=gamma_last,
            target_bnd=dyn_cfg.target_bnd,
        )
    else:
        if pso_params is None:
            from .trainer import PSOParams  # 延迟导入避免循环依赖

            pso_params = PSOParams()
        dyn_cfg.apply_to_pso(pso_params)
        window_sz = pso_params.window_size
        if window_sz is None:
            window_sz = incr_cfg.effective_window(window_sz)
        result = adapt_thresholds_windowed_pso(
            prob_levels,
            y,
            params,
            particles=getattr(pso_params, "particles", 12),
            iters=getattr(pso_params, "iters", 30),
            seed=getattr(pso_params, "seed", None),
            keep_gap=pso_params.keep_gap if pso_params.keep_gap is not None else keep_gap,
            history=history,
            window_size=window_sz,
            ema_alpha=pso_params.ema_alpha,
            median_window=pso_params.median_window,
            gamma_last=gamma_last,
            stall_rounds=getattr(pso_params, "stall_rounds", dyn_cfg.stall_rounds),
            fallback_rule=pso_params.fallback_rule,
            target_bnd=dyn_cfg.target_bnd,
        )

    detail = dict(result.details)
    detail.setdefault("dynamic_step", dyn_cfg.step)
    detail.setdefault("dynamic_strategy", dyn_cfg.resolved_strategy())
    detail.setdefault("dynamic_target_bnd", dyn_cfg.target_bnd)
    if incr_cfg.enabled:
        detail.setdefault("incr_strategy", incr_cfg.resolved_strategy())
        detail.setdefault("incr_step", incr_cfg.step)
        detail.setdefault("incr_buffer_size", incr_cfg.buffer_size)
        detail.setdefault("incr_max_samples", incr_cfg.max_samples)

    return replace(result, details=detail)
