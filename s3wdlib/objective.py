# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, Any

@dataclass
class S3WDParams:
    # 信息增益权重
    c1: float = 0.37
    c2: float = 0.63
    # 中间层最小边界占比
    xi_min: float = 0.10
    # 风险偏好 θ（正/负类）
    theta_pos: float = 2.0
    theta_neg: float = 1.0
    # —— 统一命名（来自 YAML 扁平键）——
    # S3_sigma: 后悔厌恶系数
    sigma: float = 3.0
    # S3_regret_mode: "utility" 或 "rate"
    regret_mode: str = "utility"
    # 约束惩罚
    penalty_large: float = 1e6
    # 最后一层 γ（True→0.5；或直接给 float）
    gamma_last: bool | float = True
    # α_i ≥ β_i + gap
    gap: float = 0.02

def _entropy_w(y: np.ndarray, w: np.ndarray) -> float:
    """加权二分类熵 H_w(Y)。"""
    wsum = float(np.sum(w))
    if wsum <= 0:
        return 0.0
    p1 = float(np.sum(w[y == 1])) / wsum
    p0 = 1.0 - p1
    def h(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return - p*np.log2(p) - (1-p)*np.log2(1-p)
    return h(p1)

def _info_gain_threeway(prob: np.ndarray, y: np.ndarray, a: float, b: float, c1: float, c2: float):
    """三支切分的信息增益 IG 与三域掩码。"""
    pos_m = prob >= a
    neg_m = prob <= b
    bnd_m = (~pos_m) & (~neg_m)
    w = np.where(y > 0, c1, c2)
    H_parent = _entropy_w(y, w)
    IG = 0.0
    for m in (pos_m, bnd_m, neg_m):
        if not np.any(m):
            continue
        IG += (float(np.sum(w[m]))/float(np.sum(w))) * _entropy_w(y[m], w[m])
    IG = H_parent - IG
    return IG, pos_m, bnd_m, neg_m

def _utility_pos(h: np.ndarray, theta: float) -> np.ndarray:
    """U_P(h) = (1 - exp(-theta * max(h,0))) ** theta"""
    z = np.clip(h, 0.0, None)
    return np.power(1.0 - np.exp(-theta * z), theta)

def _utility_neg(h: np.ndarray, theta: float) -> np.ndarray:
    """U_N(h) = (1 - exp(-theta * max(-h,0))) ** theta"""
    z = np.clip(-h, 0.0, None)
    return np.power(1.0 - np.exp(-theta * z), theta)

def _regret_transform(x: np.ndarray, sigma: float) -> np.ndarray:
    """R = 1 - exp(-sigma * x), x ≥ 0"""
    x = np.clip(x, 0.0, None)
    return 1.0 - np.exp(-sigma * x)

def _regret_rate_mode(prob: np.ndarray, y: np.ndarray, pos_m: np.ndarray, neg_m: np.ndarray,
                      theta_pos: float, theta_neg: float, sigma: float) -> float:
    """比例型（简化）后悔 + σ 变换。"""
    n_pos = max(1, int(np.sum(pos_m)))
    fp_rate = float(np.sum((y == 0) & pos_m)) / n_pos
    n_neg = max(1, int(np.sum(neg_m)))
    fn_rate = float(np.sum((y == 1) & neg_m)) / n_neg
    r_pos = theta_pos * float(_regret_transform(np.array([fp_rate]), sigma)[0])
    r_neg = theta_neg * float(_regret_transform(np.array([fn_rate]), sigma)[0])
    return r_pos + r_neg

def _regret_utility_mode(prob: np.ndarray, y: np.ndarray, pos_m: np.ndarray, neg_m: np.ndarray,
                         theta_pos: float, theta_neg: float, sigma: float) -> float:
    """
    效用型后悔（regret–rejoice）：
      h = 2p-1 ∈ [-1,1]; U_P, U_N 为指数效用；
      FP：1 - exp(-σ * (U_N - U_P)_+)，FN：1 - exp(-σ * (U_P - U_N)_+)。
    """
    h = 2.0*prob - 1.0
    U_p = _utility_pos(h, theta_pos)
    U_n = _utility_neg(h, theta_neg)

    mask_fp = pos_m & (y == 0)
    if np.any(mask_fp):
        u_diff_fp = np.maximum(U_n[mask_fp] - U_p[mask_fp], 0.0)
        r_fp = np.mean(_regret_transform(u_diff_fp, sigma))
    else:
        r_fp = 0.0

    mask_fn = neg_m & (y == 1)
    if np.any(mask_fn):
        u_diff_fn = np.maximum(U_p[mask_fn] - U_n[mask_fn], 0.0)
        r_fn = np.mean(_regret_transform(u_diff_fn, sigma))
    else:
        r_fn = 0.0

    return theta_pos * r_fp + theta_neg * r_fn

def regret_layer(prob: np.ndarray, y: np.ndarray, pos_m: np.ndarray, neg_m: np.ndarray,
                 theta_pos: float, theta_neg: float, sigma: float, mode: str) -> float:
    if mode == "utility":
        return _regret_utility_mode(prob, y, pos_m, neg_m, theta_pos, theta_neg, sigma)
    return _regret_rate_mode(prob, y, pos_m, neg_m, theta_pos, theta_neg, sigma)

def thresholds_penalty_monotonic(alphas: np.ndarray, betas: np.ndarray,
                                 gap: float, penalty_large: float) -> float:
    """单层单调约束：α_i ≥ β_i + gap。"""
    pen = 0.0
    for ai, bi in zip(alphas, betas):
        if ai < bi + gap:
            pen += penalty_large
    return pen

def s3wd_objective(prob_levels: Iterable[np.ndarray], y: np.ndarray,
                   alphas: Iterable[float], betas: Iterable[float],
                   gamma_last: float | None, params: S3WDParams) -> Tuple[float, Dict[str, Any]]:
    """
    f = -(Σ IG_l) + Σ Regret_l + boundary_shortfall_penalty + monotonic_penalty
    """
    prob_levels = [np.asarray(p).astype(float).ravel() for p in prob_levels]
    y = (np.asarray(y) > 0).astype(int)
    alphas = np.asarray(alphas, dtype=float).ravel()
    betas  = np.asarray(betas,  dtype=float).ravel()
    nL = len(prob_levels)
    assert nL == len(alphas) == len(betas), "levels 与阈值维度不一致"

    total_ig = 0.0
    total_regret = 0.0
    bnd_shortfall_pen = 0.0

    for li in range(nL - 1):
        IG_l, pos_m, bnd_m, neg_m = _info_gain_threeway(
            prob_levels[li], y, alphas[li], betas[li], params.c1, params.c2
        )
        total_ig += float(IG_l)

        total_regret += regret_layer(
            prob_levels[li], y, pos_m, neg_m,
            params.theta_pos, params.theta_neg, params.sigma, params.regret_mode
        )

        denom = max(1, int(np.sum(pos_m | bnd_m | neg_m)))
        bnd_ratio = float(np.sum(bnd_m)) / denom
        if bnd_ratio < float(params.xi_min):
            bnd_shortfall_pen += float(params.penalty_large)

    # 最后一层 γ
    if params.gamma_last is True:
        g = 0.5
    elif isinstance(params.gamma_last, (float, int)):
        g = float(params.gamma_last)
    else:
        g = float(gamma_last) if gamma_last is not None else 0.5

    prob_n = prob_levels[-1]
    pos_m = (prob_n >= g)
    neg_m = ~pos_m

    total_regret += regret_layer(
        prob_n, y, pos_m, neg_m, params.theta_pos, params.theta_neg, params.sigma, params.regret_mode
    )

    pen_mono = thresholds_penalty_monotonic(alphas, betas, float(params.gap), float(params.penalty_large))
    f = (-total_ig) + total_regret + bnd_shortfall_pen + pen_mono
    details = {
        "ig": float(total_ig),
        "regret": float(total_regret),
        "pen_bnd": float(bnd_shortfall_pen),
        "pen_mono": float(pen_mono),
    }
    return float(f), details
