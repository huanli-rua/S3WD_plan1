# -*- coding: utf-8 -*-
"""阈值平滑与单窗步长限幅模块。"""
from __future__ import annotations

from typing import Optional, Tuple


def ema_clip(
    alpha_star: float,
    beta_star: float,
    prev_alpha: Optional[float],
    prev_beta: Optional[float],
    ema_alpha: float,
    step_cap_alpha: float,
    step_cap_beta: float,
) -> Tuple[float, float]:
    """返回平滑+限幅后的 (alpha_t, beta_t)。"""

    if prev_alpha is None or prev_beta is None:
        return float(alpha_star), float(beta_star)

    ema_alpha = float(max(0.0, min(1.0, ema_alpha)))
    alpha_ema = ema_alpha * alpha_star + (1 - ema_alpha) * prev_alpha
    beta_ema = ema_alpha * beta_star + (1 - ema_alpha) * prev_beta

    step_cap_alpha = float(max(1e-6, step_cap_alpha))
    step_cap_beta = float(max(1e-6, step_cap_beta))

    delta_alpha = alpha_ema - prev_alpha
    delta_beta = beta_ema - prev_beta

    delta_alpha = max(-step_cap_alpha, min(step_cap_alpha, delta_alpha))
    delta_beta = max(-step_cap_beta, min(step_cap_beta, delta_beta))

    alpha_new = prev_alpha + delta_alpha
    beta_new = prev_beta + delta_beta

    alpha_new = float(max(0.0, min(1.0, alpha_new)))
    beta_new = float(max(0.0, min(1.0, beta_new)))

    return alpha_new, beta_new
