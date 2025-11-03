# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import s3wd_objective, S3WDParams

try:  # pragma: no cover - optional dependency
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency missing
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

@dataclass
class PSOParams:
    particles: int = 20
    iters: int = 60
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 2.8
    c2: float = 1.3
    seed: int = 42
    use_gpu: bool = False

def _encode_init(nL, rng):
    alphas = rng.uniform(0.55, 0.95, size=nL)
    betas  = rng.uniform(0.05, 0.45, size=nL)
    return np.concatenate([alphas, betas])

def _apply_chain(alphas, betas):
    a = np.array(alphas, float)
    b = np.array(betas, float)
    for i in range(1, len(a)):
        if a[i] > a[i-1]:
            a[i] = a[i-1]
    for i in range(1, len(b)):
        if b[i] < b[i-1]:
            b[i] = b[i-1]
    return a, b

def pso_learn_thresholds(prob_levels, y, params: S3WDParams, pso: PSOParams):
    use_gpu = bool(getattr(pso, "use_gpu", False))
    if use_gpu and not _CUPY_AVAILABLE:
        print("⚠️ 当前环境缺少 CuPy，PSO 将在 CPU 上运行。")
        use_gpu = False

    xp = cp if use_gpu else np  # type: ignore[assignment]
    rng = (cp.random.RandomState(pso.seed) if use_gpu else np.random.RandomState(pso.seed))  # type: ignore[operator]

    nL = len(prob_levels)
    dim = 2 * nL  # γ 固定为 0.5

    if use_gpu:
        alpha_init = rng.uniform(0.55, 0.95, size=(pso.particles, nL))
        beta_init = rng.uniform(0.05, 0.45, size=(pso.particles, nL))
        X = xp.concatenate([alpha_init, beta_init], axis=1)
        V = rng.uniform(-0.2, 0.2, size=(pso.particles, dim))
    else:
        X = np.vstack([_encode_init(nL, rng) for _ in range(pso.particles)])
        V = rng.uniform(-0.2, 0.2, size=(pso.particles, dim))
    if use_gpu:
        V = xp.asarray(V)

    def _to_numpy(arr):
        if use_gpu:
            return cp.asnumpy(arr)  # type: ignore[attr-defined]
        return np.asarray(arr)

    def fitness_of(vec):
        vec_np = _to_numpy(vec)
        vec_np = np.clip(vec_np, 0.0, 1.0)
        a = vec_np[:nL]
        b = vec_np[nL:2 * nL]
        a, b = _apply_chain(a, b)
        f, details = s3wd_objective(prob_levels, y, a, b, 0.5, params)
        return f, details, a, b

    pbest = X.copy()
    pbest_fit = np.zeros(pso.particles, dtype=float)
    pbest_det = [None]*pso.particles
    for i in range(pso.particles):
        f, det, a, b = fitness_of(X[i])
        pbest_fit[i] = f
        pbest_det[i] = det
    g_idx = int(np.argmin(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = pbest_fit[g_idx]
    gbest_det = pbest_det[g_idx]

    for t in range(pso.iters):
        w = pso.w_max - (pso.w_max - pso.w_min) * (t / max(1,(pso.iters-1)))
        r1 = rng.rand(pso.particles, dim)
        r2 = rng.rand(pso.particles, dim)
        V = w * V + pso.c1 * r1 * (pbest - X) + pso.c2 * r2 * (gbest - X)
        X = xp.clip(X + V, 0.0, 1.0)
        for i in range(pso.particles):
            f, det, a, b = fitness_of(X[i])
            if f < pbest_fit[i]:
                pbest[i] = X[i].copy()
                pbest_fit[i] = f
                pbest_det[i] = det
        g_idx = int(np.argmin(pbest_fit))
        if pbest_fit[g_idx] < gbest_fit:
            gbest = pbest[g_idx].copy()
            gbest_fit = pbest_fit[g_idx]
            gbest_det = pbest_det[g_idx]

    gbest_np = _to_numpy(gbest)
    a = gbest_np[:nL]
    b = gbest_np[nL:2 * nL]
    a, b = _apply_chain(a, b)
    gamma = 0.5
    return (a, b, gamma), float(gbest_fit), (gbest_det or {})
