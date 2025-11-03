# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors


try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
    try:
        _FAISS_GPU_AVAILABLE = getattr(faiss, "get_num_gpus", lambda: 0)() > 0
    except Exception:  # pragma: no cover - defensive
        _FAISS_GPU_AVAILABLE = False
except ImportError:  # pragma: no cover - optional dependency missing
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    _FAISS_GPU_AVAILABLE = False


def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (u ** 2))


def _uniform_kernel(u: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(u) <= 1.0)


def _epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    mask = np.abs(u) <= 1.0
    out = np.zeros_like(u, dtype=float)
    out[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return out


def _triangular_kernel(u: np.ndarray) -> np.ndarray:
    mask = np.abs(u) <= 1.0
    out = np.zeros_like(u, dtype=float)
    out[mask] = 1.0 - np.abs(u[mask])
    return out


def _exponential_kernel(u: np.ndarray) -> np.ndarray:
    # u >= 0 for distances; clip negatives introduced by numerical noise
    u = np.maximum(u, 0.0)
    return np.exp(-u)


_KERNELS = {
    "gaussian": _gaussian_kernel,
    "uniform": _uniform_kernel,
    "epanechnikov": _epanechnikov_kernel,
    "triangular": _triangular_kernel,
    "exponential": _exponential_kernel,
}


@dataclass
class GWBProbEstimator:
    """Kernel/Gaussian weighted Bayesian posterior estimator."""

    k: int = 6
    metric: str = "euclidean"
    eps: float = 1e-6
    min_prob: float = 0.01
    kernel: str | None = None
    mode: str = "epanechnikov"
    bandwidth: float | None = None
    bandwidth_scale: float = 1.0
    use_faiss: bool = True
    faiss_gpu: bool = True

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.kernel is not None:
            if self.mode and self.kernel != self.mode:
                raise ValueError("kernel 与 mode 参数冲突，请仅保留一个。")
            self.mode = self.kernel
        if self.mode not in _KERNELS:
            raise ValueError(f"Unknown kernel/mode '{self.mode}'. Available: {sorted(_KERNELS)}")
        self._kernel_fn: Callable[[np.ndarray], np.ndarray] = _KERNELS[self.mode]
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive if specified")
        self.nn: NearestNeighbors | None = None
        self.y_tr: np.ndarray | None = None
        self._faiss_index = None
        self._faiss_res = None
        self._use_faiss_runtime = False
        self._k_effective = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        n_samples = X.shape[0]
        self._k_effective = int(min(self.k, max(1, n_samples)))
        self.nn = None
        self._faiss_index = None
        self._faiss_res = None
        self._use_faiss_runtime = False

        if self.use_faiss:
            if self.metric != "euclidean":
                print("⚠️ GWB 仅在欧式距离下支持 FAISS，已回退到 sklearn.neighbors。")
            elif not _FAISS_AVAILABLE:
                print("⚠️ 当前环境缺少 FAISS，已回退到 sklearn.neighbors 的近邻检索。")
            else:
                try:
                    X32 = np.asarray(X, np.float32)
                    dim = X32.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    if self.faiss_gpu:
                        if _FAISS_GPU_AVAILABLE:
                            res = faiss.StandardGpuResources()
                            index = faiss.index_cpu_to_gpu(res, 0, index)
                            self._faiss_res = res
                        else:
                            print("⚠️ 当前环境未检测到可用 GPU，FAISS 将使用 CPU。")
                    index.add(X32)
                    self._faiss_index = index
                    self._use_faiss_runtime = True
                except Exception as exc:  # pragma: no cover - defensive fallback
                    print(f"⚠️ FAISS 初始化失败（{exc}），已回退到 sklearn.neighbors。")
                    self._faiss_index = None
                    self._faiss_res = None
                    self._use_faiss_runtime = False

        if not self._use_faiss_runtime:
            self.nn = NearestNeighbors(n_neighbors=self._k_effective, metric=self.metric)
            self.nn.fit(X)

        self.y_tr = y
        return self

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        if self.bandwidth is not None:
            h = np.full((distances.shape[0], 1), float(self.bandwidth), dtype=float)
        else:
            h_base = distances[:, [-1]]
            h = np.maximum(h_base * self.bandwidth_scale, self.eps)
        # Avoid zero bandwidth; ensure broadcast shape
        scaled = distances / h
        weights = self._kernel_fn(scaled)
        # In case of all zeros weights (e.g., outside kernel support)
        zero_mask = weights.sum(axis=1, keepdims=True) == 0
        if np.any(zero_mask):
            weights[zero_mask[:, 0]] = 1.0
        return weights

    def predict_proba(self, X):
        if self.y_tr is None or self._k_effective is None:
            raise RuntimeError("Estimator must be fitted before calling predict_proba().")

        X = np.asarray(X, float)
        if self._use_faiss_runtime:
            if self._faiss_index is None:
                raise RuntimeError("FAISS 索引未正确初始化。")
            query = np.asarray(X, np.float32)
            distances2, idx = self._faiss_index.search(query, self._k_effective)
            distances = np.sqrt(np.maximum(distances2, 0.0))
        else:
            if self.nn is None:
                raise RuntimeError("FAISS 初始化失败且未建立 sklearn 近邻模型。")
            distances, idx = self.nn.kneighbors(X, n_neighbors=self._k_effective, return_distance=True)
        weights = self._compute_weights(distances)

        nbr_y = self.y_tr[idx]
        weights_pos = weights * (nbr_y == 1)
        weights_neg = weights * (nbr_y == 0)

        sum_pos = weights_pos.sum(axis=1)
        sum_neg = weights_neg.sum(axis=1)

        probs = (sum_pos + 1.0) / (sum_pos + sum_neg + 2.0)
        min_p = float(self.min_prob)
        probs = np.clip(probs, min_p, 1.0 - min_p)
        return probs
