# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class KWBProbEstimator:
    k: int = 6
    metric: str = "euclidean"
    eps: float = 1e-6
    min_prob: float = 0.01
    use_faiss: bool = True
    faiss_gpu: bool = True

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be positive")
        self.nn: NearestNeighbors | None = None
        self.y_tr: np.ndarray | None = None
        self._faiss_index = None
        self._faiss_res = None
        self._use_faiss_runtime = False
        self._k_effective: int | None = None

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
                print("⚠️ KWB 仅在欧式距离下支持 FAISS，已回退到 sklearn.neighbors。")
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

    def predict_proba(self, X):
        if self.y_tr is None or self._k_effective is None:
            raise RuntimeError("Estimator must be fitted before calling predict_proba().")

        X = np.asarray(X, float)
        if self._use_faiss_runtime:
            if self._faiss_index is None:
                raise RuntimeError("FAISS 索引未正确初始化。")
            query = np.asarray(X, np.float32)
            _, idx = self._faiss_index.search(query, self._k_effective)
            # 距离用于判断带宽，KWB 实际只用索引
        else:
            if self.nn is None:
                raise RuntimeError("FAISS 初始化失败且未建立 sklearn 近邻模型。")
            _, idx = self.nn.kneighbors(X, n_neighbors=self._k_effective, return_distance=True)

        probs = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            nbr_y = self.y_tr[idx[i]]
            pos = int((nbr_y == 1).sum())
            neg = int((nbr_y == 0).sum())
            p_hat = (pos + 1) / (pos + neg + 2)  # Laplace smoothing
            p_hat = max(self.min_prob, min(1.0 - self.min_prob, p_hat))
            probs[i] = p_hat
        return probs
