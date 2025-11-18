from __future__ import annotations

"""Lightweight kNN-style posterior estimator used by the v3 pipeline."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _GWBConfig:
    k: int = 5
    metric: str = "euclidean"
    use_gpu: bool = False
    continuous_weight: float = 1.0
    categorical_weight: float = 1.0
    categorical_penalty: float = 0.5
    max_points_per_bucket: int = 50000
    rebuild_trigger_ratio: float = 0.5


class GWBProbEstimator:
    """Per-bucket probability estimator using a simple kNN backend."""

    def __init__(self, cfg_gwb: Optional[dict] = None) -> None:
        cfg_gwb = cfg_gwb or {}
        self.cfg = _GWBConfig(
            k=int(max(1, cfg_gwb.get("k", 5))),
            metric=cfg_gwb.get("metric", "euclidean"),
            use_gpu=bool(cfg_gwb.get("use_gpu", False)),
            continuous_weight=float(cfg_gwb.get("continuous_weight", 1.0)),
            categorical_weight=float(cfg_gwb.get("categorical_weight", 1.0)),
            categorical_penalty=float(cfg_gwb.get("categorical_penalty", 0.5)),
            max_points_per_bucket=int(max(1, cfg_gwb.get("max_points_per_bucket", 50000))),
            rebuild_trigger_ratio=float(cfg_gwb.get("rebuild_trigger_ratio", 0.5)),
        )
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.categorical_train: np.ndarray | None = None
        self._pending_updates: list[tuple[np.ndarray, int]] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_values: np.ndarray | None = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X 必须是二维矩阵。")
        if y.ndim != 1:
            raise ValueError("y 必须是一维数组。")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数量必须一致。")

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("训练样本数量为 0。")

        max_points = self.cfg.max_points_per_bucket
        if n_samples > max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_samples, size=max_points, replace=False)
            X = X[indices]
            y = y[indices]
            if categorical_values is not None:
                categorical_values = categorical_values[indices]

        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        if categorical_values is not None:
            if categorical_values.shape[0] != self.X_train.shape[0]:
                raise ValueError("categorical_values 与 X 的样本数不一致。")
            self.categorical_train = np.asarray(categorical_values, dtype=int)
        else:
            self.categorical_train = None
        self._pending_updates.clear()

    def _ensure_ready(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("GWBProbEstimator 尚未 fit。")

    def _pairwise_distance(
        self,
        X_query: np.ndarray,
        categorical_query: np.ndarray | None = None,
    ) -> np.ndarray:
        self._ensure_ready()
        assert self.X_train is not None
        diff = X_query[:, None, :] - self.X_train[None, :, :]
        if self.cfg.metric != "euclidean":
            raise ValueError(f"暂不支持的距离度量: {self.cfg.metric}")
        cont_dist = np.sqrt(np.sum(diff ** 2, axis=2)) * self.cfg.continuous_weight

        if (
            self.categorical_train is not None
            and categorical_query is not None
            and categorical_query.size > 0
            and self.cfg.categorical_weight > 0
        ):
            if categorical_query.shape[0] != X_query.shape[0]:
                raise ValueError("categorical_query 的样本数应与 X_query 一致。")
            mismatch = categorical_query[:, None, :] != self.categorical_train[None, :, :]
            cat_dist = mismatch.sum(axis=2) * self.cfg.categorical_penalty
            cont_dist = cont_dist + cat_dist * self.cfg.categorical_weight
        return cont_dist

    def predict_proba(
        self,
        X: np.ndarray,
        categorical_values: np.ndarray | None = None,
    ) -> np.ndarray:
        self._ensure_ready()
        if X.ndim != 2:
            raise ValueError("X 必须是二维矩阵。")
        if categorical_values is not None and categorical_values.shape[0] != X.shape[0]:
            raise ValueError("categorical_values 的样本数必须与 X 一致。")
        cat_vals = None
        if categorical_values is not None:
            cat_vals = np.asarray(categorical_values, dtype=int)
        distances = self._pairwise_distance(X, cat_vals)
        k = min(self.cfg.k, distances.shape[1])
        if k <= 0:
            raise ValueError("k 必须大于 0。")
        nearest_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        assert self.y_train is not None
        probs = self.y_train[nearest_idx].mean(axis=1)
        return probs

    def update_with_sample(self, x: np.ndarray, y: int) -> None:
        self._pending_updates.append((np.asarray(x, dtype=float), int(y)))
        if self.X_train is not None:
            trigger = int(self.cfg.rebuild_trigger_ratio * len(self.X_train))
            if trigger > 0 and len(self._pending_updates) >= trigger:
                # 标记为需要重建，实际重建逻辑后续阶段实现
                pass

    def rebuild_with_recent(self, X_recent: np.ndarray, y_recent: np.ndarray) -> None:
        self.fit(X_recent, y_recent)
