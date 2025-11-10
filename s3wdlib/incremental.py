# -*- coding: utf-8 -*-
"""增量式后验估计器维护工具。"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from .drift import DriftEvent


_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


@dataclass
class PosteriorUpdater:
    """维护后验估计器的增量更新逻辑。"""

    estimator_factory: Callable[[], object]
    buffer_size: int = 4096
    cache_strategy: str = "sliding"  # "sliding" 或 "reservoir"
    rebuild_interval: int = 512
    min_rebuild_interval: int = 64
    drift_shrink: float = 0.5
    immediate_rebuild_methods: Iterable[str] = ("adwin",)
    enable_faiss_append: bool = True
    random_state: Optional[int] = None

    def __post_init__(self) -> None:  # noqa: D401 - dataclass hook
        if self.buffer_size <= 0:
            raise ValueError("buffer_size 必须为正整数。")
        if self.rebuild_interval <= 0:
            raise ValueError("rebuild_interval 必须为正整数。")
        if self.min_rebuild_interval <= 0:
            raise ValueError("min_rebuild_interval 必须为正整数。")
        if self.drift_shrink <= 0.0 or self.drift_shrink >= 1.0:
            raise ValueError("drift_shrink 应在 (0,1) 区间内。")
        strategy = self.cache_strategy.lower()
        if strategy not in {"sliding", "reservoir"}:
            raise ValueError("cache_strategy 仅支持 'sliding' 或 'reservoir'。")
        self.cache_strategy = strategy
        self._estimator: object | None = None
        self._buffer_X: np.ndarray | None = None
        self._buffer_y: np.ndarray | None = None
        self._buffer_cat: np.ndarray | None = None
        self._n_total: int = 0
        self._since_rebuild: int = 0
        self._base_rebuild_interval: int = int(self.rebuild_interval)
        self._current_rebuild_interval: int = int(self.rebuild_interval)
        self._pending_rebuild: bool = False
        self._rng = random.Random(self.random_state)
        self._reservoir_seen: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空缓存并重置内部状态。"""
        self._estimator = None
        self._buffer_X = None
        self._buffer_y = None
        self._buffer_cat = None
        self._n_total = 0
        self._since_rebuild = 0
        self._current_rebuild_interval = self._base_rebuild_interval
        self._pending_rebuild = False
        self._reservoir_seen = 0
        _logger.info("PosteriorUpdater 已重置，等待新批次样本。")

    # ------------------------------------------------------------------
    def current_estimator(self) -> object | None:
        """返回当前可用的后验估计器。"""
        return self._estimator

    # ------------------------------------------------------------------
    def update(
        self,
        X_batch: np.ndarray | Iterable[Iterable[float]],
        y_batch: np.ndarray | Iterable[int],
        drift_event: DriftEvent | None = None,
        categorical_batch: np.ndarray | Iterable[Iterable[object]] | None = None,
    ) -> object | None:
        """摄入一批新样本，并视情况执行增量或完整重建。"""

        X_arr = np.asarray(X_batch, dtype=float)
        y_arr = np.asarray(y_batch, dtype=int).ravel()
        if X_arr.size == 0:
            _logger.debug("收到空批次，忽略。")
            return self.current_estimator()
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X_batch 与 y_batch 的样本数不一致。")

        cat_arr: np.ndarray | None = None
        if categorical_batch is not None:
            cat_arr = np.asarray(categorical_batch, dtype=object)
            if cat_arr.ndim == 1:
                cat_arr = cat_arr.reshape(-1, 1)
            if cat_arr.shape[0] != y_arr.shape[0]:
                raise ValueError("categorical_batch 与标签数量不一致。")

        self._ingest_batch(X_arr, y_arr, cat_arr)
        if drift_event is not None:
            self._on_drift_event(drift_event)

        appended = False
        if not self._pending_rebuild and self._estimator is not None:
            appended = self._try_faiss_append(X_arr, y_arr, cat_arr)
            if appended:
                self._since_rebuild += X_arr.shape[0]
                _logger.info(
                    "【后验增量更新】批量=%d，FAISS 追加成功，距强制重建剩余 %d 条。",
                    X_arr.shape[0],
                    max(0, self._current_rebuild_interval - self._since_rebuild),
                )

        if (
            self._estimator is None
            or self._pending_rebuild
            or not appended
            or self._since_rebuild >= self._current_rebuild_interval
        ):
            self._rebuild_estimator()

        return self.current_estimator()

    # ------------------------------------------------------------------
    def _ingest_batch(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        cat_arr: np.ndarray | None,
    ) -> None:
        self._n_total += int(X_arr.shape[0])
        if self.cache_strategy == "sliding":
            self._ingest_sliding(X_arr, y_arr, cat_arr)
        else:
            self._ingest_reservoir(X_arr, y_arr, cat_arr)

    def _ingest_sliding(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        cat_arr: np.ndarray | None,
    ) -> None:
        """滑动窗口缓存：顺序追加，超限后从左侧丢弃；类别数据与样本行数严格对齐。"""
        if self._buffer_X is None:
            # 首批：直接建立缓存
            self._buffer_X = np.asarray(X_arr, dtype=float)
            self._buffer_y = np.asarray(y_arr, dtype=int)
            if cat_arr is not None:
                self._buffer_cat = np.asarray(cat_arr, dtype=object)
        else:
            # 追加数值与标签
            self._buffer_X = np.concatenate([self._buffer_X, X_arr], axis=0)
            self._buffer_y = np.concatenate([self._buffer_y, y_arr], axis=0)

            # === 关键修复：类别缓存与样本对齐 ===
            if cat_arr is not None:
                cat_np = np.asarray(cat_arr, dtype=object)
                if self._buffer_cat is None:
                    # 先无类→后有类：为“历史样本”左侧补占位 "__MISSING__"，再拼本批类别
                    n_old = self._buffer_y.shape[0] - y_arr.shape[0]  # 追加后总数 - 本批 = 历史条数
                    pad = np.full((max(0, n_old), cat_np.shape[1]), "__MISSING__", dtype=object)
                    self._buffer_cat = np.concatenate([pad, cat_np], axis=0)
                else:
                    # 历史已有类别 → 直接顺序追加
                    self._buffer_cat = np.concatenate([self._buffer_cat, cat_np], axis=0)
            elif self._buffer_cat is not None:
                # 本批没有类别，但历史已开启类别缓存 → 为本批样本补占位行
                fill = np.full((X_arr.shape[0], self._buffer_cat.shape[1]), "__MISSING__", dtype=object)
                self._buffer_cat = np.concatenate([self._buffer_cat, fill], axis=0)

        # 超限处理：左裁剪，并保持类别缓存同步裁剪
        overflow = self._buffer_y.shape[0] - self.buffer_size
        if overflow > 0:
            self._buffer_X = self._buffer_X[overflow:]
            self._buffer_y = self._buffer_y[overflow:]
            if self._buffer_cat is not None:
                self._buffer_cat = self._buffer_cat[overflow:]
            self._pending_rebuild = True
            _logger.info(
                "【滑动窗口】缓存超限，丢弃最早 %d 条样本，已触发重建标记。",
                overflow,
            )


        # ==== 修复：类别缓存与样本缓存对齐（先无类→后有类；已有类但本批无类） ====
        try:
            X_arr = locals().get('X_arr', None)
            y_arr = locals().get('y_arr', None)
            cat_arr = locals().get('cat_arr', None)
            if y_arr is not None:
                if cat_arr is not None:
                    import numpy as _np
                    cat_np = _np.asarray(cat_arr, dtype=object)
                    if self._buffer_cat is None:
                        n_old = self._buffer_y.shape[0] - y_arr.shape[0]
                        pad = _np.full((max(0, n_old), cat_np.shape[1]), '__MISSING__', dtype=object)
                        self._buffer_cat = _np.concatenate([pad, cat_np], axis=0)
                    else:
                        self._buffer_cat = _np.concatenate([self._buffer_cat, cat_np], axis=0)
                elif self._buffer_cat is not None and X_arr is not None:
                    import numpy as _np
                    fill = _np.full((X_arr.shape[0], self._buffer_cat.shape[1]), '__MISSING__', dtype=object)
                    self._buffer_cat = _np.concatenate([self._buffer_cat, fill], axis=0)
        except Exception as _e:
            if hasattr(self, '_logger'):
                self._logger.warning(f"[ingest_sliding] 类别对齐补丁触发异常: {_e}")
    def _ingest_reservoir(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        cat_arr: np.ndarray | None,
    ) -> None:
        replaced = False
        cat_np = np.asarray(cat_arr, dtype=object) if cat_arr is not None else None
        if self._buffer_X is None:
            keep = min(self.buffer_size, X_arr.shape[0])
            self._buffer_X = np.asarray(X_arr[:keep], dtype=float)
            self._buffer_y = np.asarray(y_arr[:keep], dtype=int)
            if cat_np is not None:
                self._buffer_cat = np.asarray(cat_np[:keep], dtype=object)
            self._reservoir_seen = keep
        else:
            for idx, (xi, yi) in enumerate(zip(X_arr, y_arr)):
                self._reservoir_seen += 1
                cat_row = cat_np[idx] if cat_np is not None else None
                if self._buffer_y.shape[0] < self.buffer_size:
                    self._buffer_X = np.vstack([self._buffer_X, xi])
                    self._buffer_y = np.concatenate([self._buffer_y, [yi]])
                    if cat_row is not None:
                        if self._buffer_cat is None:
                            self._buffer_cat = np.asarray(cat_row, dtype=object).reshape(1, -1)
                        else:
                            self._buffer_cat = np.concatenate(
                                [self._buffer_cat, np.asarray(cat_row, dtype=object).reshape(1, -1)],
                                axis=0,
                            )
                    elif self._buffer_cat is not None:
                        self._buffer_cat = np.concatenate(
                            [
                                self._buffer_cat,
                                np.full((1, self._buffer_cat.shape[1]), "__MISSING__", dtype=object),
                            ],
                            axis=0,
                        )
                    replaced = True
                    continue
                j = self._rng.randint(0, self._reservoir_seen - 1)
                if j < self.buffer_size:
                    self._buffer_X[j] = xi
                    self._buffer_y[j] = yi
                    if cat_row is not None:
                        if self._buffer_cat is None:
                            self._buffer_cat = np.full(
                                (self.buffer_size, np.asarray(cat_row, dtype=object).reshape(1, -1).shape[1]),
                                "__MISSING__",
                                dtype=object,
                            )
                        if self._buffer_cat.shape[0] < self.buffer_size:
                            pad_rows = self.buffer_size - self._buffer_cat.shape[0]
                            self._buffer_cat = np.concatenate(
                                [
                                    self._buffer_cat,
                                    np.full(
                                        (pad_rows, self._buffer_cat.shape[1]),
                                        "__MISSING__",
                                        dtype=object,
                                    ),
                                ],
                                axis=0,
                            )
                        self._buffer_cat[j] = np.asarray(cat_row, dtype=object)
                    elif self._buffer_cat is not None:
                        self._buffer_cat[j] = np.full(self._buffer_cat.shape[1], "__MISSING__", dtype=object)
                    replaced = True
        if replaced:
            self._pending_rebuild = True
            _logger.info(
                "【水库抽样】缓存更新完毕，标记重建以确保模型覆盖最新分布。",
            )

    # ------------------------------------------------------------------
    def _try_faiss_append(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        cat_arr: np.ndarray | None,
    ) -> bool:
        """在启用 FAISS 的情况下尝试低成本追加索引；保持 y_tr/_cat_tr 与样本数一致。"""
        if not self.enable_faiss_append:
            return False
        estimator = self._estimator
        if estimator is None:
            return False

        use_faiss = bool(getattr(estimator, "_use_faiss_runtime", False))
        index = getattr(estimator, "_faiss_index", None)
        if not use_faiss or index is None:
            return False

        try:
            # 1) 向量追加到 FAISS
            index.add(np.asarray(X_arr, dtype=np.float32))

            # 2) 同步 y_tr
            if hasattr(estimator, "y_tr") and estimator.y_tr is not None:
                estimator.y_tr = np.concatenate([np.asarray(estimator.y_tr), y_arr])
            elif hasattr(estimator, "y_tr"):
                estimator.y_tr = np.asarray(y_arr)

            # 3) === 关键修复：同步 _cat_tr，与 y_tr 等长 ===
            if hasattr(estimator, "_cat_tr"):
                if cat_arr is not None:
                    cat_np = np.asarray(cat_arr, dtype=object)
                    if estimator._cat_tr is None:
                        # 第一次有类别：为旧样本补占位，再拼本批类别
                        n_old = estimator.y_tr.shape[0] - y_arr.shape[0]
                        pad = np.full((max(0, n_old), cat_np.shape[1]), "__MISSING__", dtype=object)
                        estimator._cat_tr = np.concatenate([pad, cat_np], axis=0)
                    else:
                        estimator._cat_tr = np.concatenate([estimator._cat_tr, cat_np], axis=0)
                elif estimator._cat_tr is not None and y_arr is not None:
                    # 本批无类但历史已开启 → 为本批样本补占位
                    fill = np.full((y_arr.shape[0], estimator._cat_tr.shape[1]), "__MISSING__", dtype=object)
                    estimator._cat_tr = np.concatenate([estimator._cat_tr, fill], axis=0)

            # 4) 维护有效 k（修复被破坏的那一行）
            if hasattr(estimator, "_k_effective") and estimator._k_effective is not None:
                estimator._k_effective = int(
                    min(getattr(estimator, "k", estimator._k_effective), estimator.y_tr.shape[0])
                )

            return True

        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning("FAISS 追加失败（%s），将执行完整重建。", exc)
            self._pending_rebuild = True
            return False


    # ------------------------------------------------------------------
        # ==== 修复：FAISS 追加时的类别矩阵行数对齐 ====
        try:
            estimator = locals().get('estimator', None)
            y_arr = locals().get('y_arr', None)
            cat_arr = locals().get('cat_arr', None)
            if estimator is not None and hasattr(estimator, 'y_tr'):
                import numpy as _np
                if hasattr(estimator, '_cat_tr'):
                    if cat_arr is not None:
                        cat_np = _np.asarray(cat_arr, dtype=object)
                        if getattr(estimator, '_cat_tr', None) is None:
                            n_old = estimator.y_tr.shape[0] - (0 if y_arr is None else y_arr.shape[0])
                            pad = _np.full((max(0, n_old), cat_np.shape[1]), '__MISSING__', dtype=object)
                            estimator._cat_tr = _np.concatenate([pad, cat_np], axis=0)
                        else:
                            estimator._cat_tr = _np.concatenate([estimator._cat_tr, cat_np], axis=0)
                    elif getattr(estimator, '_cat_tr', None) is not None and y_arr is not None:
                        fill = _np.full((y_arr.shape[0], estimator._cat_tr.shape[1]), '__MISSING__', dtype=object)
                        estimator._cat_tr = _np.concatenate([estimator._cat_tr, fill], axis=0)
        except Exception as _e:
            if hasattr(self, '_logger'):
                self._logger.warning(f"[_try_faiss_append] 类别对齐补丁触发异常: {_e}")
    def _rebuild_estimator(self) -> None:
        """完整重建后验估计器；在 fit() 前做类别行数兜底对齐。"""
        if self._buffer_X is None or self._buffer_y is None:
            return

        estimator = self.estimator_factory()

        # === 关键修复：重建前的类别缓存行数兜底对齐 ===
        if self._buffer_cat is not None:
            n_y = self._buffer_y.shape[0]
            n_c = self._buffer_cat.shape[0]
            if n_c != n_y:
                if n_c < n_y:
                    need = n_y - n_c
                    cols = self._buffer_cat.shape[1]
                    pad = np.full((need, cols), "__MISSING__", dtype=object)
                    self._buffer_cat = np.concatenate([pad, self._buffer_cat], axis=0)
                else:
                    # 极少数情况下类别条数更多，截断到样本数
                    self._buffer_cat = self._buffer_cat[-n_y:]

        # 按是否有类别调用 fit
        if self._buffer_cat is not None:
            estimator.fit(self._buffer_X, self._buffer_y, categorical_values=self._buffer_cat)
        else:
            estimator.fit(self._buffer_X, self._buffer_y)

        self._estimator = estimator
        self._since_rebuild = 0
        self._pending_rebuild = False
        _logger.info(
            "【后验重建】缓存样本=%d，策略=%s，FAISS=%s。",
            int(self._buffer_y.shape[0] if self._buffer_y is not None else 0),
            self.cache_strategy,
            "是" if getattr(estimator, "_use_faiss_runtime", False) else "否",
        )

    # ------------------------------------------------------------------
    def _on_drift_event(self, event: DriftEvent) -> None:
        method = (event.method or "").lower()
        if method in {m.lower() for m in self.immediate_rebuild_methods}:
            self._pending_rebuild = True
            self._current_rebuild_interval = self.min_rebuild_interval
            _logger.info(
                "【漂移告警】检测到 %s 漂移，立即触发重建并将步长压缩至 %d。",
                method.upper(),
                self._current_rebuild_interval,
            )
            return
        new_interval = max(
            self.min_rebuild_interval,
            int(self._current_rebuild_interval * self.drift_shrink),
        )
        if new_interval < self._current_rebuild_interval:
            self._current_rebuild_interval = new_interval
            _logger.info(
                "【漂移告警】检测到 %s 漂移，重建步长缩短至 %d。",
                method.upper(),
                self._current_rebuild_interval,
            )


def latest_estimator_for_flow(updater: PosteriorUpdater) -> object | None:
    """动态流程入口的便捷函数，返回 `PosteriorUpdater` 当前模型。"""

    return updater.current_estimator()
