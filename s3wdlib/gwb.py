# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Optional, Sequence

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
    categorical_features: Sequence[str] | None = None
    category_penalty: float = 0.5

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
        self.category_penalty = float(np.clip(self.category_penalty, 0.0, 1.0))
        self._cat_tr: np.ndarray | None = None
        self._cat_cols: list[str] = []
        self._numeric_columns: list[str] | None = None
        self._n_features: int | None = None
        self._numeric_indices: np.ndarray | None = None

    @staticmethod
    def _build_numeric_matrix(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            raise ValueError("GWBProbEstimator 需要至少一个数值特征列用于距离计算。")
        if len(parts) == 1:
            return np.asarray(parts[0], dtype=float)
        return np.concatenate([np.asarray(p, dtype=float) for p in parts], axis=1)

    @staticmethod
    def _parse_year_month_label(value) -> float:
        if value is None:
            raise ValueError("空值无法转换为 year-month 浮点标签。")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        text = str(value).strip()
        if not text:
            raise ValueError("空字符串无法转换为 year-month 浮点标签。")
        match = re.fullmatch(r"(-?\d+)[-_/](\d{1,2})", text)
        if match is None:
            raise ValueError("字符串不符合 year-month 模式。")
        year = int(match.group(1))
        month = int(match.group(2))
        if month < 1 or month > 12:
            raise ValueError("月份应在 1 到 12 之间。")
        month_component = month / 100.0
        value = year + month_component if year >= 0 else year - month_component
        return float(f"{value:.2f}")

    @classmethod
    def _convert_object_column(cls, column: np.ndarray) -> tuple[bool, np.ndarray]:
        try:
            numeric = np.asarray(column, dtype=float)
            return True, numeric.reshape(-1, 1)
        except (TypeError, ValueError):
            pass
        try:
            parsed = np.array([cls._parse_year_month_label(v) for v in column], dtype=float)
            return True, parsed.reshape(-1, 1)
        except ValueError:
            return False, np.empty((column.shape[0], 0), dtype=float)

    def _categorical_mask(self, arr_obj: np.ndarray, categorical_values) -> np.ndarray:
        mask = np.zeros(arr_obj.shape[1], dtype=bool)
        if categorical_values is None:
            return mask
        cat_arr, _ = self._prepare_categorical(categorical_values)
        if cat_arr is None or cat_arr.size == 0:
            return mask
        arr_str = arr_obj.astype(str)
        cat_arr_str = cat_arr.astype(str)
        for j in range(cat_arr_str.shape[1]):
            cat_col = cat_arr_str[:, j]
            matches = np.all(arr_str == cat_col[:, None], axis=0)
            mask |= matches
        return mask

    def _select_numeric_object_columns(
        self, arr_obj: np.ndarray, categorical_values
    ) -> tuple[list[int], list[np.ndarray]]:
        mask = self._categorical_mask(arr_obj, categorical_values)
        numeric_indices: list[int] = []
        numeric_parts: list[np.ndarray] = []
        for idx in range(arr_obj.shape[1]):
            if mask[idx]:
                continue
            success, converted = self._convert_object_column(arr_obj[:, idx])
            if success:
                numeric_indices.append(idx)
                numeric_parts.append(converted)
        return numeric_indices, numeric_parts

    @staticmethod
    def _is_pandas_frame(obj) -> bool:
        """Best-effort detection for pandas-like DataFrame objects."""

        return hasattr(obj, "select_dtypes") and hasattr(obj, "columns")

    def _ensure_numeric_matrix(self, X, *, fit: bool, categorical_values=None) -> np.ndarray:
        """Extract the numeric feature matrix while filtering categorical columns."""

        if self._is_pandas_frame(X):
            df = X
            drop_cols = list(self.categorical_features or [])
            if drop_cols:
                drop_present = [col for col in drop_cols if col in df.columns]
                if drop_present:
                    df = df.drop(columns=drop_present)
            numeric_df = df.select_dtypes(include=["number", "bool"])
            if fit:
                self._numeric_columns = list(numeric_df.columns)
                self._numeric_indices = None
            elif self._numeric_columns is not None:
                missing = [col for col in self._numeric_columns if col not in numeric_df.columns]
                if missing:
                    raise ValueError(
                        "预测数据缺少在拟合阶段使用的数值特征列: " + ", ".join(missing)
                    )
                numeric_df = numeric_df[self._numeric_columns]
            if numeric_df.shape[1] == 0:
                raise ValueError("GWBProbEstimator 需要至少一个数值特征列用于距离计算。")
            arr = np.asarray(numeric_df.to_numpy(), float)
            if self._n_features is not None and not fit and arr.shape[1] != self._n_features:
                raise ValueError(
                    "数值特征维度与拟合阶段不一致：期望 %d，得到 %d"
                    % (self._n_features, arr.shape[1])
                )
            return arr

        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if fit:
            self._numeric_columns = None

        if arr.dtype.kind in ("O", "U", "S"):
            arr = self._extract_numeric_from_object_array(arr, categorical_values, fit)
        else:
            arr = np.asarray(arr, float)
            if fit:
                self._numeric_indices = np.arange(arr.shape[1], dtype=int)
            elif self._numeric_indices is not None and arr.shape[1] != len(self._numeric_indices):
                try:
                    arr = arr[:, self._numeric_indices]
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        "输入特征维度与拟合阶段不一致：期望 %d，得到 %d"
                        % (len(self._numeric_indices), arr.shape[1])
                    ) from exc

        if self._n_features is not None and not fit and arr.shape[1] != self._n_features:
            raise ValueError(
                "数值特征维度与拟合阶段不一致：期望 %d，得到 %d" % (self._n_features, arr.shape[1])
            )
        try:
            return np.asarray(arr, float)
        except ValueError as exc:
            raise ValueError(
                "GWBProbEstimator 仅支持数值特征，请在调用前移除或编码类别列。原始错误: %s"
                % exc
            ) from exc

    def _extract_numeric_from_object_array(self, arr, categorical_values, fit: bool) -> np.ndarray:
        arr_obj = np.asarray(arr, dtype=object)
        if arr_obj.ndim != 2:
            arr_obj = arr_obj.reshape(arr_obj.shape[0], -1)

        if fit:
            numeric_indices, numeric_parts = self._select_numeric_object_columns(arr_obj, categorical_values)
            if not numeric_indices:
                raise ValueError("在输入矩阵中未找到可转换的数值特征列，请检查输入数据。")
            self._numeric_indices = np.asarray(numeric_indices, dtype=int)
            return self._build_numeric_matrix(numeric_parts)

        if self._numeric_indices is None:
            numeric_indices, numeric_parts = self._select_numeric_object_columns(arr_obj, categorical_values)
            if not numeric_indices:
                raise ValueError("在输入矩阵中未找到可转换的数值特征列，请检查输入数据。")
            self._numeric_indices = np.asarray(numeric_indices, dtype=int)
            return self._build_numeric_matrix(numeric_parts)

        if np.any(self._numeric_indices >= arr_obj.shape[1]):
            raise ValueError(
                "输入特征维度与拟合阶段不一致：期望至少 %d 列，得到 %d"
                % (int(self._numeric_indices.max()) + 1, arr_obj.shape[1])
            )

        numeric_parts: list[np.ndarray] = []
        for idx in self._numeric_indices:
            success, converted = self._convert_object_column(arr_obj[:, idx])
            if not success:
                raise ValueError(
                    "检测到非数值特征列，请确认输入与拟合阶段一致（列索引=%d）。" % int(idx)
                )
            numeric_parts.append(converted)
        return self._build_numeric_matrix(numeric_parts)

    def _infer_numeric_indices(self, arr, categorical_values) -> np.ndarray:
        arr_obj = np.asarray(arr, dtype=object)
        numeric_indices, _ = self._select_numeric_object_columns(arr_obj, categorical_values)
        if not numeric_indices:
            raise ValueError("在输入矩阵中未找到可转换的数值特征列，请检查输入数据。")
        return np.asarray(numeric_indices, dtype=int)

    @staticmethod
    def _normalize_categories(values: np.ndarray) -> np.ndarray:
        normalized = np.empty(values.shape, dtype=object)
        for idx, val in np.ndenumerate(values):
            if val is None:
                normalized[idx] = "__MISSING__"
                continue
            try:
                if isinstance(val, (float, np.floating)) and math.isnan(float(val)):
                    normalized[idx] = "__MISSING__"
                    continue
            except TypeError:
                pass
            normalized[idx] = str(val)
        return normalized

    def _prepare_categorical(
        self,
        values,
        *,
        expected: Optional[Sequence[str]] = None,
    ) -> tuple[np.ndarray | None, list[str] | None]:
        if values is None:
            return None, None
        if hasattr(values, "to_numpy") and callable(getattr(values, "to_numpy")):
            columns = list(getattr(values, "columns", []))
            arr = values.to_numpy()
        else:
            columns = None
            arr = np.asarray(values, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        arr = np.asarray(arr, dtype=object)
        arr = self._normalize_categories(arr)
        if expected is not None:
            expected = list(expected)
            aligned_parts: list[np.ndarray] = []
            if columns is not None:
                name_to_idx = {name: idx for idx, name in enumerate(columns)}
                for name in expected:
                    idx = name_to_idx.get(name)
                    if idx is None:
                        aligned_parts.append(np.full((arr.shape[0], 1), "__MISSING__", dtype=object))
                    else:
                        aligned_parts.append(arr[:, idx : idx + 1])
                if aligned_parts:
                    arr = np.concatenate(aligned_parts, axis=1)
                else:
                    arr = np.full((arr.shape[0], len(expected)), "__MISSING__", dtype=object)
            else:
                if arr.shape[1] < len(expected):
                    pad = np.full((arr.shape[0], len(expected) - arr.shape[1]), "__MISSING__", dtype=object)
                    arr = np.concatenate([arr, pad], axis=1)
                elif arr.shape[1] > len(expected):
                    arr = arr[:, : len(expected)]
            columns = expected
        elif columns is None:
            columns = [f"cat_{i}" for i in range(arr.shape[1])]
        return arr, list(columns) if columns is not None else None

    def fit(self, X, y, categorical_values=None):
        self._numeric_columns = None
        self._n_features = None
        X = self._ensure_numeric_matrix(X, fit=True, categorical_values=categorical_values)
        y = np.asarray(y, int)
        n_samples = X.shape[0]
        self._n_features = X.shape[1]
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
        cat_arr, cat_cols = self._prepare_categorical(
            categorical_values, expected=self.categorical_features
        )
        if cat_arr is not None and cat_arr.size > 0:
            self._cat_tr = cat_arr
            self._cat_cols = cat_cols or list(self.categorical_features or [])
        else:
            self._cat_tr = None
            self._cat_cols = []
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

    def _apply_categorical_kernel(
        self,
        weights: np.ndarray,
        idx: np.ndarray,
        categorical_values,
    ) -> np.ndarray:
        if self._cat_tr is None or not self._cat_cols:
            return weights
        cat_query, _ = self._prepare_categorical(categorical_values, expected=self._cat_cols)
        if cat_query is None or cat_query.shape[1] != self._cat_tr.shape[1]:
            return weights
        if self.category_penalty >= 1.0:
            return weights
        penalty = float(max(self.category_penalty, 1e-3))
        nbr_cats = self._cat_tr[idx]
        mismatches = cat_query[:, None, :] != nbr_cats
        mismatch_counts = mismatches.sum(axis=2)
        reweight = np.power(penalty, mismatch_counts)
        return weights * reweight

    def predict_proba(self, X, categorical_values=None):
        if self.y_tr is None or self._k_effective is None:
            raise RuntimeError("Estimator must be fitted before calling predict_proba().")

        X = self._ensure_numeric_matrix(X, fit=False, categorical_values=categorical_values)
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
        if categorical_values is not None or self._cat_tr is not None:
            weights = self._apply_categorical_kernel(weights, idx, categorical_values)

        nbr_y = self.y_tr[idx]
        weights_pos = weights * (nbr_y == 1)
        weights_neg = weights * (nbr_y == 0)

        sum_pos = weights_pos.sum(axis=1)
        sum_neg = weights_neg.sum(axis=1)

        probs = (sum_pos + 1.0) / (sum_pos + sum_neg + 2.0)
        min_p = float(self.min_prob)
        probs = np.clip(probs, min_p, 1.0 - min_p)
        return probs
