from __future__ import annotations

"""Lightweight label encoders for v3 preprocessing."""

from typing import Iterable

import numpy as np
import pandas as pd


class SimpleLabelEncoder:
    """Map arbitrary categorical values to consecutive integer ids."""

    def __init__(self) -> None:
        self.mapping: dict[str, int] = {}
        self.id_unknown: int | None = None
        self._fitted: bool = False

    @staticmethod
    def _normalize(values: Iterable) -> pd.Series:
        series = pd.Series(values)
        if series.empty:
            return pd.Series(dtype="object")
        series = series.fillna("__MISSING__").astype(str)
        return series

    def fit(self, values: Iterable) -> "SimpleLabelEncoder":
        series = self._normalize(values)
        for val in series.unique():
            if val not in self.mapping:
                self.mapping[val] = len(self.mapping)
        self.id_unknown = len(self.mapping)
        self._fitted = True
        return self

    def transform(self, values: Iterable) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SimpleLabelEncoder must be fitted before calling transform().")
        series = self._normalize(values)
        encoded = series.map(self.mapping)
        if encoded.isna().any():
            encoded = encoded.fillna(self.id_unknown)
        return encoded.to_numpy(dtype=int)

    def fit_transform(self, values: Iterable) -> np.ndarray:
        return self.fit(values).transform(values)
