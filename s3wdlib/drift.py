"""Utilities for streaming concept drift detection."""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Any

_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


@dataclass
class DriftConfig:
    """配置包装器（对应 YAML 中的 DRIFT 分组）。"""

    enabled: bool = False
    strategy: str = "kswin"
    window_size: int = 200
    stat_size: int = 60
    significance: float = 0.05
    delta: float = 0.002
    cooldown: int = 0
    min_window_length: int = 5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DriftConfig":
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)

    @staticmethod
    def ensure(cfg: Optional[Mapping[str, Any] | "DriftConfig"]) -> "DriftConfig":
        if cfg is None:
            return DriftConfig()
        if isinstance(cfg, DriftConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return DriftConfig.from_mapping(cfg)
        raise TypeError(f"无法识别的漂移检测配置类型: {type(cfg)!r}")

    def resolved_strategy(self) -> str:
        name = (self.strategy or "kswin").lower()
        if name in {"kswin", "adwin"}:
            return name
        raise ValueError(f"Unsupported drift strategy: {self.strategy}")

    def build_detector(self) -> Optional["DriftDetector"]:
        if not self.enabled:
            return None
        return DriftDetector(
            method=self.resolved_strategy(),
            window_size=self.window_size,
            stat_size=self.stat_size,
            significance=self.significance,
            delta=self.delta,
            cooldown=self.cooldown,
            min_window_length=self.min_window_length,
        )

    def apply(self, detector: "DriftDetector") -> "DriftDetector":
        detector.configure(
            method=self.resolved_strategy(),
            window_size=self.window_size,
            stat_size=self.stat_size,
            significance=self.significance,
            delta=self.delta,
            cooldown=self.cooldown,
            min_window_length=self.min_window_length,
        )
        return detector


@dataclass
class DriftEvent:
    """Structured event emitted when a detector triggers."""

    index: int
    value: float
    method: str
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Return a serialisable representation of the event."""
        payload = {
            "index": self.index,
            "value": self.value,
            "method": self.method,
        }
        payload.update(self.details)
        return payload

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return True


class DriftDetector:
    """Streaming drift detector supporting KSWIN and ADWIN strategies."""

    def __init__(
        self,
        method: str = "kswin",
        window_size: int = 100,
        stat_size: int = 30,
        significance: float = 0.05,
        delta: float = 0.002,
        cooldown: int = 0,
        min_window_length: int = 5,
    ) -> None:
        self.method = method.lower()
        self.window_size = int(window_size)
        self.stat_size = int(stat_size)
        self.significance = float(significance)
        self.delta = float(delta)
        self.cooldown = max(int(cooldown), 0)
        self.min_window_length = max(int(min_window_length), 2)
        if self.method not in {"kswin", "adwin"}:
            raise ValueError(f"Unsupported method: {method}")
        self.reset()

    def reset(self) -> None:
        """Reset the internal state of the detector."""
        self._window: Deque[float] = deque(maxlen=self.window_size)
        self._adwin_window: List[float] = []
        self._n_samples = 0
        self._last_drift_index: Optional[int] = None

    def configure(self, **kwargs: float) -> None:
        """Update detector parameters on the fly."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        if "window_size" in kwargs:
            self.window_size = int(self.window_size)
            self._window = deque(self._window, maxlen=self.window_size)
            if len(self._adwin_window) > self.window_size:
                self._adwin_window = self._adwin_window[-self.window_size :]
        if "stat_size" in kwargs:
            self.stat_size = int(self.stat_size)
        if "cooldown" in kwargs:
            self.cooldown = max(int(self.cooldown), 0)
        if "min_window_length" in kwargs:
            self.min_window_length = max(int(self.min_window_length), 2)
        if "significance" in kwargs:
            self.significance = float(self.significance)
        if "delta" in kwargs:
            self.delta = float(self.delta)
        if "method" in kwargs:
            self.method = str(self.method).lower()
            if self.method not in {"kswin", "adwin"}:
                raise ValueError(f"Unsupported method: {self.method}")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def update(self, value: float, index: Optional[int] = None) -> Optional[DriftEvent]:
        """Consume a value and return a drift event when triggered."""
        self._n_samples += 1
        if index is None:
            index = self._n_samples
        if self._last_drift_index is not None and index - self._last_drift_index <= self.cooldown:
            self._append_value(value)
            return None
        if self.method == "kswin":
            return self._update_kswin(value, index)
        if self.method == "adwin":
            return self._update_adwin(value, index)
        raise RuntimeError("Unknown method encountered during update")

    # --- internal helpers -------------------------------------------------
    def _append_value(self, value: float) -> None:
        self._window.append(float(value))
        self._adwin_window.append(float(value))
        if len(self._adwin_window) > self.window_size:
            self._adwin_window.pop(0)

    def _update_kswin(self, value: float, index: int) -> Optional[DriftEvent]:
        self._append_value(value)
        if len(self._window) < max(self.stat_size * 2, self.window_size // 2):
            return None
        reference_size = len(self._window) - self.stat_size
        if reference_size <= 0:
            return None
        reference = list(self._window)[:reference_size]
        recent = list(self._window)[-self.stat_size :]
        d_stat = self._ks_statistic(reference, recent)
        p_value = self._ks_pvalue(d_stat, len(reference), len(recent))
        if p_value < self.significance:
            self._last_drift_index = index
            self._window = deque(recent, maxlen=self.window_size)
            self._adwin_window = list(recent)
            details = {"ks_stat": d_stat, "p_value": p_value}
            _logger.info("第 %d 个样本触发 KSWIN 漂移 (p=%.4f)", index, p_value)
            return DriftEvent(index=index, value=value, method="kswin", details=details)
        return None

    def _update_adwin(self, value: float, index: int) -> Optional[DriftEvent]:
        self._append_value(value)
        window = self._adwin_window
        if len(window) < 2 * self.min_window_length:
            return None
        total_length = len(window)
        best_result: Optional[Dict[str, float]] = None
        prefix_sums = self._prefix_sums(window)
        for cut in range(self.min_window_length, total_length - self.min_window_length + 1):
            left_mean = (prefix_sums[cut] - prefix_sums[0]) / cut
            right_count = total_length - cut
            right_mean = (prefix_sums[total_length] - prefix_sums[cut]) / right_count
            diff = abs(left_mean - right_mean)
            epsilon = self._adwin_epsilon(cut, right_count)
            if diff > epsilon:
                if best_result is None or diff - epsilon > best_result["margin"]:
                    best_result = {
                        "cut": cut,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                        "epsilon": epsilon,
                        "margin": diff - epsilon,
                    }
        if best_result is not None:
            cut = int(best_result["cut"])
            self._last_drift_index = index
            self._adwin_window = window[cut:]
            self._window = deque(self._adwin_window[-self.window_size :], maxlen=self.window_size)
            details = {
                "cut_point": float(cut),
                "left_mean": best_result["left_mean"],
                "right_mean": best_result["right_mean"],
                "epsilon": best_result["epsilon"],
            }
            _logger.info("第 %d 个样本触发 ADWIN 漂移 (均值差 %.4f)", index, best_result["margin"])
            return DriftEvent(index=index, value=value, method="adwin", details=details)
        return None

    def _adwin_epsilon(self, n0: int, n1: int) -> float:
        delta = max(self.delta, 1e-10)
        m = 1 / n0 + 1 / n1
        return math.sqrt(0.5 * m * math.log(2 / delta))

    @staticmethod
    def _prefix_sums(values: Iterable[float]) -> List[float]:
        prefix = [0.0]
        total = 0.0
        for item in values:
            total += item
            prefix.append(total)
        return prefix

    @staticmethod
    def _ks_statistic(data1: Iterable[float], data2: Iterable[float]) -> float:
        a = sorted(float(v) for v in data1)
        b = sorted(float(v) for v in data2)
        n1 = len(a)
        n2 = len(b)
        if n1 == 0 or n2 == 0:
            return 0.0
        i = j = 0
        cdf1 = cdf2 = 0.0
        d = 0.0
        while i < n1 and j < n2:
            if a[i] <= b[j]:
                x = a[i]
                while i < n1 and a[i] == x:
                    i += 1
                cdf1 = i / n1
            else:
                x = b[j]
                while j < n2 and b[j] == x:
                    j += 1
                cdf2 = j / n2
            d = max(d, abs(cdf1 - cdf2))
        while i < n1:
            i += 1
            cdf1 = i / n1
            d = max(d, abs(cdf1 - cdf2))
        while j < n2:
            j += 1
            cdf2 = j / n2
            d = max(d, abs(cdf1 - cdf2))
        return d

    @staticmethod
    def _ks_pvalue(d_stat: float, n1: int, n2: int) -> float:
        if n1 == 0 or n2 == 0:
            return 1.0
        en = math.sqrt(n1 * n2 / (n1 + n2))
        if en == 0:
            return 1.0
        lam = (en + 0.12 + 0.11 / en) * d_stat
        if lam <= 0:
            return 1.0
        summation = 0.0
        k = 1
        while True:
            term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * (lam ** 2) * (k ** 2))
            summation += term
            if abs(term) < 1e-8 or k > 100:
                break
            k += 1
        return max(min(summation, 1.0), 0.0)


__all__ = ["DriftDetector", "DriftEvent"]
