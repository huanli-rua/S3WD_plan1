# -*- coding: utf-8 -*-
"""Helper dataclasses for configuring dynamic streaming components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple

from .drift import DriftDetector
from .incremental import PosteriorUpdater
from .trainer import PSOParams


@dataclass
class DynamicLoopConfig:
    """Configuration for threshold adaptation in streaming mode."""

    strategy: str = "windowed_pso"
    step: int = 512
    window_size: Optional[int] = None
    target_bnd: float = 0.18
    ema_alpha: float = 0.6
    median_window: int = 3
    keep_gap: Optional[float] = None
    fallback_rule: bool = True
    gamma_last: Optional[float] = None
    stall_rounds: int = 6

    def make_pso_params(self, base: Optional[PSOParams] = None) -> PSOParams:
        """Return a :class:`PSOParams` instance tailored for the configured strategy."""

        params = PSOParams() if base is None else base
        params.window_mode = self.strategy.lower() == "windowed_pso"
        params.window_size = self.window_size
        params.ema_alpha = self.ema_alpha
        params.median_window = self.median_window
        params.keep_gap = self.keep_gap
        params.fallback_rule = self.fallback_rule
        params.step = self.step
        params.target_bnd = self.target_bnd
        params.gamma_last = self.gamma_last
        params.stall_rounds = self.stall_rounds
        return params

    def is_windowed(self) -> bool:
        """Whether the dynamic loop should run the windowed PSO strategy."""

        return self.strategy.lower() == "windowed_pso"


@dataclass
class DriftDetectorConfig:
    """Configuration bundle for :class:`DriftDetector`."""

    method: str = "kswin"
    window_size: int = 512
    stat_size: int = 128
    significance: float = 0.05
    delta: float = 0.002
    cooldown: int = 0
    min_window_length: int = 32

    def build(self) -> DriftDetector:
        """Instantiate a :class:`DriftDetector` with the stored parameters."""

        return DriftDetector(
            method=self.method,
            window_size=self.window_size,
            stat_size=self.stat_size,
            significance=self.significance,
            delta=self.delta,
            cooldown=self.cooldown,
            min_window_length=self.min_window_length,
        )


@dataclass
class PosteriorUpdaterConfig:
    """Configuration container for :class:`PosteriorUpdater`."""

    buffer_size: int = 4096
    cache_strategy: str = "sliding"
    rebuild_interval: int = 512
    min_rebuild_interval: int = 64
    drift_shrink: float = 0.5
    immediate_rebuild_methods: Iterable[str] = field(default_factory=lambda: ("adwin",))
    enable_faiss_append: bool = True
    random_state: Optional[int] = None

    def build(self, estimator_factory: Callable[[], object]) -> PosteriorUpdater:
        """Create a :class:`PosteriorUpdater` from the stored parameters."""

        return PosteriorUpdater(
            estimator_factory=estimator_factory,
            buffer_size=self.buffer_size,
            cache_strategy=self.cache_strategy,
            rebuild_interval=self.rebuild_interval,
            min_rebuild_interval=self.min_rebuild_interval,
            drift_shrink=self.drift_shrink,
            immediate_rebuild_methods=tuple(self.immediate_rebuild_methods),
            enable_faiss_append=self.enable_faiss_append,
            random_state=self.random_state,
        )


def build_dynamic_components(
    dynamic: Optional[DynamicLoopConfig],
    drift: Optional[DriftDetectorConfig],
    incremental: Optional[PosteriorUpdaterConfig],
    estimator_factory: Optional[Callable[[], object]] = None,
    base_pso: Optional[PSOParams] = None,
) -> Tuple[Optional[PSOParams], Optional[DriftDetector], Optional[PosteriorUpdater]]:
    """Instantiate dynamic components according to the provided configs."""

    pso_params: Optional[PSOParams] = None
    if dynamic is not None:
        pso_params = dynamic.make_pso_params(base=base_pso)

    detector: Optional[DriftDetector] = None
    if drift is not None:
        detector = drift.build()

    updater: Optional[PosteriorUpdater] = None
    if incremental is not None:
        if estimator_factory is None:
            raise ValueError("estimator_factory 必须在启用增量配置时提供。")
        updater = incremental.build(estimator_factory)

    return pso_params, detector, updater
