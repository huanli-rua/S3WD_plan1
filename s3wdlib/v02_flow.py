"""v02 流式前滚主流程实现（严格 year-month 窗口）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Tuple

import logging
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .config_loader import load_yaml_cfg, show_cfg
from .data_io import assign_year_from_month_sequence, augment_airline_features, load_table_auto
from .bucketizer import assign_buckets, configure as bucket_configure
from .similarity import configure as similarity_configure, current_config as similarity_current_config, corr_to_set
from .ref_tuple import build_ref_tuples
from .gwb import GWBProbEstimator
from .batch_measure import compute_region_masks, expected_cost, expected_fbeta, to_trisect_probs
from .threshold_selector import select_alpha_beta
from .smoothing import ema_clip
from .drift_controller import apply_actions, detect_drift
from .evalx import classification_metrics
from .zh_utils import fix_minus, set_chinese_font


_logger = logging.getLogger(__name__)
if not _logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _logger.addHandler(handler)
_logger.setLevel(logging.INFO)


_PERIOD_LABEL_BASE_YEAR: int | None = None


def _set_period_label_base(year: int) -> None:
    """Set the reference year for window labels."""

    global _PERIOD_LABEL_BASE_YEAR
    _PERIOD_LABEL_BASE_YEAR = int(year)


def year_month_to_float(year: int, month: int, base_year: int | None = None) -> float:
    """Encode a year-month pair into the requested float representation."""

    if month < 1 or month > 12:
        raise ValueError("Month value must be in [1, 12] for window labels.")
    if base_year is None:
        base_year = _PERIOD_LABEL_BASE_YEAR
    if base_year is None:
        base_year = year
    year_offset = int(year) - int(base_year)
    month_component = int(month) / 100.0
    if year_offset >= 0:
        label_value = year_offset + month_component
    else:
        label_value = year_offset - month_component
    return float(f"{label_value:.2f}")


def format_window_label(value: float) -> str:
    """Format window label floats with two decimal places."""

    return f"{value:.2f}"


@dataclass
class WindowPlan:
    """定义 warmup/stream 年度窗口计划。"""

    warmup_periods: List[pd.Period]
    stream_periods: List[pd.Period]
    warmup_years: List[int]
    stream_years: List[int]
    all_years: List[int]


@dataclass
class WindowRecord:
    """缓存单个窗口的特征/标签/预测，用于季节桶增量。"""

    period: pd.Period
    X: pd.DataFrame
    y: np.ndarray
    buckets: np.ndarray
    gwb_prob: Optional[np.ndarray]
    p_pos: np.ndarray
    pred_region: np.ndarray
    unlock_period: pd.Period


@dataclass
class SeasonalSamples:
    """聚合后的示范样本。"""

    X: pd.DataFrame
    y: np.ndarray
    buckets: np.ndarray
    gwb_prob: Optional[np.ndarray]
    weight: np.ndarray
    source_detail: Dict[str, object] = field(default_factory=dict)


@dataclass
class SeasonalReservoir:
    """维护季节桶（Month Bucket），支持延迟解锁与权重计算。"""

    keep_history_ratio: float = 0.3
    history_cap: int = 12
    recent_windows: int = 3
    time_decay: float = 0.15
    seasonal_neighbor: float = 0.7
    seasonal_other: float = 0.4
    weight_max: float = 2.5

    history_by_month: Dict[int, List[WindowRecord]] = field(default_factory=dict)
    global_history: List[WindowRecord] = field(default_factory=list)
    pending: List[WindowRecord] = field(default_factory=list)
    drift_multiplier: float = 1.0

    def set_keep_ratio(self, value: float) -> None:
        self.keep_history_ratio = float(max(0.0, min(1.0, value)))

    def update_drift(self, level: str) -> None:
        """根据漂移级别调整近期样本权重的放大系数。"""

        level = str(level or "NONE").upper()
        if level == "S3":
            self.drift_multiplier = max(self.drift_multiplier, 1.5)
        elif level == "S2":
            self.drift_multiplier = max(self.drift_multiplier, 1.3)
        elif level == "S1":
            self.drift_multiplier = max(self.drift_multiplier, 1.15)
        else:
            # 漂移稳定期逐步回落
            if self.drift_multiplier > 1.0:
                self.drift_multiplier = max(1.0, self.drift_multiplier * 0.9)

    def schedule(self, record: WindowRecord) -> None:
        self.pending.append(record)

    def unlock_until(self, period: pd.Period) -> List[WindowRecord]:
        ready: List[WindowRecord] = []
        remain: List[WindowRecord] = []
        for record in self.pending:
            if record.unlock_period <= period:
                ready.append(record)
            else:
                remain.append(record)
        self.pending = remain
        for record in ready:
            self._add_history(record)
        return ready

    def _add_history(self, record: WindowRecord) -> None:
        month = int(record.period.month)
        bucket = self.history_by_month.setdefault(month, [])
        bucket.append(record)
        bucket.sort(key=lambda r: r.period)
        if self.history_cap and len(bucket) > self.history_cap:
            self.history_by_month[month] = bucket[-self.history_cap :]
        self.global_history.append(record)
        self.global_history.sort(key=lambda r: r.period)

    def gather(self, period: pd.Period) -> SeasonalSamples | None:
        month = int(period.month)
        same_month_records = [r for r in self.history_by_month.get(month, []) if r.period < period]
        same_month_records.sort(key=lambda r: r.period)
        if self.recent_windows > 0:
            extras: List[WindowRecord] = []
            for record in reversed(self.global_history):
                if record.period >= period:
                    continue
                if record.period.month == month:
                    continue
                extras.append(record)
                if len(extras) >= self.recent_windows:
                    break
            extras = list(reversed(extras))
        else:
            extras = []

        records = same_month_records + extras
        if not records:
            return None

        X_parts: List[pd.DataFrame] = []
        y_parts: List[np.ndarray] = []
        bucket_parts: List[np.ndarray] = []
        gwb_parts: List[np.ndarray] = []
        weight_parts: List[np.ndarray] = []

        same_month_periods = [r.period for r in same_month_records]
        newest_same_month = max(same_month_periods) if same_month_periods else None
        for idx, record in enumerate(records):
            base_weight = self._weights_for_record(record, period, idx, newest_same_month)
            X_parts.append(record.X)
            y_parts.append(record.y)
            bucket_parts.append(record.buckets)
            if record.gwb_prob is not None:
                gwb_parts.append(record.gwb_prob)
            weight_parts.append(base_weight)

        X_all = pd.concat(X_parts, ignore_index=True)
        y_all = np.concatenate(y_parts)
        bucket_all = np.concatenate(bucket_parts)
        weight_all = np.concatenate(weight_parts)
        gwb_all = np.concatenate(gwb_parts) if gwb_parts else None
        detail = {
            "same_month_windows": [_period_to_str(p) for p in same_month_periods],
            "same_month_years": sorted({int(p.year) for p in same_month_periods}),
            "same_month_samples": int(sum(len(r.y) for r in same_month_records)),
            "neighbor_windows": [_period_to_str(r.period) for r in extras],
            "neighbor_samples": int(sum(len(r.y) for r in extras)),
        }
        return SeasonalSamples(X_all, y_all, bucket_all, gwb_all, weight_all, detail)

    def _weights_for_record(
        self,
        record: WindowRecord,
        current_period: pd.Period,
        rank: int,
        newest_same_month: Optional[pd.Period],
    ) -> np.ndarray:
        months_diff = max(int(current_period - record.period), 1)
        w_time = math.exp(-self.time_decay * (months_diff - 1))
        month_gap = abs(int(current_period.month) - int(record.period.month))
        month_gap = min(month_gap, 12 - month_gap)
        if month_gap == 0:
            w_season = 1.0
        elif month_gap == 1:
            w_season = self.seasonal_neighbor
        else:
            w_season = self.seasonal_other
        keep_factor = 1.0
        if (
            record.period.month == current_period.month
            and newest_same_month is not None
        ):
            # 越旧的同月样本按 keep_history_ratio 衰减
            delta = int(newest_same_month - record.period)
            keep_factor = self.keep_history_ratio ** max(0, delta)
        w_base = w_time * w_season * keep_factor * self.drift_multiplier

        if record.p_pos is not None:
            prob = record.p_pos
        else:
            prob = np.full(len(record.y), 0.5, dtype=float)
        region = record.pred_region if record.pred_region is not None else np.zeros(len(record.y), dtype=int)
        conf_from_prob = np.clip(np.abs(prob - 0.5) * 2.0, 0.2, 1.0)
        conf = conf_from_prob
        if region.size:
            conf = conf * np.where(region == 0, 0.7, 1.0)
            pred_binary = np.where(region == 1, 1, 0)
            mismatch = (pred_binary != record.y) & (region != 0)
            conf = np.where(mismatch, conf * 0.6, conf)

        weight = np.clip(w_base * conf, 1e-6, self.weight_max)
        return weight.astype(float)


def _calc_distribution_metrics(base_probs: np.ndarray, curr_probs: np.ndarray, bins: np.ndarray) -> Tuple[float, float]:
    base_hist, _ = np.histogram(base_probs, bins=bins)
    curr_hist, _ = np.histogram(curr_probs, bins=bins)
    base_ratio = np.clip(base_hist / max(base_hist.sum(), 1), 1e-6, None)
    curr_ratio = np.clip(curr_hist / max(curr_hist.sum(), 1), 1e-6, None)
    psi = float(np.sum((curr_ratio - base_ratio) * np.log(curr_ratio / base_ratio)))
    tv = float(0.5 * np.sum(np.abs(curr_ratio - base_ratio)))
    return psi, tv


def _compute_similarity_block(
    X_block: pd.DataFrame,
    bucket_ids: np.ndarray,
    ref_map: Dict[str, dict],
    sigma: float,
    sim_params: MutableMapping[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    cat_weights = sim_params.get("cat_weights", {})
    combine = sim_params.get("combine", "product")
    mix_alpha = sim_params.get("mix_alpha", 0.7)
    E_pos = np.zeros(len(X_block), dtype=float)
    E_neg = np.zeros(len(X_block), dtype=float)
    index_map: Dict[str, List[int]] = {}
    for idx, bucket in enumerate(bucket_ids):
        index_map.setdefault(str(bucket), []).append(idx)
    for bucket_id, positions in index_map.items():
        refs = ref_map.get(bucket_id)
        if not refs:
            continue
        subset = X_block.iloc[positions]
        if refs.get("pos"):
            E_pos[positions] = corr_to_set(
                subset,
                refs["pos"],
                sigma,
                cat_weights,
                combine=combine,
                mix_alpha=mix_alpha,
            )
        if refs.get("neg"):
            E_neg[positions] = corr_to_set(
                subset,
                refs["neg"],
                sigma,
                cat_weights,
                combine=combine,
                mix_alpha=mix_alpha,
            )
    return E_pos, E_neg


def _ensure_year_month(X: pd.DataFrame, start_year: Optional[int] = None) -> pd.DataFrame:
    if "Month" not in X.columns:
        raise ValueError("数据集缺少 Month 列，无法按 year-month 窗口前滚。")

    prepared = assign_year_from_month_sequence(X, start_year=start_year, copy=True)
    prepared["Month"] = prepared["Month"].astype(int)
    prepared["Year"] = prepared["Year"].astype(int)
    if "Year_synth" in prepared.columns:
        prepared["Year_synth"] = prepared["Year_synth"].astype(int)
    min_year = int(prepared["Year"].min())
    _set_period_label_base(min_year)
    prepared["period"] = pd.PeriodIndex(year=prepared["Year"], month=prepared["Month"], freq="M")
    prepared.sort_values(["period"], inplace=True)
    prepared.reset_index(drop=True, inplace=True)
    return prepared


def _prepare_windows(X: pd.DataFrame, warmup_windows: int) -> WindowPlan:
    periods = sorted(X["period"].unique())
    years = sorted({int(p.year) for p in periods})
    if not years:
        raise RuntimeError("无法根据数据构建窗口年份序列。")

    requested = max(0, int(warmup_windows))
    if requested == 0:
        requested = 12
    warmup_year_count = max(1, math.ceil(requested / 12))

    if warmup_year_count >= len(years):
        warmup_years = years[:-1]
    else:
        warmup_years = years[:warmup_year_count]

    if not warmup_years:
        warmup_years = [years[0]]

    stream_years = [year for year in years if year not in warmup_years]
    warmup_periods = [p for p in periods if int(p.year) in warmup_years]
    stream_periods = [p for p in periods if int(p.year) in stream_years]

    return WindowPlan(
        warmup_periods=warmup_periods,
        stream_periods=stream_periods,
        warmup_years=warmup_years,
        stream_years=stream_years,
        all_years=years,
    )


def _period_to_str(period: pd.Period) -> float:
    return year_month_to_float(int(period.year), int(period.month))


def _build_figures(metrics_by_year: pd.DataFrame, output_dir: Path) -> None:
    set_chinese_font()
    fix_minus()
    metric_map = {
        "Precision": "精确率",
        "Recall": "召回率",
        "F1": "F1",
        "BAC": "平衡准确率",
        "MCC": "MCC",
        "Kappa": "Kappa",
        "AUC": "AUC",
        "BND_ratio": "边界域占比",
        "POS_coverage": "正类覆盖率",
    }
    years = metrics_by_year.index.astype(int)
    for metric, zh_label in metric_map.items():
        if metric not in metrics_by_year.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(years, metrics_by_year[metric], marker="o", linewidth=2)
        ax.set_xlabel("年份")
        ax.set_ylabel(zh_label)
        ax.set_title(f"【年度指标】{zh_label}（按年加权平均）")
        ax.set_xticks(years)
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig_path = output_dir / f"yearly_{metric.lower()}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


def _normalize_grid_range(entry: Optional[List[float] | Tuple[float, ...]], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """标准化阈值网格范围配置。"""

    if entry is None:
        entry = default
    if len(entry) != 3:
        raise ValueError("阈值网格配置需包含 [start, end, step] 三个数值。")
    start, end, step = float(entry[0]), float(entry[1]), float(entry[2])
    if step <= 0:
        raise ValueError("阈值网格 step 必须为正数。")
    if end < start:
        start, end = end, start
    return start, end, step


def _limit_grid_range(base_range: Tuple[float, float, float], center: Optional[float], radius: float) -> Tuple[float, float, float]:
    """将网格范围限制到以 `center` 为中心的步长约束内。"""

    if center is None or radius <= 0:
        return base_range
    low = max(base_range[0], center - radius)
    high = min(base_range[1], center + radius)
    if high < low:
        low = high = max(min(center, base_range[1]), base_range[0])
    return low, high, base_range[2]


def run_streaming_flow(
    cfg_path: str,
    *,
    warmup_windows: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """执行严格 year-month 的 v02 流程，返回生成的 DataFrame。"""

    cfg = load_yaml_cfg(cfg_path)
    show_cfg(cfg)

    if not cfg.get("SWITCH", {}).get("enable_ref_tuple", True):
        raise RuntimeError("当前配置未启用 Reference Tuple 主线（SWITCH.enable_ref_tuple=false）。")
    if cfg.get("SWITCH", {}).get("enable_pso", False):
        _logger.warning("SWITCH.enable_pso=true 时将回退旧主线，本流程不支持 PSO。")

    flow_cfg = cfg.get("FLOW", {}) or {}
    warmup_span = int(flow_cfg.get("warmup_windows", warmup_windows or 6))
    recent_windows = int(flow_cfg.get("recent_windows", 3))
    time_decay = float(flow_cfg.get("time_decay", 0.15))
    seasonal_neighbor = float(flow_cfg.get("seasonal_neighbor", 0.7))
    seasonal_other = float(flow_cfg.get("seasonal_other", 0.4))
    weight_max = float(flow_cfg.get("weight_max", 2.5))
    history_cap = int(flow_cfg.get("history_cap", 12))

    data_dir = Path(cfg["DATA"]["data_dir"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    start_year_cfg = cfg["DATA"].get("start_year")
    if start_year_cfg is None:
        raise KeyError("配置缺少 DATA.start_year，用于按年份拆分窗口。")
    start_year_cfg = int(start_year_cfg)

    time_cfg = cfg.get("TIME", {}) or {}
    split_mode = str(time_cfg.get("split", "year_month"))
    if split_mode != "year_month":
        raise ValueError("TIME.split 必须为 'year_month'，以保证窗口按年-月前滚。")

    val_cfg = cfg.get("VAL", {}) or {}
    inline_delay = bool(val_cfg.get("inline_delay", True))
    if not inline_delay:
        _logger.warning("VAL.inline_delay=false 将导致验证混入历史样本，请确认需求。")
    else:
        _logger.info("【验证阶段】VAL.inline_delay=true → 当前月评估仅使用延迟到达的标注。")

    bucket_configure(cfg.get("BUCKET"))
    similarity_configure(cfg.get("SIMILARITY"))
    sim_cfg = similarity_current_config()

    data_path = data_dir / cfg["DATA"]["data_file"]
    X_raw, y = load_table_auto(
        str(data_path),
        continuous_label=cfg["DATA"].get("continuous_label"),
        threshold=cfg["DATA"].get("threshold"),
        threshold_op=cfg["DATA"].get("threshold_op"),
    )
    X_enriched = augment_airline_features(X_raw)
    X_enriched = _ensure_year_month(X_enriched, start_year=start_year_cfg)
    year_counts = (
        X_enriched.groupby("Year", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    categorical_candidates = [
        c for c in ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month"] if c in X_enriched.columns
    ]
    categorical_set = set(categorical_candidates)
    numeric_cols = [
        col
        for col in X_enriched.columns
        if pd.api.types.is_numeric_dtype(X_enriched[col]) and col not in categorical_set
    ]
    if not numeric_cols:
        raise ValueError("数据集中缺少可用于 GWB 估计的数值特征列。")
    X_numeric = X_enriched[numeric_cols]

    window_plan = _prepare_windows(X_enriched, warmup_span)
    warmup_periods = window_plan.warmup_periods
    stream_periods = window_plan.stream_periods
    warmup_years = window_plan.warmup_years
    stream_years = window_plan.stream_years
    all_years = window_plan.all_years
    warmup_labels = [_period_to_str(p) for p in warmup_periods]
    stream_labels = [_period_to_str(p) for p in stream_periods]
    warmup_labels_fmt = [format_window_label(val) for val in warmup_labels]
    stream_labels_fmt = [format_window_label(val) for val in stream_labels]
    warmup_years_fmt = [str(year) for year in warmup_years]
    stream_years_fmt = [str(year) for year in stream_years]
    all_years_fmt = [str(year) for year in all_years]
    if not stream_periods:
        raise RuntimeError("流式窗口不足，请调整 FLOW.warmup_windows。")

    _logger.info(
        "【初始化】warmup(月)=%s，stream(月)=%s，总窗口=%d",
        warmup_labels_fmt,
        stream_labels_fmt,
        len(warmup_periods) + len(stream_periods),
    )
    _logger.info("【年度样本量】%s", year_counts)
    _logger.info("【窗口计划】年度顺序=%s", all_years_fmt)
    _logger.info("【窗口计划】warmup(年)=%s，stream(年)=%s", warmup_years_fmt, stream_years_fmt)

    warmup_mask = X_enriched["period"].isin(warmup_periods)
    X_warm = X_enriched.loc[warmup_mask].reset_index(drop=True)
    X_warm_num = X_numeric.loc[warmup_mask].reset_index(drop=True)
    y_warm = y.loc[warmup_mask].reset_index(drop=True)
    buckets_warm = assign_buckets(X_warm)

    cat_cols = list(categorical_candidates)

    gwb_cfg = cfg["GWB"]
    gwb_estimator = GWBProbEstimator(
        k=gwb_cfg.get("k", 6),
        metric=gwb_cfg.get("metric", "euclidean"),
        eps=gwb_cfg.get("eps", 1e-6),
        mode=gwb_cfg.get("mode", "epanechnikov"),
        bandwidth=gwb_cfg.get("bandwidth"),
        bandwidth_scale=gwb_cfg.get("bandwidth_scale", 1.0),
        use_faiss=gwb_cfg.get("use_faiss", True),
        faiss_gpu=gwb_cfg.get("faiss_gpu", True),
        categorical_features=cat_cols,
        category_penalty=gwb_cfg.get("category_penalty", 0.3),
    )
    gwb_estimator.fit(
        X_warm_num.values,
        y_warm.to_numpy(),
        categorical_values=X_warm[cat_cols] if cat_cols else None,
    )

    use_gwb_weight = cfg["REF_TUPLE"].get("use_gwb_weight", True)
    gwb_prob_warm = None
    if use_gwb_weight:
        gwb_prob_warm = gwb_estimator.predict_proba(
            X_warm_num.values,
            categorical_values=X_warm[cat_cols] if cat_cols else None,
        )

    reservoir = SeasonalReservoir(
        keep_history_ratio=cfg["REF_TUPLE"].get("keep_history_ratio", 0.3),
        history_cap=history_cap,
        recent_windows=recent_windows,
        time_decay=time_decay,
        seasonal_neighbor=seasonal_neighbor,
        seasonal_other=seasonal_other,
        weight_max=weight_max,
    )

    warmup_records: List[WindowRecord] = []
    for period in warmup_periods:
        block_mask = X_warm["period"] == period
        if not block_mask.any():
            continue
        X_block = X_warm.loc[block_mask].reset_index(drop=True)
        y_block = y_warm.loc[block_mask].to_numpy()
        buckets_block = assign_buckets(X_block)
        gwb_block = None
        if gwb_prob_warm is not None:
            gwb_block = gwb_prob_warm[block_mask.to_numpy()]
        record = WindowRecord(
            period=period,
            X=X_block,
            y=y_block,
            buckets=buckets_block,
            gwb_prob=gwb_block,
            p_pos=np.full(len(X_block), 0.5, dtype=float),
            pred_region=np.zeros(len(X_block), dtype=int),
            unlock_period=period,
        )
        reservoir._add_history(record)
        warmup_records.append(record)

    seasonal_samples = reservoir.gather(stream_periods[0])
    if seasonal_samples is None:
        raise RuntimeError("季节桶为空，无法启动流式评估。")

    ref_tuples = build_ref_tuples(
        seasonal_samples.X,
        seasonal_samples.y,
        seasonal_samples.buckets,
        topk_per_class=cfg["REF_TUPLE"].get("topk_per_class", 256),
        pos_quantile=cfg["REF_TUPLE"].get("pos_quantile", 0.7),
        keep_history_ratio=reservoir.keep_history_ratio,
        gwb_prob=seasonal_samples.gwb_prob,
        sample_weight=seasonal_samples.weight,
        weight_clip=weight_max,
    )

    E_pos_warm, E_neg_warm = _compute_similarity_block(
        X_warm,
        buckets_warm,
        ref_tuples,
        sigma=sim_cfg.get("sigma", 0.5),
        sim_params=sim_cfg,
    )
    P_warm = to_trisect_probs(E_pos_warm, E_neg_warm)
    objective = cfg["MEASURE"].get("objective", "expected_cost")
    raw_grid = cfg["MEASURE"].get("grid", {"alpha": [0.55, 0.9, 0.02], "beta": [0.05, 0.45, 0.02]})
    alpha_range_cfg = _normalize_grid_range(raw_grid.get("alpha"), (0.55, 0.9, 0.02))
    beta_range_cfg = _normalize_grid_range(raw_grid.get("beta"), (0.05, 0.45, 0.02))
    measure_grid = {
        "alpha": list(alpha_range_cfg),
        "beta": list(beta_range_cfg),
    }
    measure_constraints = cfg["MEASURE"].get("constraints", {})
    measure_costs = cfg["MEASURE"].get("costs", {})
    beta_weight = cfg["MEASURE"].get("beta_weight", 1.0)

    alpha_prev, beta_prev, _ = select_alpha_beta(
        P_warm,
        measure_grid,
        measure_constraints,
        objective=objective,
        costs=measure_costs,
        beta_weight=beta_weight,
    )
    if objective == "expected_cost":
        baseline_perf = expected_cost(
            P_warm,
            alpha_prev,
            beta_prev,
            measure_costs.get("c_fn", 1.0),
            measure_costs.get("c_fp", 1.0),
            measure_costs.get("c_bnd", 0.5),
        )
    else:
        baseline_perf = expected_fbeta(P_warm, alpha_prev, beta_prev, beta_weight=beta_weight)

    baseline_bins = np.linspace(0, 1, 11)
    baseline_info = {
        "p_pos": P_warm["p_pos"],
        "posrate": float(y_warm.mean()),
        "perf": baseline_perf,
        "bins": baseline_bins,
    }

    runtime_state: Dict[str, object] = {
        "sigma": sim_cfg.get("sigma", 0.5),
        "constraints": dict(measure_constraints),
        "step_cap": dict(cfg["SMOOTH"].get("step_cap", {"alpha": 0.08, "beta": 0.08})),
        "actions_cfg": cfg.get("DRIFT", {}).get("actions", {}),
        "keep_history_ratio": reservoir.keep_history_ratio,
        "use_gwb_weight": use_gwb_weight,
        "ref_tuples": ref_tuples,
        "gwb_estimator": gwb_estimator,
        "cat_weights": sim_cfg.get("cat_weights", {}),
        "combine": sim_cfg.get("combine", "product"),
        "mix_alpha": sim_cfg.get("mix_alpha", 0.7),
    }

    ema_alpha_coeff = cfg["SMOOTH"].get("ema_alpha", 0.6)

    threshold_trace: List[Dict[str, float]] = []
    window_metrics: List[Dict[str, float]] = []
    drift_events: List[Dict[str, object]] = []
    prediction_records: List[Dict[str, object]] = []

    for seq_idx, period in enumerate(stream_periods, start=1):
        unlockeds = reservoir.unlock_until(period)
        if unlockeds:
            keep_summary = {
                "解锁数": len(unlockeds),
                "窗口": [_period_to_str(rec.period) for rec in unlockeds],
            }
            total_samples = sum(len(rec.y) for rec in reservoir.global_history)
            pos_samples = sum(int(rec.y.sum()) for rec in reservoir.global_history)
            neg_samples = total_samples - pos_samples
            _logger.info(
                "【增量】解锁标签月=%s；合入季节桶=%s；keep_history_ratio=%.2f/容量=%d；类均衡(POS/NEG)=%d/%d",
                ",".join(format_window_label(w) for w in keep_summary["窗口"]),
                len(reservoir.global_history),
                reservoir.keep_history_ratio,
                reservoir.history_cap,
                pos_samples,
                neg_samples,
            )

        seasonal_samples = reservoir.gather(period)
        if seasonal_samples is None or len(seasonal_samples.X) == 0:
            _logger.warning(
                "窗口 %s 缺少历史示范，跳过。",
                format_window_label(_period_to_str(period)),
            )
            continue

        detail = seasonal_samples.source_detail or {}
        same_windows = detail.get("same_month_windows") or []
        neighbor_windows = detail.get("neighbor_windows") or []
        same_years = detail.get("same_month_years") or []
        same_samples = detail.get("same_month_samples", 0)
        neighbor_samples = detail.get("neighbor_samples", 0)
        _logger.info(
            "【参考库】year-month=%s；同月历史窗口=%s（年份=%s，样本=%d）；邻近窗口=%s（样本=%d）；历史仅用于构建Γ/Ψ候选，不混入当月评估",
            format_window_label(_period_to_str(period)),
            "、".join(format_window_label(w) for w in same_windows) if same_windows else "无",
            "、".join(str(y) for y in same_years) if same_years else "无",
            same_samples,
            "、".join(format_window_label(w) for w in neighbor_windows) if neighbor_windows else "无",
            neighbor_samples,
        )

        ref_tuples = build_ref_tuples(
            seasonal_samples.X,
            seasonal_samples.y,
            seasonal_samples.buckets,
            topk_per_class=cfg["REF_TUPLE"].get("topk_per_class", 256),
            pos_quantile=cfg["REF_TUPLE"].get("pos_quantile", 0.7),
            keep_history_ratio=reservoir.keep_history_ratio,
            gwb_prob=seasonal_samples.gwb_prob,
            sample_weight=seasonal_samples.weight,
            weight_clip=weight_max,
        )
        runtime_state["ref_tuples"] = ref_tuples

        block_mask = X_enriched["period"] == period
        X_block = X_enriched.loc[block_mask].reset_index(drop=True)
        y_block = y.loc[block_mask].reset_index(drop=True)
        buckets_block = assign_buckets(X_block)

        def _recompute(sigma_value: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
            runtime_state["sigma"] = sigma_value
            similarity_configure({"sigma": sigma_value})
            E_pos, E_neg = _compute_similarity_block(
                X_block,
                buckets_block,
                ref_tuples,
                sigma=sigma_value,
                sim_params=runtime_state,
            )
            P_block = to_trisect_probs(E_pos, E_neg)
            alpha_star, beta_star, grid_info = select_alpha_beta(
                P_block,
                measure_grid,
                runtime_state["constraints"],
                objective=objective,
                costs=measure_costs,
                beta_weight=beta_weight,
            )

            step_caps = runtime_state.get("step_cap", {"alpha": 0.08, "beta": 0.08})
            alpha_cap = float(step_caps.get("alpha", 0.08))
            beta_cap = float(step_caps.get("beta", 0.08))

            def _evaluate(alpha_val: float, beta_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
                pos_mask, neg_mask, bnd_mask = compute_region_masks(P_block, alpha_val, beta_val)
                pos_cov_val = float(pos_mask.mean())
                bnd_ratio_val = float(bnd_mask.mean())
                return pos_mask, neg_mask, bnd_mask, pos_cov_val, bnd_ratio_val

            if alpha_prev is None or beta_prev is None:
                alpha_smooth, beta_smooth = alpha_star, beta_star
            else:
                alpha_smooth, beta_smooth = ema_clip(
                    alpha_star,
                    beta_star,
                    alpha_prev,
                    beta_prev,
                    ema_alpha_coeff,
                    alpha_cap,
                    beta_cap,
                )

            positive, negative, boundary, pos_cov, bnd_ratio = _evaluate(alpha_smooth, beta_smooth)

            constraint_cfg = runtime_state.get("constraints", {})
            min_pos_cov = float(constraint_cfg.get("min_pos_coverage", 0.0))
            bnd_cap = float(constraint_cfg.get("bnd_cap", 1.0))
            constraint_causes: List[str] = []
            if pos_cov < min_pos_cov:
                constraint_causes.append("POS<min_pos_coverage")
            if bnd_ratio > bnd_cap:
                constraint_causes.append("BND>bnd_cap")
            constraint_note = " & ".join(constraint_causes)
            constraint_resolution = "满足约束"

            if constraint_causes and alpha_prev is not None and beta_prev is not None:
                limited_alpha = _limit_grid_range(alpha_range_cfg, alpha_prev, alpha_cap)
                limited_beta = _limit_grid_range(beta_range_cfg, beta_prev, beta_cap)
                limited_grid = {"alpha": list(limited_alpha), "beta": list(limited_beta)}
                alt_alpha, alt_beta, _ = select_alpha_beta(
                    P_block,
                    limited_grid,
                    runtime_state["constraints"],
                    objective=objective,
                    costs=measure_costs,
                    beta_weight=beta_weight,
                )
                alt_pos, alt_neg, alt_bnd, alt_cov, alt_ratio = _evaluate(alt_alpha, alt_beta)
                if alt_cov >= min_pos_cov and alt_ratio <= bnd_cap:
                    alpha_smooth, beta_smooth = alt_alpha, alt_beta
                    positive, negative, boundary = alt_pos, alt_neg, alt_bnd
                    pos_cov, bnd_ratio = alt_cov, alt_ratio
                    constraint_resolution = "步长网格重算"
                    if constraint_note:
                        constraint_note = f"{constraint_note} → {constraint_resolution}"
                    constraint_causes = []
                else:
                    alpha_smooth, beta_smooth = alpha_prev, beta_prev
                    positive, negative, boundary, pos_cov, bnd_ratio = _evaluate(alpha_smooth, beta_smooth)
                    constraint_resolution = "回退上月阈值"
                    if constraint_note:
                        constraint_note = f"{constraint_note} → {constraint_resolution}"
                    constraint_causes = []
            elif constraint_causes:
                # 首个窗口直接使用网格解即可满足约束
                alpha_smooth, beta_smooth = alpha_star, beta_star
                positive, negative, boundary, pos_cov, bnd_ratio = _evaluate(alpha_smooth, beta_smooth)
                constraint_resolution = "首窗取网格解"
                if constraint_note:
                    constraint_note = f"{constraint_note} → {constraint_resolution}"
                constraint_causes = []

            if objective == "expected_cost":
                score_smoothed = expected_cost(
                    P_block,
                    alpha_smooth,
                    beta_smooth,
                    measure_costs.get("c_fn", 1.0),
                    measure_costs.get("c_fp", 1.0),
                    measure_costs.get("c_bnd", 0.5),
                )
                perf_drop = max(0.0, score_smoothed - baseline_info["perf"])
            else:
                score_smoothed = expected_fbeta(
                    P_block,
                    alpha_smooth,
                    beta_smooth,
                    beta_weight=beta_weight,
                )
                perf_drop = max(0.0, baseline_info["perf"] - score_smoothed)
            stats = {
                "P_block": P_block,
                "alpha_star": alpha_star,
                "beta_star": beta_star,
                "alpha_smooth": alpha_smooth,
                "beta_smooth": beta_smooth,
                "grid_info": grid_info,
                "positive": positive,
                "negative": negative,
                "boundary": boundary,
                "pos_cov": pos_cov,
                "bnd_ratio": bnd_ratio,
                "score": score_smoothed,
                "perf_drop": perf_drop,
                "constraint_resolution": constraint_resolution,
                "constraint_note": constraint_note,
                "E_pos": E_pos,
                "E_neg": E_neg,
                "alpha_cap": alpha_cap,
                "beta_cap": beta_cap,
            }
            return positive, boundary, stats

        _, _, stats = _recompute(float(runtime_state.get("sigma", 0.5)))

        psi_val, tv_val = _calc_distribution_metrics(
            baseline_info["p_pos"], stats["P_block"]["p_pos"], baseline_info["bins"]
        )
        posrate_curr = float(y_block.mean())
        posrate_gap = abs(posrate_curr - baseline_info["posrate"])


        stats_curr = {
            "window_id": int(f"{period.year}{period.month:02d}"),
            "psi": psi_val,
            "tv": tv_val,
            "posrate": posrate_gap,
            "perf_drop": stats["perf_drop"],
        }
        drift_level = detect_drift(stats_curr, baseline_info, cfg.get("DRIFT", {}))
        runtime_state = apply_actions(drift_level, runtime_state)
        reservoir.set_keep_ratio(float(runtime_state.get("keep_history_ratio", reservoir.keep_history_ratio)))
        reservoir.update_drift(drift_level)

        if drift_level == "S3":
            baseline_info["p_pos"] = stats["P_block"]["p_pos"]
            baseline_info["posrate"] = posrate_curr
            baseline_info["perf"] = stats["score"]

        actions_taken: List[str] = []
        if runtime_state.get("redo_threshold"):
            actions_taken.append("阈值重算")
        if runtime_state.get("rebuild_ref_tuple"):
            actions_taken.append("重建ΓΨ")
        if runtime_state.get("rebuild_gwb_index"):
            actions_taken.append("重建GWB")

        canonical_label = ""
        if drift_level != "NONE":
            canonical_actions: List[str] = []
            if drift_level in {"S1", "S3"}:
                canonical_actions.append("调σ")
            if drift_level in {"S2", "S3"}:
                canonical_actions.append("重建ΓΨ")
            if drift_level == "S3":
                canonical_actions.append("紧护栏")
                canonical_actions.append("重建GWB")
            canonical_label = "、".join(canonical_actions) if canonical_actions else "观察"

        if runtime_state.get("redo_threshold"):
            _, _, stats = _recompute(float(runtime_state.get("sigma", stats["alpha_smooth"])))
            runtime_state["redo_threshold"] = False

        grid_info = stats.get("grid_info", {})
        window_index = len(warmup_periods) + seq_idx
        period_label = _period_to_str(period)
        period_label_str = format_window_label(period_label)
        sample_count = len(X_block)
        month_value = int(period.month)
        year_label = "Year_synth" if "Year_synth" in X_block.columns else "Year"
        year_value = int(X_block[year_label].iloc[0])
        gamma_total = sum(len(bucket.get("pos", [])) for bucket in ref_tuples.values())
        psi_total = sum(len(bucket.get("neg", [])) for bucket in ref_tuples.values())
        same_windows = detail.get("same_month_windows") or []
        same_windows_str = (
            "、".join(format_window_label(w) for w in same_windows) if same_windows else "无"
        )
        E_pos_arr = np.asarray(stats.get("E_pos"), dtype=float)
        E_neg_arr = np.asarray(stats.get("E_neg"), dtype=float)
        mean_E_pos = float(np.nanmean(E_pos_arr)) if E_pos_arr.size else float("nan")
        mean_E_neg = float(np.nanmean(E_neg_arr)) if E_neg_arr.size else float("nan")
        corr_diff = float(np.nanmean(E_pos_arr - E_neg_arr)) if E_pos_arr.size else float("nan")
        P_block = stats["P_block"]
        mean_p_pos = float(np.nanmean(P_block["p_pos"])) if len(P_block["p_pos"]) else float("nan")
        mean_p_bnd = float(np.nanmean(P_block["p_bnd"])) if len(P_block["p_bnd"]) else float("nan")
        mean_p_neg = float(np.nanmean(P_block["p_neg"])) if len(P_block["p_neg"]) else float("nan")
        feasible_count = int(grid_info.get("feasible_count", 0))
        best_score = float(grid_info.get("best_score", float("nan")))
        second_score = float(grid_info.get("second_score", float("nan")))
        grid_delta = float(grid_info.get("delta", float("nan")))
        alpha_grid_str = f"[{alpha_range_cfg[0]:.2f},{alpha_range_cfg[1]:.2f},{alpha_range_cfg[2]:.2f}]"
        beta_grid_str = f"[{beta_range_cfg[0]:.2f},{beta_range_cfg[1]:.2f},{beta_range_cfg[2]:.2f}]"
        delta_alpha = (
            float("nan") if alpha_prev is None else abs(stats["alpha_smooth"] - alpha_prev)
        )
        delta_beta = (
            float("nan") if beta_prev is None else abs(stats["beta_smooth"] - beta_prev)
        )
        step_cap_str = f"{{'alpha':{stats['alpha_cap']:.3f},'beta':{stats['beta_cap']:.3f}}}"
        action_summary = ""
        if drift_level != "NONE":
            action_summary = canonical_label or ("、".join(actions_taken) if actions_taken else "观察")
        elif actions_taken:
            action_summary = "、".join(actions_taken)
        else:
            action_summary = "无"

        _logger.info(
            "【数据切片】win=%d(%s) 样本数=%d，Month=%02d，%s=%d",
            window_index,
            period_label_str,
            sample_count,
            month_value,
            year_label,
            year_value,
        )
        _logger.info(
            "【示范库】Γ数=%d，Ψ数=%d，历史同月来源窗口=%s",
            gamma_total,
            psi_total,
            same_windows_str,
        )
        _logger.info(
            "【相关度摘要】mean(E_pos)=%.4f，mean(E_neg)=%.4f，corr_diff=%.4f",
            mean_E_pos,
            mean_E_neg,
            corr_diff,
        )
        _logger.info(
            "【三域概率摘要】mean(p_pos)=%.4f，mean(p_bnd)=%.4f，mean(p_neg)=%.4f",
            mean_p_pos,
            mean_p_bnd,
            mean_p_neg,
        )
        _logger.info(
            "【网格信息】alpha网格=%s，beta网格=%s，可行格数=%d",
            alpha_grid_str,
            beta_grid_str,
            feasible_count,
        )
        _logger.info(
            "【目标面】best_score=%.4f，次优_score=%.4f，Δ=%.4f",
            best_score,
            second_score,
            grid_delta,
        )
        drift_msg = (
            f"【漂移判级】PSI={psi_val:.4f}，TV={tv_val:.4f}，PosRateΔ={posrate_gap:.4f}，等级={drift_level}"
        )
        drift_msg += f"，动作={action_summary}"
        _logger.info(drift_msg)
        _logger.info(
            "【重选后】α*=%.3f，β*=%.3f，EMA后=α=%.3f/β=%.3f，Δα=%.3f，Δβ=%.3f，ema=%.3f，step_cap=%s，BND占比=%.3f，目标=%.4f(%s)",
            stats["alpha_star"],
            stats["beta_star"],
            stats["alpha_smooth"],
            stats["beta_smooth"],
            delta_alpha,
            delta_beta,
            float(ema_alpha_coeff),
            step_cap_str,
            stats["bnd_ratio"],
            stats["score"],
            objective,
        )

        threshold_trace.append(
            {
                "window": period_label,
                "window_index": len(warmup_periods) + seq_idx,
                "year": int(period.year),
                "month": int(period.month),
                "alpha_star": float(stats["alpha_star"]),
                "beta_star": float(stats["beta_star"]),
                "alpha_smoothed": float(stats["alpha_smooth"]),
                "beta_smoothed": float(stats["beta_smooth"]),
                "objective_score": float(stats["score"]),
                "bnd_ratio": float(stats["bnd_ratio"]),
                "pos_coverage": float(stats["pos_cov"]),
                "constraint_note": stats.get("constraint_note", ""),
                "constraint_resolution": stats.get("constraint_resolution", ""),
                "grid_best_score": float(grid_info.get("best_score", float("nan"))),
                "grid_second_score": float(grid_info.get("second_score", float("nan"))),
                "grid_delta": float(grid_info.get("delta", float("nan"))),
                "grid_feasible": int(grid_info.get("feasible_count", 0)),
            }
        )

        pred_region = np.where(stats["positive"], 1, np.where(stats["negative"], -1, 0))
        y_pred_label = np.where(pred_region == 1, 1, 0)
        metrics_month = classification_metrics(y_block, y_pred_label, stats["P_block"]["p_pos"])
        precision = float(metrics_month.pop("Prec"))
        recall = float(metrics_month.pop("Rec"))

        window_metrics.append(
            {
                "year": int(period.year),
                "month": int(period.month),
                "window": period_label,
                "n_samples": int(len(X_block)),
                "Precision": precision,
                "Recall": recall,
                **metrics_month,
                "psi": float(psi_val),
                "tv": float(tv_val),
                "posrate_gap": float(posrate_gap),
                "posrate": posrate_curr,
                "perf_drop": float(stats["perf_drop"]),
                "BND_ratio": float(stats["bnd_ratio"]),
                "POS_coverage": float(stats["pos_cov"]),
                "alpha": float(stats["alpha_smooth"]),
                "beta": float(stats["beta_smooth"]),
                "constraint_resolution": stats.get("constraint_resolution", ""),
            }
        )

        alpha_prev, beta_prev = stats["alpha_smooth"], stats["beta_smooth"]

        record = WindowRecord(
            period=period,
            X=X_block,
            y=y_block.to_numpy(),
            buckets=buckets_block,
            gwb_prob=None,
            p_pos=stats["P_block"]["p_pos"],
            pred_region=pred_region,
            unlock_period=period + 1,
        )

        if use_gwb_weight:
            record.gwb_prob = gwb_estimator.predict_proba(
                X_block[numeric_cols].values,
                categorical_values=X_block[cat_cols] if cat_cols else None,
            )

        reservoir.schedule(record)
        if runtime_state.get("rebuild_gwb_index") and use_gwb_weight:
            if reservoir.global_history:
                hist_X = pd.concat([rec.X for rec in reservoir.global_history], ignore_index=True)
                hist_X_num = hist_X[numeric_cols]
                hist_y = np.concatenate([rec.y for rec in reservoir.global_history])
                gwb_estimator.fit(
                    hist_X_num.values,
                    hist_y,
                    categorical_values=hist_X[cat_cols] if cat_cols else None,
                )
                for rec in reservoir.global_history:
                    rec.gwb_prob = gwb_estimator.predict_proba(
                        rec.X[numeric_cols].values,
                        categorical_values=rec.X[cat_cols] if cat_cols else None,
                    )
            runtime_state["rebuild_gwb_index"] = False

        if runtime_state.get("rebuild_ref_tuple"):
            runtime_state["rebuild_ref_tuple"] = False

        if drift_level != "NONE":
            drift_events.append(
                {
                    "window": period_label,
                    "level": drift_level,
                    "psi": float(psi_val),
                    "tv": float(tv_val),
                    "posrate_gap": float(posrate_gap),
                    "perf_drop": float(stats["perf_drop"]),
                    "actions": canonical_label or ("、".join(actions_taken) if actions_taken else "观察"),
                }
            )

        prediction_records.append(
            {
                "period": period,
                "y_true": y_block.to_numpy(),
                "y_prob": stats["P_block"]["p_pos"],
                "y_pred": y_pred_label,
            }
        )

    trace_df = pd.DataFrame(threshold_trace)
    metrics_df = pd.DataFrame(window_metrics)
    if not metrics_df.empty:
        metrics_df.sort_values(["year", "month"], inplace=True)
    drift_df = pd.DataFrame(drift_events)

    trace_path = data_dir / "threshold_trace_v02.csv"
    metrics_path = data_dir / "window_metrics.csv"
    drift_path = data_dir / "drift_events.csv"
    trace_df.to_csv(trace_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    drift_df.to_csv(drift_path, index=False)

    yearly_rows: List[Dict[str, float]] = []
    metric_columns = [
        "Precision",
        "Recall",
        "F1",
        "BAC",
        "MCC",
        "Kappa",
        "AUC",
        "BND_ratio",
        "POS_coverage",
    ]

    if metrics_df.empty:
        metrics_by_year = pd.DataFrame(columns=["n_samples", *metric_columns])
        metrics_by_year.to_csv(data_dir / "yearly_metrics.csv")
        _build_figures(metrics_by_year, data_dir)
        return {
            "threshold_trace": trace_df,
            "window_metrics": metrics_df,
            "drift_events": drift_df,
            "metrics_by_year": metrics_by_year,
            "yearly_metrics": metrics_by_year,
            "window_order": {
                "warmup": warmup_years_fmt,
                "stream": stream_years_fmt,
                "warmup_periods": warmup_labels,
                "stream_periods": stream_labels,
                "all_years": all_years_fmt,
            },
        }

    def _weighted_average(values: pd.Series, weights: Optional[pd.Series]) -> float:
        arr = values.to_numpy(dtype=float)
        mask = ~np.isnan(arr)
        if not mask.any():
            return float("nan")
        if weights is None:
            return float(np.nanmean(arr[mask]))
        weight_arr = weights.to_numpy(dtype=float)[mask]
        total = np.sum(weight_arr)
        if not np.isfinite(total) or total <= 0:
            return float(np.nanmean(arr[mask]))
        return float(np.dot(arr[mask], weight_arr) / total)

    for year, group in metrics_df.groupby("year", sort=True):
        weights = group["n_samples"].astype(float)
        weight_series = None if weights.isna().any() else weights
        summary: Dict[str, float] = {"year": int(year)}
        total_samples = float(np.nansum(weights.to_numpy()))
        summary["n_samples"] = int(total_samples) if np.isfinite(total_samples) else 0
        for column in metric_columns:
            if column in group.columns:
                summary[column] = _weighted_average(group[column], weight_series)
            else:
                summary[column] = float("nan")
        yearly_rows.append(summary)

        coverage = group["month"].nunique()
        if coverage < 12:
            log_msg = (
                f"【年度汇总】year={int(year)}，F1={summary['F1']:.3f}，BAC={summary['BAC']:.3f}"
                f"（权重=样本数={summary['n_samples']}，月份覆盖率={coverage}/12）"
            )
        else:
            log_msg = (
                f"【年度汇总】year={int(year)}，F1={summary['F1']:.3f}，BAC={summary['BAC']:.3f}"
                f"（权重=样本数={summary['n_samples']}）"
            )
        _logger.info(log_msg)

    metrics_by_year = pd.DataFrame(yearly_rows).set_index("year").sort_index()
    yearly_path = data_dir / "yearly_metrics.csv"
    metrics_by_year.to_csv(yearly_path)

    _build_figures(metrics_by_year, data_dir)

    return {
        "threshold_trace": trace_df,
        "window_metrics": metrics_df,
        "drift_events": drift_df,
        "metrics_by_year": metrics_by_year,
        "yearly_metrics": metrics_by_year,
        "window_order": {
            "warmup": warmup_years_fmt,
            "stream": stream_years_fmt,
            "warmup_periods": warmup_labels,
            "stream_periods": stream_labels,
            "all_years": all_years_fmt,
        },
    }


__all__ = ["run_streaming_flow"]

