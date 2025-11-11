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
from .data_io import augment_airline_features, load_table_auto
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
        records = [r for r in self.history_by_month.get(month, []) if r.period < period]
        records.sort(key=lambda r: r.period)
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
            records.extend(reversed(extras))

        if not records:
            return None

        X_parts: List[pd.DataFrame] = []
        y_parts: List[np.ndarray] = []
        bucket_parts: List[np.ndarray] = []
        gwb_parts: List[np.ndarray] = []
        weight_parts: List[np.ndarray] = []

        newest_period = max(r.period for r in records)
        for idx, record in enumerate(records):
            base_weight = self._weights_for_record(record, period, idx, newest_period)
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
        return SeasonalSamples(X_all, y_all, bucket_all, gwb_all, weight_all)

    def _weights_for_record(
        self,
        record: WindowRecord,
        current_period: pd.Period,
        rank: int,
        newest_period: pd.Period,
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
        if record.period.month == current_period.month:
            # 越旧的同月样本按 keep_history_ratio 衰减
            delta = int(newest_period - record.period)
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


def _ensure_year_month(X: pd.DataFrame) -> pd.DataFrame:
    if "Year" not in X.columns or "Month" not in X.columns:
        raise ValueError("数据集缺少 Year / Month 列，无法按 year-month 窗口前滚。")
    X = X.copy()
    X["Year"] = X["Year"].astype(int)
    X["Month"] = X["Month"].astype(int)
    X["period"] = pd.PeriodIndex(year=X["Year"], month=X["Month"], freq="M")
    X.sort_values(["period"], inplace=True)
    X.reset_index(drop=True, inplace=True)
    return X


def _prepare_windows(X: pd.DataFrame, warmup_windows: int) -> Tuple[List[pd.Period], List[pd.Period]]:
    periods = X["period"].unique()
    periods = sorted(periods)
    if warmup_windows >= len(periods):
        warmup = periods[:-1]
    else:
        warmup = periods[:warmup_windows]
    stream = [p for p in periods if p not in warmup]
    return warmup, stream


def _period_to_str(period: pd.Period) -> str:
    return f"{period.year:04d}-{period.month:02d}"


def _build_figures(metrics_by_year: pd.DataFrame, output_dir: Path) -> None:
    set_chinese_font()
    fix_minus()
    for metric in ["Prec", "Rec", "F1", "BAC", "MCC", "Kappa", "AUC"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(metrics_by_year.index.astype(int), metrics_by_year[metric], marker="o", linewidth=2)
        ax.set_xlabel("年份")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} 年度走势")
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig_path = output_dir / f"metric_{metric.lower()}_by_year.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


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
    X_enriched = _ensure_year_month(X_enriched)

    warmup_periods, stream_periods = _prepare_windows(X_enriched, warmup_span)
    if not stream_periods:
        raise RuntimeError("流式窗口不足，请调整 FLOW.warmup_windows。")

    _logger.info(
        "【初始化】warmup=%s，stream=%s，总窗口=%d",
        [
            _period_to_str(p)
            for p in warmup_periods
        ],
        [
            _period_to_str(p)
            for p in stream_periods
        ],
        len(warmup_periods) + len(stream_periods),
    )

    warmup_mask = X_enriched["period"].isin(warmup_periods)
    X_warm = X_enriched.loc[warmup_mask].reset_index(drop=True)
    y_warm = y.loc[warmup_mask].reset_index(drop=True)
    buckets_warm = assign_buckets(X_warm)

    cat_cols = [c for c in ["UniqueCarrier", "Origin", "Dest", "DayOfWeek", "Month"] if c in X_warm.columns]

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
        X_warm.values,
        y_warm.to_numpy(),
        categorical_values=X_warm[cat_cols] if cat_cols else None,
    )

    use_gwb_weight = cfg["REF_TUPLE"].get("use_gwb_weight", True)
    gwb_prob_warm = None
    if use_gwb_weight:
        gwb_prob_warm = gwb_estimator.predict_proba(
            X_warm.values,
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
    measure_grid = cfg["MEASURE"].get("grid", {"alpha": [0.55, 0.9, 0.02], "beta": [0.05, 0.45, 0.02]})
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

    for period in stream_periods:
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
                ",".join(keep_summary["窗口"]),
                len(reservoir.global_history),
                reservoir.keep_history_ratio,
                reservoir.history_cap,
                pos_samples,
                neg_samples,
            )

        seasonal_samples = reservoir.gather(period)
        if seasonal_samples is None or len(seasonal_samples.X) == 0:
            _logger.warning("窗口 %s 缺少历史示范，跳过。", _period_to_str(period))
            continue

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
            alpha_star, beta_star, grid_score = select_alpha_beta(
                P_block,
                measure_grid,
                runtime_state["constraints"],
                objective=objective,
                costs=measure_costs,
                beta_weight=beta_weight,
            )
            step_caps = runtime_state.get("step_cap", {"alpha": 0.08, "beta": 0.08})
            alpha_smooth, beta_smooth = ema_clip(
                alpha_star,
                beta_star,
                alpha_prev,
                beta_prev,
                ema_alpha_coeff,
                step_caps.get("alpha", 0.08),
                step_caps.get("beta", 0.08),
            )
            if alpha_prev is None or beta_prev is None:
                alpha_smooth, beta_smooth = alpha_star, beta_star
            positive, negative, boundary = compute_region_masks(P_block, alpha_smooth, beta_smooth)
            pos_cov = float(positive.mean())
            bnd_ratio = float(boundary.mean())
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
                "grid_score": grid_score,
                "positive": positive,
                "negative": negative,
                "boundary": boundary,
                "pos_cov": pos_cov,
                "bnd_ratio": bnd_ratio,
                "score": score_smoothed,
                "perf_drop": perf_drop,
            }
            return positive, boundary, stats

        _, _, stats = _recompute(float(runtime_state.get("sigma", 0.5)))

        psi_val, tv_val = _calc_distribution_metrics(
            baseline_info["p_pos"], stats["P_block"]["p_pos"], baseline_info["bins"]
        )
        posrate_curr = float(y_block.mean())
        posrate_gap = abs(posrate_curr - baseline_info["posrate"])

        _logger.info(
            "【窗口】year-month=%s；样本数=%d",
            _period_to_str(period),
            len(X_block),
        )
        delta_alpha = float("nan") if alpha_prev is None else abs(stats["alpha_smooth"] - alpha_prev)
        delta_beta = float("nan") if beta_prev is None else abs(stats["beta_smooth"] - beta_prev)
        _logger.info(
            "【阈值】alpha*=%0.3f,beta*=%0.3f → α_t=%0.3f/β_t=%0.3f；Δα=%0.3f；Δβ=%0.3f；BND占比=%0.3f；POS覆盖=%0.3f；目标值=%0.4f；σ=%0.3f",
            stats["alpha_star"],
            stats["beta_star"],
            stats["alpha_smooth"],
            stats["beta_smooth"],
            delta_alpha,
            delta_beta,
            stats["bnd_ratio"],
            stats["pos_cov"],
            stats["score"],
            float(runtime_state.get("sigma", 0.0)),
        )

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

        sigma_after = float(runtime_state.get("sigma", stats_curr["window_id"]))
        actions_taken: List[str] = []
        if runtime_state.get("redo_threshold"):
            actions_taken.append("阈值重算")
        if runtime_state.get("rebuild_ref_tuple"):
            actions_taken.append("重建ΓΨ")
        if runtime_state.get("rebuild_gwb_index"):
            actions_taken.append("重建GWB")

        if drift_level != "NONE":
            _logger.info(
                "【漂移】级别=%s；动作=%s；作用域=参考/示范/后续月度",
                drift_level,
                "、".join(actions_taken) if actions_taken else "观察",
            )

        if runtime_state.get("redo_threshold"):
            _, _, stats = _recompute(float(runtime_state.get("sigma", stats["alpha_smooth"])))
            runtime_state["redo_threshold"] = False

        threshold_trace.append(
            {
                "window": _period_to_str(period),
                "alpha_star": float(stats["alpha_star"]),
                "beta_star": float(stats["beta_star"]),
                "alpha_smoothed": float(stats["alpha_smooth"]),
                "beta_smoothed": float(stats["beta_smooth"]),
                "objective_score": float(stats["score"]),
                "bnd_ratio": float(stats["bnd_ratio"]),
                "pos_coverage": float(stats["pos_cov"]),
            }
        )

        window_metrics.append(
            {
                "window": _period_to_str(period),
                "psi": float(psi_val),
                "tv": float(tv_val),
                "posrate_gap": float(posrate_gap),
                "posrate": posrate_curr,
                "perf_drop": float(stats["perf_drop"]),
                "bnd_ratio": float(stats["bnd_ratio"]),
                "alpha": float(stats["alpha_smooth"]),
                "beta": float(stats["beta_smooth"]),
            }
        )

        pred_region = np.where(stats["positive"], 1, np.where(stats["negative"], -1, 0))
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
                X_block.values,
                categorical_values=X_block[cat_cols] if cat_cols else None,
            )

        reservoir.schedule(record)
        if runtime_state.get("rebuild_gwb_index") and use_gwb_weight:
            if reservoir.global_history:
                hist_X = pd.concat([rec.X for rec in reservoir.global_history], ignore_index=True)
                hist_y = np.concatenate([rec.y for rec in reservoir.global_history])
                gwb_estimator.fit(
                    hist_X.values,
                    hist_y,
                    categorical_values=hist_X[cat_cols] if cat_cols else None,
                )
                for rec in reservoir.global_history:
                    rec.gwb_prob = gwb_estimator.predict_proba(
                        rec.X.values,
                        categorical_values=rec.X[cat_cols] if cat_cols else None,
                    )
            runtime_state["rebuild_gwb_index"] = False

        if runtime_state.get("rebuild_ref_tuple"):
            runtime_state["rebuild_ref_tuple"] = False

        if drift_level != "NONE":
            drift_events.append(
                {
                    "window": _period_to_str(period),
                    "level": drift_level,
                    "psi": float(psi_val),
                    "tv": float(tv_val),
                    "posrate_gap": float(posrate_gap),
                    "perf_drop": float(stats["perf_drop"]),
                    "actions": "、".join(actions_taken) if actions_taken else "观察",
                }
            )

        prediction_records.append(
            {
                "period": period,
                "y_true": y_block.to_numpy(),
                "y_prob": stats["P_block"]["p_pos"],
                "y_pred": np.where(pred_region == 1, 1, 0),
            }
        )

    trace_df = pd.DataFrame(threshold_trace)
    metrics_df = pd.DataFrame(window_metrics)
    drift_df = pd.DataFrame(drift_events)

    trace_path = data_dir / "threshold_trace_v02.csv"
    metrics_path = data_dir / "window_metrics.csv"
    drift_path = data_dir / "drift_events.csv"
    trace_df.to_csv(trace_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    drift_df.to_csv(drift_path, index=False)

    eval_rows: List[Dict[str, float]] = []
    grouped: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for record in prediction_records:
        year = int(record["period"].year)
        grouped.setdefault(year, {"y": [], "pred": [], "prob": []})
        grouped[year]["y"].append(record["y_true"])
        grouped[year]["pred"].append(record["y_pred"])
        grouped[year]["prob"].append(record["y_prob"])

    for year, payload in sorted(grouped.items()):
        y_true = np.concatenate(payload["y"])
        y_pred = np.concatenate(payload["pred"])
        y_prob = np.concatenate(payload["prob"])
        metrics = classification_metrics(y_true, y_pred, y_prob)
        metrics["Year"] = year
        eval_rows.append(metrics)

    metrics_by_year = pd.DataFrame(eval_rows).set_index("Year").sort_index()
    metrics_by_year.to_csv(data_dir / "metrics_by_year.csv")

    _build_figures(metrics_by_year, data_dir)

    return {
        "threshold_trace": trace_df,
        "window_metrics": metrics_df,
        "drift_events": drift_df,
        "metrics_by_year": metrics_by_year,
    }


__all__ = ["run_streaming_flow"]

