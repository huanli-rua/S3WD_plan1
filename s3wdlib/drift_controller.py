# -*- coding: utf-8 -*-
"""漂移检测与响应控制模块。"""
from __future__ import annotations

from typing import Dict

_STATE: dict = {
    "warn_streak": 0,
    "alert_streak": 0,
    "cooldown": 0,
    "last_window": None,
}


def _update_window(stats_curr: dict, thresholds: dict) -> None:
    """维护窗口级别的去抖状态。"""

    window_id = stats_curr.get("window_id")
    if window_id is None:
        return
    if _STATE.get("last_window") == window_id:
        return
    _STATE["last_window"] = window_id
    if _STATE.get("cooldown", 0) > 0:
        _STATE["cooldown"] = max(0, int(_STATE["cooldown"]) - 1)
    debounce = max(int(thresholds.get("debounce_windows", 1)), 1)
    if debounce <= 1:
        return
    # 当窗口推进且未触发高级别时，逐步衰减 warn/alert 计数
    _STATE["warn_streak"] = max(0, int(_STATE.get("warn_streak", 0)) - 1)
    _STATE["alert_streak"] = max(0, int(_STATE.get("alert_streak", 0)) - 1)


def _metric_value(stats_curr: dict, stats_ref: dict, metric_key: str) -> float | None:
    """从当前/参考统计中提取指标值，兼容直接传入差异或原值。"""

    if metric_key in stats_curr:
        return float(stats_curr[metric_key])
    if stats_ref and metric_key in stats_ref and f"{metric_key}_curr" in stats_curr:
        return abs(float(stats_curr[f"{metric_key}_curr"]) - float(stats_ref[metric_key]))
    return None


def detect_drift(stats_curr: dict, stats_ref: dict, thresholds: dict) -> str:
    """
    输入：当前窗与参考窗的 PSI/TV/PosRate/性能 等统计
    输出：'NONE' | 'S1' | 'S2' | 'S3'
    """

    _update_window(stats_curr, thresholds)
    if _STATE.get("cooldown", 0) > 0:
        return "NONE"

    severity = 0
    for key, metric in [
        ("psi_thresholds", "psi"),
        ("tv_thresholds", "tv"),
        ("posrate_shift", "posrate"),
        ("perf_drop", "perf_drop"),
    ]:
        th = thresholds.get(key)
        if not isinstance(th, dict):
            continue
        value = _metric_value(stats_curr, stats_ref, metric)
        if value is None:
            continue
        warn = float(th.get("warn", float("inf")))
        alert = float(th.get("alert", float("inf")))
        if value >= alert:
            severity = max(severity, 2)
        elif value >= warn:
            severity = max(severity, 1)

    if severity == 0:
        _STATE["warn_streak"] = 0
        _STATE["alert_streak"] = 0
        return "NONE"

    debounce = max(int(thresholds.get("debounce_windows", 1)), 1)
    if severity == 2:
        _STATE["alert_streak"] = int(_STATE.get("alert_streak", 0)) + 1
        _STATE["warn_streak"] = int(_STATE.get("warn_streak", 0)) + 1
        if _STATE["alert_streak"] >= debounce:
            return "S3"
        return "S2"
    else:
        _STATE["alert_streak"] = 0
        _STATE["warn_streak"] = int(_STATE.get("warn_streak", 0)) + 1
        if _STATE["warn_streak"] >= debounce:
            return "S2"
        return "S1"


def apply_actions(level: str, state: Dict) -> Dict:
    """
    根据级别更新运行状态：
      S1：调整 sigma（×1.2 或 ×0.85）
      S2：重建受影响桶的 Γ/Ψ（保留 keep_history_ratio）
      S3：在 S2 基础上增量/重建 GWB 筛权；tighten 护栏与步长；cooldown 去抖
    返回：更新后的 state（包含 sigma/ΓΨ/护栏等）
    """

    if state is None:
        state = {}
    new_state = dict(state)
    actions_cfg = new_state.get("actions_cfg")
    if actions_cfg is None:
        actions_cfg = thresholds_from_state(new_state)
    if actions_cfg is None:
        actions_cfg = {}
    new_state["actions_cfg"] = actions_cfg

    if level == "NONE":
        return new_state

    if level == "S1":
        cfg = actions_cfg.get("S1", {})
        factors = cfg.get("sigma_factor", [1.0])
        if factors:
            toggle = int(new_state.get("sigma_cycle", 0))
            factor = float(factors[toggle % len(factors)])
            new_state["sigma"] = float(new_state.get("sigma", 0.5) * factor)
            new_state["sigma_cycle"] = toggle + 1
        if cfg.get("redo_threshold"):
            new_state["redo_threshold"] = True
        _STATE["warn_streak"] = 0
        return new_state

    if level == "S2":
        cfg = actions_cfg.get("S2", {})
        if cfg.get("rebuild_ref_tuple"):
            new_state["rebuild_ref_tuple"] = True
        keep_ratio = cfg.get("keep_history_ratio")
        if keep_ratio is not None:
            new_state["keep_history_ratio"] = float(keep_ratio)
        if cfg.get("redo_threshold"):
            new_state["redo_threshold"] = True
        _STATE["warn_streak"] = 0
        _STATE["alert_streak"] = 0
        return new_state

    if level == "S3":
        cfg = actions_cfg.get("S3", {})
        if cfg.get("rebuild_ref_tuple"):
            new_state["rebuild_ref_tuple"] = True
        if cfg.get("rebuild_gwb_index"):
            new_state["rebuild_gwb_index"] = True
        keep_ratio = cfg.get("keep_history_ratio")
        if keep_ratio is not None:
            new_state["keep_history_ratio"] = float(keep_ratio)
        tighten = cfg.get("tighten", {}) or {}
        constraints = dict(new_state.get("constraints", {}))
        if "keep_gap" in tighten:
            constraints["keep_gap"] = float(tighten["keep_gap"])
        if "bnd_cap" in tighten:
            constraints["bnd_cap"] = float(tighten["bnd_cap"])
        new_state["constraints"] = constraints
        if "step_cap" in tighten:
            step_cap_cfg = tighten["step_cap"]
            step_cap_dict = dict(new_state.get("step_cap", {}))
            if isinstance(step_cap_cfg, dict):
                for key, val in step_cap_cfg.items():
                    step_cap_dict[key] = float(val)
            else:
                step_cap_dict["alpha"] = float(step_cap_cfg)
                step_cap_dict["beta"] = float(step_cap_cfg)
            new_state["step_cap"] = step_cap_dict
        cooldown = int(cfg.get("cooldown_windows", 0))
        if cooldown > 0:
            _STATE["cooldown"] = cooldown
        new_state["redo_threshold"] = True
        _STATE["warn_streak"] = 0
        _STATE["alert_streak"] = 0
        return new_state

    return new_state


def thresholds_from_state(state: Dict) -> Dict | None:
    """从状态字典中提取 actions 配置，兼容直接传入。"""

    return state.get("actions") or state.get("actions_cfg")
