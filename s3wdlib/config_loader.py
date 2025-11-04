# -*- coding: utf-8 -*-
from __future__ import annotations
import os, yaml
from typing import Optional, Tuple

from .streaming import DynamicLoopConfig, DriftDetectorConfig, PosteriorUpdaterConfig

# 支持分组与扁平配置；本项目统一推荐扁平键名：
#   S3_sigma: 3.0
#   S3_regret_mode: utility
REQUIRED_GROUPS = {"DATA", "LEVEL", "KWB", "GWB", "S3WD", "PSO"}
OPTIONAL_GROUPS = {"DYN", "DRIFT", "INCR"}
GROUPS = REQUIRED_GROUPS | OPTIONAL_GROUPS

def _normalize_flat_to_grouped(raw: dict) -> dict:
    """将扁平键名（例如 S3_sigma）映射为内部分组结构，兼容旧配置。"""
    D = {}

    # DATA
    dp = raw.get("DATA_PATH")
    if dp:
        ddir, dfile = os.path.split(dp)
    else:
        ddir = raw.get("DATA_DIR")
        dfile = raw.get("DATA_FILE")
    D["DATA"] = {
        "data_dir": ddir,
        "data_file": dfile,
        "continuous_label": raw.get("CONT_LABEL"),
        "threshold": raw.get("CONT_THRESH"),
        "threshold_op": raw.get("CONT_OP"),
        "label_col": raw.get("LABEL_COL"),
        "positive_label": raw.get("POSITIVE_LABEL"),
        "test_size": raw.get("TEST_SIZE"),
        "val_size": raw.get("VAL_SIZE"),
        "random_state": raw.get("SEED"),
    }

    # LEVEL
    D["LEVEL"] = {
        "level_pcts": raw.get("LEVEL_PCTS"),
        "ranker": raw.get("RANKER"),
    }

    # KWB
    D["KWB"] = {
        "k": raw.get("KWB_K"),
        "metric": raw.get("KWB_metric","euclidean"),
        "eps": raw.get("KWB_eps", 1e-6),
        "use_faiss": raw.get("KWB_use_faiss", True),
        "faiss_gpu": raw.get("KWB_faiss_gpu", True),
    }

    # GWB
    D["GWB"] = {
        "k": raw.get("GWB_K"),
        "metric": raw.get("GWB_metric", "euclidean"),
        "eps": raw.get("GWB_eps", 1e-6),
        "mode": raw.get("GWB_mode", raw.get("GWB_kernel", "epanechnikov")),
        "bandwidth": raw.get("GWB_bandwidth"),
        "bandwidth_scale": raw.get("GWB_bandwidth_scale", 1.0),
        "use_faiss": raw.get("GWB_use_faiss", True),
        "faiss_gpu": raw.get("GWB_faiss_gpu", True),
    }

    # S3WD —— 关键：读取扁平键 S3_sigma / S3_regret_mode
    pen = raw.get("S3_penalty_large", raw.get("S3_pentalty_large"))
    D["S3WD"] = {
        "c1": raw.get("S3_c1"),
        "c2": raw.get("S3_c2"),
        "xi_min": raw.get("S3_xi_min"),
        "theta_pos": raw.get("S3_theta_pos"),
        "theta_neg": raw.get("S3_theta_neg"),
        "sigma": raw.get("S3_sigma"),                         # ← 统一命名
        "regret_mode": raw.get("S3_regret_mode","utility"),   # ← 统一命名
        "penalty_large": pen,
        "gamma_last": raw.get("S3_gamma_last", True),
        "gap": raw.get("S3_gap", 0.02),
    }

    # PSO
    D["PSO"] = {
        "particles": raw.get("PSO_particles"),
        "iters": raw.get("PSO_iters"),
        "w_max": raw.get("PSO_w_max"),
        "w_min": raw.get("PSO_w_min"),
        "c1": raw.get("PSO_c1"),
        "c2": raw.get("PSO_c2"),
        "seed": raw.get("PSO_seed"),
        "use_gpu": raw.get("PSO_use_gpu", True),
    }

    # Dynamic loop
    if any(k.startswith("DYN_") for k in raw):
        D["DYN"] = {
            "strategy": raw.get("DYN_strategy"),
            "step": raw.get("DYN_step"),
            "window_size": raw.get("DYN_window_size"),
            "target_bnd": raw.get("DYN_target_bnd"),
            "ema_alpha": raw.get("DYN_ema_alpha"),
            "median_window": raw.get("DYN_median_window"),
            "keep_gap": raw.get("DYN_keep_gap"),
            "fallback_rule": raw.get("DYN_fallback_rule"),
            "gamma_last": raw.get("DYN_gamma_last"),
            "stall_rounds": raw.get("DYN_stall_rounds"),
        }

    # Drift detector
    if any(k.startswith("DRIFT_") for k in raw):
        D["DRIFT"] = {
            "method": raw.get("DRIFT_method"),
            "window_size": raw.get("DRIFT_window_size"),
            "stat_size": raw.get("DRIFT_stat_size"),
            "significance": raw.get("DRIFT_significance"),
            "delta": raw.get("DRIFT_delta"),
            "cooldown": raw.get("DRIFT_cooldown"),
            "min_window_length": raw.get("DRIFT_min_window_length"),
        }

    # Incremental updater
    if any(k.startswith("INCR_") for k in raw):
        D["INCR"] = {
            "buffer_size": raw.get("INCR_buffer_size"),
            "cache_strategy": raw.get("INCR_cache_strategy"),
            "rebuild_interval": raw.get("INCR_rebuild_interval"),
            "min_rebuild_interval": raw.get("INCR_min_rebuild_interval"),
            "drift_shrink": raw.get("INCR_drift_shrink"),
            "immediate_rebuild_methods": raw.get("INCR_immediate_rebuild_methods"),
            "enable_faiss_append": raw.get("INCR_enable_faiss_append"),
            "random_state": raw.get("INCR_random_state"),
        }
    return D

def _require(G: dict, name: str, keys: list[str]):
    missing = [k for k in keys if G.get(k) is None]
    if missing:
        raise KeyError(f"{name} 缺少必需键: {missing}")

def load_yaml_cfg(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"YAML 为空或结构不是字典: {path}")

    # 若直接提供了分组则直接使用，否则把扁平键归一化为分组
    if GROUPS & set(raw.keys()):
        cfg = raw
    else:
        cfg = _normalize_flat_to_grouped(raw)

    missing = [g for g in REQUIRED_GROUPS if g not in cfg]
    if missing:
        raise KeyError(f"YAML 缺少分组: {missing}，必须包含 {sorted(REQUIRED_GROUPS)}")

    # 严格项校验
    _require(cfg["DATA"], "DATA", ["data_dir","data_file","test_size","val_size","random_state"])
    # Either continuous or label_col
    has_cont = all(cfg["DATA"].get(x) is not None for x in ["continuous_label","threshold","threshold_op"])
    has_label = all(cfg["DATA"].get(x) is not None for x in ["label_col","positive_label"])
    if not (has_cont or has_label):
        raise KeyError("DATA 需满足：连续标签 {continuous_label,threshold,threshold_op} 或 现成标签 {label_col,positive_label} 二选一")

    _require(cfg["LEVEL"], "LEVEL", ["level_pcts","ranker"])
    _require(cfg["KWB"],   "KWB",   ["k"])
    _require(cfg["GWB"],   "GWB",   ["k"])
    _require(cfg["S3WD"],  "S3WD",  ["c1","c2","xi_min","theta_pos","theta_neg","sigma","penalty_large","gamma_last"])
    _require(cfg["PSO"],   "PSO",   ["particles","iters","w_max","w_min","c1","c2","seed"])

    if "DYN" in cfg:
        _require(cfg["DYN"], "DYN", ["strategy", "step", "target_bnd"])
    if "DRIFT" in cfg:
        _require(cfg["DRIFT"], "DRIFT", ["method", "window_size", "stat_size"])
    if "INCR" in cfg:
        _require(cfg["INCR"], "INCR", ["buffer_size", "cache_strategy", "rebuild_interval"])

    return cfg

def extract_vars(cfg: dict) -> dict:
    """导出扁平变量名（统一风格），供其余模块直接使用。"""
    V = {}
    D = cfg["DATA"]
    V["DATA_PATH"] = os.path.join(D["data_dir"], D["data_file"])
    if D.get("continuous_label") is not None:
        V["CONT_LABEL"] = D["continuous_label"]
        V["CONT_THRESH"] = D["threshold"]
        V["CONT_OP"] = D["threshold_op"]
    if D.get("label_col") is not None:
        V["LABEL_COL"] = D["label_col"]
        V["POSITIVE_LABEL"] = D["positive_label"]
    V["TEST_SIZE"] = D["test_size"]
    V["VAL_SIZE"] = D["val_size"]
    V["SEED"] = D["random_state"]

    L = cfg["LEVEL"]
    V["LEVEL_PCTS"] = L["level_pcts"]; V["RANKER"] = L["ranker"]

    K = cfg["KWB"]
    V["KWB_K"] = K["k"]; V["KWB_metric"] = K["metric"]; V["KWB_eps"] = K["eps"]
    V["KWB_use_faiss"] = K.get("use_faiss", True)
    V["KWB_faiss_gpu"] = K.get("faiss_gpu", True)

    G = cfg["GWB"]
    V["GWB_K"] = G["k"]
    V["GWB_metric"] = G["metric"]
    V["GWB_eps"] = G["eps"]
    V["GWB_mode"] = G["mode"]
    V["GWB_bandwidth"] = G["bandwidth"]
    V["GWB_bandwidth_scale"] = G["bandwidth_scale"]
    V["GWB_use_faiss"] = G["use_faiss"]
    V["GWB_faiss_gpu"] = G["faiss_gpu"]

    S = cfg["S3WD"]
    V["S3_c1"]=S["c1"]; V["S3_c2"]=S["c2"]; V["S3_xi_min"]=S["xi_min"]
    V["S3_theta_pos"]=S["theta_pos"]; V["S3_theta_neg"]=S["theta_neg"]
    # —— 统一命名输出 —— #
    V["S3_sigma"]=S["sigma"]
    V["S3_regret_mode"]=S.get("regret_mode","utility")
    V["S3_penalty_large"]=S["penalty_large"]; V["S3_gamma_last"]=S["gamma_last"]; V["S3_gap"]=S.get("gap",0.02)

    P = cfg["PSO"]
    V["PSO_particles"]=P["particles"]; V["PSO_iters"]=P["iters"]
    V["PSO_w_max"]=P["w_max"]; V["PSO_w_min"]=P["w_min"]
    V["PSO_c1"]=P["c1"]; V["PSO_c2"]=P["c2"]; V["PSO_seed"]=P["seed"]
    V["PSO_use_gpu"]=P.get("use_gpu", True)

    if "DYN" in cfg:
        Y = cfg["DYN"]
        V["DYN_strategy"] = Y["strategy"]
        V["DYN_step"] = Y["step"]
        V["DYN_window_size"] = Y.get("window_size")
        V["DYN_target_bnd"] = Y["target_bnd"]
        V["DYN_ema_alpha"] = Y.get("ema_alpha")
        V["DYN_median_window"] = Y.get("median_window")
        V["DYN_keep_gap"] = Y.get("keep_gap")
        V["DYN_fallback_rule"] = Y.get("fallback_rule")
        V["DYN_gamma_last"] = Y.get("gamma_last")
        V["DYN_stall_rounds"] = Y.get("stall_rounds")

    if "DRIFT" in cfg:
        R = cfg["DRIFT"]
        V["DRIFT_method"] = R["method"]
        V["DRIFT_window_size"] = R["window_size"]
        V["DRIFT_stat_size"] = R["stat_size"]
        V["DRIFT_significance"] = R.get("significance")
        V["DRIFT_delta"] = R.get("delta")
        V["DRIFT_cooldown"] = R.get("cooldown")
        V["DRIFT_min_window_length"] = R.get("min_window_length")

    if "INCR" in cfg:
        I = cfg["INCR"]
        V["INCR_buffer_size"] = I["buffer_size"]
        V["INCR_cache_strategy"] = I["cache_strategy"]
        V["INCR_rebuild_interval"] = I["rebuild_interval"]
        V["INCR_min_rebuild_interval"] = I.get("min_rebuild_interval")
        V["INCR_drift_shrink"] = I.get("drift_shrink")
        V["INCR_immediate_rebuild_methods"] = I.get("immediate_rebuild_methods")
        V["INCR_enable_faiss_append"] = I.get("enable_faiss_append")
        V["INCR_random_state"] = I.get("random_state")
    return V

def show_cfg(cfg: dict) -> None:
    print("【配置快照】")
    for grp in ["DATA","LEVEL","KWB","GWB","S3WD","PSO","DYN","DRIFT","INCR"]:
        if grp in cfg:
            print(f"- {grp}: {cfg[grp]}")


def extract_dynamic_configs(
    cfg: dict,
) -> Tuple[
    Optional[DynamicLoopConfig],
    Optional[DriftDetectorConfig],
    Optional[PosteriorUpdaterConfig],
]:
    """提取 DYN/DRIFT/INCR 配置并实例化对应 dataclass。"""

    def _compact(section: Optional[dict]) -> Optional[dict]:
        if not section:
            return None
        return {k: v for k, v in section.items() if v is not None}

    dyn = _compact(cfg.get("DYN"))
    drift = _compact(cfg.get("DRIFT"))
    incr = _compact(cfg.get("INCR"))

    dyn_cfg = DynamicLoopConfig(**dyn) if dyn else None
    drift_cfg = DriftDetectorConfig(**drift) if drift else None
    incr_cfg = PosteriorUpdaterConfig(**incr) if incr else None
    return dyn_cfg, drift_cfg, incr_cfg

