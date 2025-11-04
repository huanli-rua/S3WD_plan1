# -*- coding: utf-8 -*-
from __future__ import annotations
import os, yaml

# 支持分组与扁平配置；本项目统一推荐扁平键名：
#   S3_sigma: 3.0
#   S3_regret_mode: utility
GROUPS_REQUIRED = {"DATA", "LEVEL", "KWB", "GWB", "S3WD", "PSO"}
GROUPS_OPTIONAL = {"DYN", "DRIFT", "INCR"}
GROUPS = GROUPS_REQUIRED | GROUPS_OPTIONAL

DYN_DEFAULTS = {
    "enabled": True,
    "strategy": "windowed_pso",
    "step": 1,
    "target_bnd": 0.12,
    "ema_alpha": 0.6,
    "median_window": 3,
    "keep_gap": None,
    "window_size": None,
    "stall_rounds": 6,
    "fallback_rule": True,
}

DRIFT_DEFAULTS = {
    "enabled": False,
    "strategy": "kswin",
    "window_size": 200,
    "stat_size": 60,
    "significance": 0.05,
    "delta": 0.002,
    "cooldown": 0,
    "min_window_length": 5,
}

INCR_DEFAULTS = {
    "enabled": False,
    "strategy": "rolling",
    "step": 1,
    "buffer_size": 2048,
    "max_samples": None,
}

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
        "window_mode": raw.get("PSO_window_mode", False),
        "window_size": raw.get("PSO_window_size"),
        "ema_alpha": raw.get("PSO_ema_alpha"),
        "median_window": raw.get("PSO_median_window"),
        "keep_gap": raw.get("PSO_keep_gap"),
        "fallback_rule": raw.get("PSO_fallback_rule", True),
        "stall_rounds": raw.get("PSO_stall_rounds"),
    }

    D["DYN"] = {
        "enabled": raw.get("DYN_enabled", True),
        "strategy": raw.get("DYN_strategy"),
        "step": raw.get("DYN_step"),
        "target_bnd": raw.get("DYN_target_bnd"),
        "ema_alpha": raw.get("DYN_ema_alpha"),
        "median_window": raw.get("DYN_median_window"),
        "keep_gap": raw.get("DYN_keep_gap"),
        "window_size": raw.get("DYN_window_size"),
        "stall_rounds": raw.get("DYN_stall_rounds"),
        "fallback_rule": raw.get("DYN_fallback_rule"),
    }

    D["DRIFT"] = {
        "enabled": raw.get("DRIFT_enabled", True),
        "strategy": raw.get("DRIFT_strategy"),
        "window_size": raw.get("DRIFT_window_size"),
        "stat_size": raw.get("DRIFT_stat_size"),
        "significance": raw.get("DRIFT_significance"),
        "delta": raw.get("DRIFT_delta"),
        "cooldown": raw.get("DRIFT_cooldown"),
        "min_window_length": raw.get("DRIFT_min_window_length"),
    }

    D["INCR"] = {
        "enabled": raw.get("INCR_enabled", True),
        "strategy": raw.get("INCR_strategy"),
        "step": raw.get("INCR_step"),
        "buffer_size": raw.get("INCR_buffer_size"),
        "max_samples": raw.get("INCR_max_samples"),
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

    missing = [g for g in GROUPS_REQUIRED if g not in cfg]
    if missing:
        raise KeyError(f"YAML 缺少分组: {missing}，必须包含 {sorted(GROUPS)}")

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

    # 动态组件默认值
    for grp_name, defaults in (
        ("DYN", DYN_DEFAULTS),
        ("DRIFT", DRIFT_DEFAULTS),
        ("INCR", INCR_DEFAULTS),
    ):
        grp_val = cfg.get(grp_name)
        if grp_val is None:
            grp_val = {}
        for key, value in defaults.items():
            grp_val.setdefault(key, value)
        cfg[grp_name] = grp_val

    if cfg.get("DYN"):
        _require(cfg["DYN"], "DYN", ["strategy", "step", "target_bnd"])
    if cfg.get("DRIFT"):
        _require(cfg["DRIFT"], "DRIFT", ["strategy"])
    if cfg.get("INCR"):
        _require(cfg["INCR"], "INCR", ["strategy", "step"])

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
    V["PSO_window_mode"]=P.get("window_mode", False)
    V["PSO_window_size"]=P.get("window_size")
    V["PSO_ema_alpha"]=P.get("ema_alpha")
    V["PSO_median_window"]=P.get("median_window")
    V["PSO_keep_gap"]=P.get("keep_gap")
    V["PSO_fallback_rule"]=P.get("fallback_rule", True)
    V["PSO_stall_rounds"]=P.get("stall_rounds")

    DY = cfg.get("DYN", {})
    V["DYN_enabled"] = DY.get("enabled", True)
    V["DYN_strategy"] = DY.get("strategy")
    V["DYN_step"] = DY.get("step")
    V["DYN_target_bnd"] = DY.get("target_bnd")
    V["DYN_ema_alpha"] = DY.get("ema_alpha")
    V["DYN_median_window"] = DY.get("median_window")
    V["DYN_keep_gap"] = DY.get("keep_gap")
    V["DYN_window_size"] = DY.get("window_size")
    V["DYN_stall_rounds"] = DY.get("stall_rounds")
    V["DYN_fallback_rule"] = DY.get("fallback_rule", True)

    DR = cfg.get("DRIFT", {})
    V["DRIFT_enabled"] = DR.get("enabled", True)
    V["DRIFT_strategy"] = DR.get("strategy")
    V["DRIFT_window_size"] = DR.get("window_size")
    V["DRIFT_stat_size"] = DR.get("stat_size")
    V["DRIFT_significance"] = DR.get("significance")
    V["DRIFT_delta"] = DR.get("delta")
    V["DRIFT_cooldown"] = DR.get("cooldown")
    V["DRIFT_min_window_length"] = DR.get("min_window_length")

    IN = cfg.get("INCR", {})
    V["INCR_enabled"] = IN.get("enabled", True)
    V["INCR_strategy"] = IN.get("strategy")
    V["INCR_step"] = IN.get("step")
    V["INCR_buffer_size"] = IN.get("buffer_size")
    V["INCR_max_samples"] = IN.get("max_samples")
    return V

def show_cfg(cfg: dict) -> None:
    print("【配置快照】")
    for grp in ["DATA","LEVEL","KWB","GWB","S3WD","PSO","DYN","DRIFT","INCR"]:
        if grp in cfg:
            print(f"- {grp}: {cfg[grp]}")

