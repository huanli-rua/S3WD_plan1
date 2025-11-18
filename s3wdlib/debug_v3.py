from __future__ import annotations

import random
from pprint import pprint

import numpy as np

from .bucketizer import Bucketizer
from .config_loader import extract_vars, load_yaml_cfg
from .data_io import (
    add_time_columns,
    apply_simple_label_encoders,
    fit_simple_label_encoders,
    load_table_auto,
)
from .gwb_estimator import GWBProbEstimator
from .threshold_learner import learn_global_thresholds_gwb, plot_threshold_grid


def debug_v3_time_columns(cfg_path: str = "configs/s3wd_v3.yaml") -> None:
    """Quick helper to verify v3 time preprocessing logic."""

    cfg = load_yaml_cfg(cfg_path)
    vars_dict = extract_vars(cfg)
    data_cfg = cfg.get("DATA", {})

    X, _ = load_table_auto(
        vars_dict["data_path"],
        label_col=vars_dict["label_col"],
        positive_label=vars_dict["positive_label"],
    )

    enriched = add_time_columns(
        X,
        dep_time_col=data_cfg["dep_time_col"],
        season_col=data_cfg["season_col"],
        dep_slot_col=data_cfg["dep_slot_col"],
        dep_slot_def=data_cfg["dep_slot_def"],
        start_year=data_cfg["start_year"],
        sort_by_time=data_cfg.get("sort_by_time", True),
    )

    print(f"总行数: {len(enriched):,}")
    print(
        f"年份范围: {int(enriched['year'].min())} - {int(enriched['year'].max())}"
    )
    slot_counts = enriched[data_cfg["dep_slot_col"]].value_counts().sort_index()
    print("时段分布 (dep_slot -> count):")
    pprint(slot_counts.to_dict())
    print("前 5 行 season_col/dep_slot:")
    print(enriched[[data_cfg["season_col"], data_cfg["dep_slot_col"]]].head())

def debug_v3_buckets(cfg_path: str = "configs/s3wd_v3.yaml", sample_bucket_count: int = 5) -> None:
    """Quick helper to validate bucket structure and encoders for v3."""

    cfg = load_yaml_cfg(cfg_path)
    vars_dict = extract_vars(cfg)
    data_cfg = cfg.get("DATA", {})
    bucket_cfg = cfg.get("BUCKET", {})
    split_cfg = cfg.get("SPLIT", {})

    X, y = load_table_auto(
        vars_dict["data_path"],
        label_col=vars_dict["label_col"],
        positive_label=vars_dict["positive_label"],
    )
    X[data_cfg["label_col"]] = y

    enriched = add_time_columns(
        X,
        dep_time_col=data_cfg["dep_time_col"],
        season_col=data_cfg["season_col"],
        dep_slot_col=data_cfg["dep_slot_col"],
        dep_slot_def=data_cfg["dep_slot_def"],
        start_year=data_cfg["start_year"],
        sort_by_time=data_cfg.get("sort_by_time", True),
    )

    train_mask = (
        (enriched["year"] >= split_cfg["train_start_year"])
        & (enriched["year"] <= split_cfg["train_end_year"])
    )
    train0 = enriched.loc[train_mask]
    if train0.empty:
        raise ValueError("Train₀ 数据为空，请检查年份划分。")

    categorical_cols = list(data_cfg.get("categorical_cols") or [])
    encoders = fit_simple_label_encoders(train0, categorical_cols)
    enriched = apply_simple_label_encoders(enriched, encoders, inplace=True)
    train0 = enriched.loc[train_mask].copy()

    feature_cols = list(data_cfg.get("continuous_cols") or [])
    feature_cols += [f"{col}_enc" for col in categorical_cols]

    bucketizer = Bucketizer(
        fine_keys=bucket_cfg["fine_keys"],
        coarse_keys_level1=bucket_cfg["coarse_keys_level1"],
        min_fine_size=bucket_cfg["min_fine_bucket_size"],
        min_level1_size=bucket_cfg["min_level1_bucket_size"],
        airport_col=data_cfg["airport_col"],
        dep_slot_col=data_cfg["dep_slot_col"],
        feature_cols=feature_cols,
    )
    bucketizer.build_from_train0(train0)

    print(f"Fine 桶数量: {len(bucketizer.fine_mapping)}")
    print(f"粗桶数量: {len(bucketizer.coarse_mapping)}")

    bucket_items = list(bucketizer.bucket_sizes.items())
    if bucket_items:
        random.shuffle(bucket_items)
        for bucket_id, size in bucket_items[: min(sample_bucket_count, len(bucket_items))]:
            print(f"桶 {bucket_id}: 样本数={size}")

    sample_rows = train0.sample(n=min(3, len(train0)), random_state=42)
    for idx, row in sample_rows.iterrows():
        bucket_id = bucketizer.assign_bucket(row)
        airport = row[data_cfg["airport_col"]]
        slot = row[data_cfg["dep_slot_col"]]
        print(f"样本 idx={idx} -> {airport}/{slot} 分配到 {bucket_id}")


def debug_v3_gwb_and_threshold(cfg_path: str = "configs/s3wd_v3.yaml") -> None:
    cfg = load_yaml_cfg(cfg_path)
    vars_dict = extract_vars(cfg)
    data_cfg = cfg.get("DATA", {})
    bucket_cfg = cfg.get("BUCKET", {})
    split_cfg = cfg.get("SPLIT", {})
    gwb_cfg = cfg.get("GWB", {})
    th_cfg = cfg.get("THRESHOLD_LEARN", {})

    X, y = load_table_auto(
        vars_dict["data_path"],
        label_col=vars_dict["label_col"],
        positive_label=vars_dict["positive_label"],
    )
    X[data_cfg["label_col"]] = y

    enriched = add_time_columns(
        X,
        dep_time_col=data_cfg["dep_time_col"],
        season_col=data_cfg["season_col"],
        dep_slot_col=data_cfg["dep_slot_col"],
        dep_slot_def=data_cfg["dep_slot_def"],
        start_year=data_cfg["start_year"],
        sort_by_time=data_cfg.get("sort_by_time", True),
    )

    train_mask = (
        (enriched["year"] >= split_cfg["train_start_year"])
        & (enriched["year"] <= split_cfg["train_end_year"])
    )
    val_mask = (
        (enriched["year"] >= split_cfg["val_start_year"])
        & (enriched["year"] <= split_cfg["val_end_year"])
    )

    train0 = enriched.loc[train_mask].copy()
    val0 = enriched.loc[val_mask].copy()
    if train0.empty or val0.empty:
        raise ValueError("Train₀ 或 Val₀ 数据为空，请检查年份划分。")

    categorical_cols = list(data_cfg.get("categorical_cols") or [])
    encoders = fit_simple_label_encoders(train0, categorical_cols)
    enriched = apply_simple_label_encoders(enriched, encoders, inplace=True)
    train0 = enriched.loc[train_mask].copy()
    val0 = enriched.loc[val_mask].copy()

    feature_cols = list(data_cfg.get("continuous_cols") or [])
    categorical_feature_cols = [f"{col}_enc" for col in categorical_cols]
    feature_cols += categorical_feature_cols

    bucketizer = Bucketizer(
        fine_keys=bucket_cfg["fine_keys"],
        coarse_keys_level1=bucket_cfg["coarse_keys_level1"],
        min_fine_size=bucket_cfg["min_fine_bucket_size"],
        min_level1_size=bucket_cfg["min_level1_bucket_size"],
        airport_col=data_cfg["airport_col"],
        dep_slot_col=data_cfg["dep_slot_col"],
        feature_cols=feature_cols,
    )
    bucketizer.build_from_train0(train0)

    def _cat_values(df_slice):
        if not categorical_feature_cols:
            return None
        return df_slice[categorical_feature_cols].to_numpy(dtype=int, copy=False)

    gwb_models: dict[str, GWBProbEstimator] = {}
    for bucket_id in bucketizer.bucket_sizes:
        mask = bucketizer._mask_for_bucket(train0, bucket_id)
        bucket_df = train0.loc[mask]
        if bucket_df.empty:
            continue
        estimator = GWBProbEstimator(gwb_cfg)
        estimator.fit(
            bucket_df[feature_cols].to_numpy(dtype=float, copy=False),
            bucket_df[data_cfg["label_col"]].to_numpy(copy=False),
            categorical_values=_cat_values(bucket_df),
        )
        gwb_models[bucket_id] = estimator

    global_estimator = GWBProbEstimator(gwb_cfg)
    global_estimator.fit(
        train0[feature_cols].to_numpy(dtype=float, copy=False),
        train0[data_cfg["label_col"]].to_numpy(copy=False),
        categorical_values=_cat_values(train0),
    )

    p_list: list[float] = []
    y_list: list[int] = []
    year_list: list[int] = []

    for _, row in val0.iterrows():
        bucket_id = bucketizer.assign_bucket(row)
        estimator = gwb_models.get(bucket_id, global_estimator)
        x = bucketizer.extract_features(row).reshape(1, -1)
        cat_vals = None
        if categorical_feature_cols:
            cat_vals = np.asarray([[row[col] for col in categorical_feature_cols]], dtype=int)
        prob = float(estimator.predict_proba(x, cat_vals)[0])
        p_list.append(prob)
        y_list.append(int(row[data_cfg["label_col"]]))
        year_list.append(int(row["year"]))

    p_val = np.asarray(p_list, dtype=float)
    y_val = np.asarray(y_list, dtype=int)
    year_val = np.asarray(year_list, dtype=int)

    alpha_base, beta_base, summary = learn_global_thresholds_gwb(
        p_val, y_val, year_val, th_cfg
    )
    print(f"alpha_base={alpha_base:.4f}, beta_base={beta_base:.4f}")
    try:
        plot_threshold_grid(summary)
    except Exception as exc:
        print(f"plot_threshold_grid 失败: {exc}")


if __name__ == "__main__":
    debug_v3_time_columns()
