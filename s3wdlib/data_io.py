from __future__ import annotations
# -*- coding: utf-8 -*-
"""
数据读写与基础预处理
- 支持 CSV / ARFF 自动识别
- 支持已有二分类标签列，或对连续列按阈值二值化
- 将字符串特征转换为整数编码；清理缺失与无穷值
- 提供 MinMax 归一化（论文推荐步骤）
"""
import os
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff
import csv

from .encoders import SimpleLabelEncoder


def _time_to_minutes(val) -> int:
    """将 HHMM 时间编码转换为分钟，输入可能为字符串或数值。"""

    if pd.isna(val):
        return 0
    try:
        iv = int(float(val))
    except (TypeError, ValueError):
        return 0
    hour = iv // 100
    minute = iv % 100
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    return hour * 60 + minute

# ---------------------------------------------------------------------------
# 数据加载与预处理
# ---------------------------------------------------------------------------


def load_table_auto(path: str,
                    label_col: Optional[str|int]=None,
                    positive_label=1,
                    continuous_label: Optional[str]=None,
                    threshold: Optional[float]=None,
                    threshold_op: str = ">=") -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载数据并返回 (X, y).
    用法三选一：
      1) 直接使用已有二分类列（label_col 指向该列，或自动匹配 'label/class/target/y'）。
      2) 对连续列进行阈值二值化（continuous_label + threshold + threshold_op）。
      3) 未指定时，默认用最后一列作为标签。
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".arff":
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        # ARFF 中 nominal 常以 bytes 存储，这里统一转为 str
        for c in df.columns:
            if df[c].dtype == object and len(df[c])>0 and isinstance(df[c].iloc[0], (bytes, bytearray)):
                df[c] = df[c].apply(lambda b: b.decode("utf-8"))
    else:
        # —— CSV/文本：自动识别分隔符（优先 sniff，其次尝试 ; / ,）——
        try:
            # 先用 csv.Sniffer 猜分隔符
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t','|'])
            sep_guess = dialect.delimiter
            df = pd.read_csv(path, sep=sep_guess)
        except Exception:
            # 保底：依次尝试 ; 和 ,，最后退回 pandas 自动
            try:
                df = pd.read_csv(path, sep=';')
            except Exception:
                try:
                    df = pd.read_csv(path, sep=',')
                except Exception:
                    # 终极兜底：让 pandas 自己推断（python 引擎更宽松）
                    df = pd.read_csv(path, sep=None, engine='python')


    # 情况B：连续列阈值二值化
    if continuous_label is not None:
        if threshold is None:
            raise ValueError("使用 continuous_label 时必须提供 threshold（阈值）。")
        if continuous_label not in df.columns:
            raise ValueError(f"指定的连续列 continuous_label='{continuous_label}' 不存在。")
        y_cont = pd.to_numeric(df[continuous_label], errors="coerce")
        if   threshold_op == ">=": y = (y_cont >= threshold).astype(int)
        elif threshold_op ==  ">": y = (y_cont >  threshold).astype(int)
        elif threshold_op == "<=": y = (y_cont <= threshold).astype(int)
        elif threshold_op ==  "<": y = (y_cont <  threshold).astype(int)
        else:
            raise ValueError("threshold_op 取值应为: '>=','>','<=','<'")
        X = df.drop(columns=[continuous_label]).copy()
    else:
        # 情况A：已有二分类标签列
        if label_col is None:
            for cand in ["label","class","target","y"]:
                if cand in df.columns:
                    label_col = cand
                    break
            if label_col is None:
                label_col = df.columns[-1]

        y_raw = df[label_col]
        X = df.drop(columns=[label_col]).copy()

        # 标签二值化到 {0,1}
        y = y_raw.copy()
        if y.dtype == object:
            mapping = {"1":1,"0":0,"yes":1,"no":0,"true":1,"false":0}
            y = y.str.lower().map(mapping).fillna(y)
        if y.dtype == object:
            vc = y.value_counts()
            if str(positive_label) in vc.index.astype(str):
                y = (y.astype(str) == str(positive_label)).astype(int)
            else:
                rare = vc.index[-1] if len(vc)>1 else vc.index[0]
                y = (y == rare).astype(int)
        else:
            uniq = sorted(pd.unique(y))
            if len(uniq) != 2:
                raise ValueError("当前脚手架仅支持二分类（或请使用 continuous_label+threshold 方式）。")
            y = (y == positive_label).astype(int)

    # 字符特征编码；缺失/无穷清理
    categorical_features: list[str] = []
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes
            categorical_features.append(c)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y = y.loc[X.index]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X.attrs["categorical_features"] = categorical_features

    print(f"【数据加载完毕】样本数={len(X)}，特征数={X.shape[1]}，正类比例={float(y.mean()):.4f}")
    return X, y


def assign_year_from_month_sequence(
    df: pd.DataFrame,
    *,
    start_year: Optional[int] = None,
    month_candidates: Iterable[str] = ("Month", "month", "MONTH"),
    year_candidates: Iterable[str] = ("Year", "year", "YEAR"),
    copy: bool = True,
) -> pd.DataFrame:
    """Ensure the dataframe contains an integer ``Year`` column.

    When the original data lacks a ``Year`` column, the function generates
    consecutive years using ``start_year`` and a monotonically increasing
    month sequence (rolling over when the month value decreases).

    Parameters
    ----------
    df:
        Source dataframe containing at least a month column.
    start_year:
        Base year used when the dataframe does not already contain year
        information.  Must be provided in that case.
    month_candidates / year_candidates:
        Candidate column names to locate month/year fields.
    copy:
        Whether to operate on a copy of the dataframe.
    """

    month_col = next((c for c in month_candidates if c in df.columns), None)
    if month_col is None:
        raise ValueError("数据集缺少 Month 列，无法基于月份构造年份信息。")

    result = df.copy() if copy else df
    months = pd.to_numeric(result[month_col], errors="coerce")
    if months.isna().all():
        raise ValueError("Month 列无法转换为数值，无法构造年份信息。")
    months = (
        months.fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(1)
        .astype(int)
        .clip(1, 12)
    )
    result[month_col] = months
    if month_col != "Month":
        result["Month"] = months

    year_col = next((c for c in year_candidates if c in result.columns), None)
    if year_col is not None:
        years = pd.to_numeric(result[year_col], errors="coerce")
        if years.isna().any():
            raise ValueError("Year 列包含无法转换为整数的值。")
        result["Year"] = years.astype(int)
    else:
        if start_year is None:
            raise ValueError("数据集缺少 Year 列，且未提供 start_year 用于生成年份。")
        month_values = months.to_numpy(dtype=int, copy=False)
        if month_values.size == 0:
            raise ValueError("数据集为空，无法生成年份序列。")
        base_year = int(start_year)
        year_offset = 0
        prev_month = month_values[0]
        year_values: list[int] = []
        for idx, month_val in enumerate(month_values):
            if idx > 0 and month_val < prev_month:
                year_offset += 1
            prev_month = month_val
            year_values.append(base_year + year_offset)
        result["Year"] = year_values
        result["Year_synth"] = year_values

    return result


def add_time_columns(df: pd.DataFrame,
                     dep_time_col: str,
                     season_col: str,
                     dep_slot_col: str,
                     dep_slot_def: dict,
                     start_year: int,
                     sort_by_time: bool = True) -> pd.DataFrame:
    """Add year/month/season and departure slot columns to the dataframe."""

    if df is None:
        raise ValueError("df 不能为空。")
    if dep_time_col not in df.columns:
        raise KeyError(f"数据缺少指定的出发时间列: {dep_time_col}")
    if dep_slot_def is None or not isinstance(dep_slot_def, dict):
        raise ValueError("dep_slot_def 必须是包含 ranges 的字典。")
    mode = dep_slot_def.get("mode", "by_hour_ranges")
    if mode != "by_hour_ranges":
        raise ValueError(f"暂不支持的 dep_slot_def.mode: {mode}")
    ranges = dep_slot_def.get("ranges")
    if not isinstance(ranges, (list, tuple)) or len(ranges) == 0:
        raise ValueError("dep_slot_def['ranges'] 必须是非空列表。")

    enriched = df.copy()
    enriched = assign_year_from_month_sequence(
        enriched,
        start_year=start_year,
        copy=False,
    )

    dep_minutes = enriched[dep_time_col].apply(_time_to_minutes)
    dep_hours = (dep_minutes // 60).astype(int)

    dep_slots = np.full(len(enriched), -1, dtype=int)
    for idx, hour_range in enumerate(ranges):
        if not isinstance(hour_range, (list, tuple)) or len(hour_range) != 2:
            raise ValueError("每个时段范围必须是长度为 2 的 [start, end) 序列。")
        start, end = hour_range
        start = int(start)
        end = int(end)
        mask = (dep_hours >= start) & (dep_hours < end)
        dep_slots[mask.to_numpy(dtype=bool, copy=False)] = idx

    enriched[dep_slot_col] = dep_slots.astype(int)
    enriched["year"] = enriched["Year"].astype(int)
    enriched["month"] = enriched["Month"].astype(int)
    enriched[season_col] = (
        enriched["year"].astype(str)
        + "-"
        + enriched["month"].map(lambda m: f"{int(m):02d}")
    )

    if sort_by_time:
        enriched["_dep_minutes_tmp"] = dep_minutes
        enriched = (
            enriched.sort_values(by=["year", "month", "_dep_minutes_tmp"]).reset_index(drop=True)
        )
        enriched = enriched.drop(columns=["_dep_minutes_tmp"])

    return enriched


def fit_simple_label_encoders(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, SimpleLabelEncoder]:
    """Fit label encoders on the specified categorical columns."""

    encoders: dict[str, SimpleLabelEncoder] = {}
    for col in categorical_cols:
        if col not in df.columns:
            raise KeyError(f"数据缺少类别列: {col}")
        encoder = SimpleLabelEncoder()
        encoder.fit(df[col])
        encoders[col] = encoder
    return encoders


def apply_simple_label_encoders(
    df: pd.DataFrame,
    encoders: dict[str, SimpleLabelEncoder],
    *,
    suffix: str = "_enc",
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply fitted encoders to dataframe columns and append encoded versions."""

    target = df if inplace else df.copy()
    for col, encoder in encoders.items():
        if col not in target.columns:
            raise KeyError(f"数据缺少类别列: {col}")
        target[f"{col}{suffix}"] = encoder.transform(target[col])
    return target

def minmax_scale_fit_transform(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    """对训练集拟合 MinMaxScaler，并同步变换测试集。"""
    scaler = MinMaxScaler()
    Xtr2 = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    Xte2 = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
    print("【归一化】已对训练/测试集进行 MinMax 缩放到 [0,1]。")
    return Xtr2, Xte2, scaler


def augment_airline_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加航空运输常用的派生时间特征，保持输入副本不变。"""

    if df is None or df.empty:
        return df

    enriched = df.copy()

    # ========================
    # 1）出发时间相关特征
    # ========================
    if "CRSDepTime" in enriched.columns:
        # dep_hour：计划起飞小时（0~23）
        enriched["dep_hour"] = enriched["CRSDepTime"].apply(
            lambda v: _time_to_minutes(v) // 60
        )

        # dep_block：按业务时段划分的出发时段块（方案一）
        # 0: 深夜/维护   00:00 - 04:59
        # 1: 早高峰     05:00 - 09:59
        # 2: 午间平峰   10:00 - 15:59
        # 3: 晚高峰     16:00 - 19:59
        # 4: 晚间末班   20:00 - 23:59
        def _hour_to_block(h):
            if pd.isna(h):
                return -1  # 缺失值单独标记，不参与正常桶
            h = int(h) % 24
            if 0 <= h < 5:
                return 0
            elif 5 <= h < 10:
                return 1
            elif 10 <= h < 16:
                return 2
            elif 16 <= h < 20:
                return 3
            else:
                return 4

        enriched["dep_block"] = enriched["dep_hour"].apply(_hour_to_block).astype("int8")

    # ========================
    # 2）到达时间相关特征
    # ========================
    if "CRSArrTime" in enriched.columns:
        # arr_hour：计划到达小时（0~23）
        enriched["arr_hour"] = enriched["CRSArrTime"].apply(
            lambda v: _time_to_minutes(v) // 60
        )

    # ========================
    # 3）计划飞行时间（分钟）
    # ========================
    if "CRSDepTime" in enriched.columns and "CRSArrTime" in enriched.columns:
        dep_min = enriched["CRSDepTime"].apply(_time_to_minutes)
        arr_min = enriched["CRSArrTime"].apply(_time_to_minutes)
        block = (arr_min - dep_min) % (24 * 60)  # 考虑跨天
        enriched["block_time_min"] = block

    return enriched

