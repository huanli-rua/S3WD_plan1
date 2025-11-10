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
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff
import csv


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
    if "CRSDepTime" in enriched.columns:
        enriched["dep_hour"] = enriched["CRSDepTime"].apply(lambda v: _time_to_minutes(v) // 60)
    if "CRSArrTime" in enriched.columns:
        enriched["arr_hour"] = enriched["CRSArrTime"].apply(lambda v: _time_to_minutes(v) // 60)
    if "CRSDepTime" in enriched.columns and "CRSArrTime" in enriched.columns:
        dep_min = enriched["CRSDepTime"].apply(_time_to_minutes)
        arr_min = enriched["CRSArrTime"].apply(_time_to_minutes)
        block = (arr_min - dep_min) % (24 * 60)
        enriched["block_time_min"] = block
    return enriched
