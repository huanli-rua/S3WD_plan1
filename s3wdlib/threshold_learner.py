from __future__ import annotations

"""Global threshold learning utilities for the v3 workflow."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class ThresholdSummary:
    alpha_candidates: np.ndarray
    beta_candidates: np.ndarray
    scores: np.ndarray
    best_alpha: float
    best_beta: float
    metrics_per_combo: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_candidates": self.alpha_candidates.tolist(),
            "beta_candidates": self.beta_candidates.tolist(),
            "scores": self.scores.tolist(),
            "best_alpha": float(self.best_alpha),
            "best_beta": float(self.best_beta),
            "metrics_per_combo": self.metrics_per_combo,
        }


def _quantile_candidates(values: np.ndarray, percents: Iterable[float]) -> np.ndarray:
    percents = list(percents or [])
    if not percents:
        return np.asarray([], dtype=float)
    return np.percentile(values, percents)


def _safe_divide(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def _compute_year_metrics(
    p: np.ndarray,
    y_true: np.ndarray,
    year: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[pd.DataFrame, float, float, float, float]:
    pos_mask = p >= alpha
    neg_mask = p <= beta
    bnd_mask = ~(pos_mask | neg_mask)
    pred = np.zeros_like(y_true)
    pred[pos_mask] = 1
    pred[bnd_mask] = 0

    data = []
    unique_years = np.unique(year)
    for yr in unique_years:
        mask = year == yr
        y_year = y_true[mask]
        pred_year = pred[mask]
        bnd_year = bnd_mask[mask]

        tp = np.logical_and(y_year == 1, pred_year == 1).sum()
        fp = np.logical_and(y_year == 0, pred_year == 1).sum()
        fn = np.logical_and(y_year == 1, pred_year == 0).sum()
        tn = np.logical_and(y_year == 0, pred_year == 0).sum()

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        tpr = recall
        tnr = _safe_divide(tn, tn + fp)
        bac = 0.5 * (tpr + tnr)
        bnd_ratio = _safe_divide(bnd_year.sum(), bnd_year.size)
        data.append(
            {
                "year": int(yr),
                "F1": f1,
                "BAC": bac,
                "BND_ratio": bnd_ratio,
            }
        )

    df = pd.DataFrame(data)
    mean_F1 = df["F1"].mean() if not df.empty else 0.0
    mean_BAC = df["BAC"].mean() if not df.empty else 0.0
    mean_BND = df["BND_ratio"].mean() if not df.empty else 0.0
    var_F1 = df["F1"].var(ddof=0) if len(df) > 1 else 0.0
    return df, mean_F1, mean_BAC, mean_BND, var_F1


def learn_global_thresholds_gwb(
    p_val: np.ndarray,
    y_val: np.ndarray,
    year_val: np.ndarray,
    cfg_th: dict,
) -> tuple[float, float, ThresholdSummary]:
    p_val = np.asarray(p_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)
    year_val = np.asarray(year_val, dtype=int)
    if not (p_val.shape == y_val.shape == year_val.shape):
        raise ValueError("p_val, y_val, year_val 必须长度一致。")

    beta_cands = _quantile_candidates(p_val, cfg_th.get("beta_quantiles", []))
    alpha_cands = _quantile_candidates(p_val, cfg_th.get("alpha_quantiles", []))
    gap_min = float(cfg_th.get("gap_min", 0.0))

    lambda_F1 = float(cfg_th.get("lambda_F1", 1.0))
    lambda_BAC = float(cfg_th.get("lambda_BAC", 0.5))
    mu_BND = float(cfg_th.get("mu_BND", 1.0))
    gamma_var = float(cfg_th.get("gamma_var", 0.5))
    target_BND = float(cfg_th.get("target_BND", 0.2))
    bnd_min = float(cfg_th.get("BND_min", 0.0))
    bnd_max = float(cfg_th.get("BND_max", 1.0))

    scores = np.full((len(alpha_cands), len(beta_cands)), -np.inf, dtype=float)
    metrics_rows: list[dict[str, Any]] = []

    best_alpha = float(alpha_cands[0]) if len(alpha_cands) else 0.5
    best_beta = float(beta_cands[0]) if len(beta_cands) else 0.0
    best_score = -np.inf

    for i, alpha in enumerate(alpha_cands):
        for j, beta in enumerate(beta_cands):
            if alpha < beta + gap_min:
                continue
            df_metrics, mean_F1, mean_BAC, mean_BND, var_F1 = _compute_year_metrics(
                p_val, y_val, year_val, alpha, beta
            )
            if not (bnd_min <= mean_BND <= bnd_max):
                continue
            score = (
                lambda_F1 * mean_F1
                + lambda_BAC * mean_BAC
                - mu_BND * abs(mean_BND - target_BND)
                - gamma_var * var_F1
            )
            scores[i, j] = score
            metrics_rows.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "score": score,
                    "mean_F1": mean_F1,
                    "mean_BAC": mean_BAC,
                    "mean_BND": mean_BND,
                    "var_F1": var_F1,
                    "metrics_per_year": df_metrics.to_dict(orient="records"),
                }
            )
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_beta = float(beta)

    summary = ThresholdSummary(
        alpha_candidates=alpha_cands,
        beta_candidates=beta_cands,
        scores=scores,
        best_alpha=best_alpha,
        best_beta=best_beta,
        metrics_per_combo=pd.DataFrame(metrics_rows),
    )
    return best_alpha, best_beta, summary


def plot_threshold_grid(summary: ThresholdSummary, *, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt

    alpha = summary.alpha_candidates
    beta = summary.beta_candidates
    scores = summary.scores

    if scores.size == 0:
        print("No valid α-β grid points to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        scores,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[beta.min() if len(beta) else 0, beta.max() if len(beta) else 1,
                alpha.min() if len(alpha) else 0, alpha.max() if len(alpha) else 1],
    )
    ax.set_xlabel("β")
    ax.set_ylabel("α")
    ax.set_title("Global threshold grid scores")
    plt.colorbar(im, ax=ax, label="score")
    ax.scatter([summary.best_beta], [summary.best_alpha], color="red", marker="x")
    plt.tight_layout()
    plt.show()
