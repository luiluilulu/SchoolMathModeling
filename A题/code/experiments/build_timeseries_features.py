#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从附件6逐秒原始数据构造窗口级时序特征。

输入（默认位于 ../problem）:
  - attachment6_window_raw_samples.csv
  - attachment7_meter_geometry.csv
  - attachment1_window_data.csv

输出（默认位于 ../output/features/timeseries）:
  - timeseries_features_full.csv       全部时序特征
  - timeseries_features_compact.csv    首轮推荐的紧凑特征集
  - merged_window_data_timeseries.csv  与附件1合并后的窗口级数据
  - timeseries_feature_diagnostics.csv 特征与物理残差的总体/组内/组间相关诊断
  - timeseries_feature_manifest.json   特征分组与参数说明

特征构造严格只使用在线可获得的逐秒测量数据；standard_volume_m3 仅用于事后诊断，
不会参与任何特征计算。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

AREA_M2 = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874], dtype=float)
N_CHORDS = 5
EPS = 1e-12
START_FRAC = 0.20
END_FRAC = 0.20

META_COLS = [
    "window_id",
    "date",
    "flow_point",
    "condition_note",
    "disturbance_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造附件6逐秒时序特征")
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "problem",
        help="附件所在目录，默认 ../problem",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "output" / "features" / "timeseries",
        help="输出目录",
    )
    parser.add_argument(
        "--start-frac",
        type=float,
        default=START_FRAC,
        help="启动段占窗口比例，默认0.20",
    )
    parser.add_argument(
        "--end-frac",
        type=float,
        default=END_FRAC,
        help="结束段占窗口比例，默认0.20",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def numeric_array(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def finite_values(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def safe_mean(x: Sequence[float]) -> float:
    a = finite_values(x)
    return float(np.mean(a)) if a.size else np.nan


def safe_std(x: Sequence[float]) -> float:
    a = finite_values(x)
    return float(np.std(a, ddof=1)) if a.size >= 2 else 0.0


def safe_mad(x: Sequence[float]) -> float:
    a = finite_values(x)
    if not a.size:
        return np.nan
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def safe_quantile(x: Sequence[float], q: float) -> float:
    a = finite_values(x)
    return float(np.quantile(a, q)) if a.size else np.nan


def relative_scale(x: Sequence[float]) -> float:
    """稳定的相对量分母，避免接近零时爆炸。"""
    a = finite_values(x)
    if not a.size:
        return np.nan
    return float(max(abs(np.mean(a)), np.median(np.abs(a)), EPS))


def safe_slope(t: Sequence[float], x: Sequence[float]) -> float:
    tt = np.asarray(t, dtype=float)
    xx = np.asarray(x, dtype=float)
    mask = np.isfinite(tt) & np.isfinite(xx)
    tt, xx = tt[mask], xx[mask]
    if len(tt) < 3 or np.ptp(tt) <= EPS:
        return 0.0
    tt = tt - np.mean(tt)
    denom = float(np.dot(tt, tt))
    return float(np.dot(tt, xx - np.mean(xx)) / denom) if denom > EPS else 0.0


def safe_lag1(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return 0.0
    x0, x1 = a[:-1], a[1:]
    if np.std(x0) <= EPS or np.std(x1) <= EPS:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def zero_cross_rate(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    centered = a - np.median(a)
    signs = np.sign(centered)
    # 将0延续为前一个非零符号，降低恰好等于中位数造成的假变号。
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def normalized_stats(t: np.ndarray, x: np.ndarray, prefix: str) -> Dict[str, float]:
    """对单个时序生成尺度无关的波动/漂移统计。"""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    scale = relative_scale(x)
    mean = safe_mean(x)
    std = safe_std(x)
    mad = safe_mad(x)
    p05 = safe_quantile(x, 0.05)
    p25 = safe_quantile(x, 0.25)
    p75 = safe_quantile(x, 0.75)
    p95 = safe_quantile(x, 0.95)
    slope = safe_slope(t, x)
    duration = float(np.ptp(t[np.isfinite(t)])) if np.isfinite(t).sum() >= 2 else 0.0
    mean_abs_step = safe_mean(np.abs(np.diff(x))) if len(x) >= 2 else 0.0

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_cv": std / scale,
        f"{prefix}_mad_rel": mad / scale,
        f"{prefix}_iqr_rel": (p75 - p25) / scale,
        f"{prefix}_p90span_rel": (p95 - p05) / scale,
        f"{prefix}_range_rel": (np.nanmax(x) - np.nanmin(x)) / scale if np.isfinite(x).any() else np.nan,
        f"{prefix}_drift_rel": slope * duration / scale,
        f"{prefix}_tv_step_rel": mean_abs_step / scale,
        f"{prefix}_lag1": safe_lag1(x),
    }


def split_stages(n: int, start_frac: float, end_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 9:
        # 极小窗口兜底：尽量等分。
        idx = np.arange(n)
        parts = np.array_split(idx, 3)
        return parts[0], parts[1], parts[2]

    n_start = max(3, int(math.ceil(n * start_frac)))
    n_end = max(3, int(math.ceil(n * end_frac)))
    if n_start + n_end > n - 3:
        n_start = max(3, n // 5)
        n_end = max(3, n // 5)
    start_idx = np.arange(0, n_start)
    end_idx = np.arange(n - n_end, n)
    plateau_idx = np.arange(n_start, n - n_end)
    return start_idx, plateau_idx, end_idx


def stage_features(x: np.ndarray, prefix: str, indices: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, float]:
    start_idx, plateau_idx, end_idx = indices
    start = x[start_idx]
    plateau = x[plateau_idx]
    end = x[end_idx]
    plateau_mean = safe_mean(plateau)
    denom = max(abs(plateau_mean), np.median(np.abs(plateau[np.isfinite(plateau)])) if np.isfinite(plateau).any() else 0.0, EPS)
    return {
        f"{prefix}_start_over_plateau": safe_mean(start) / denom - np.sign(plateau_mean),
        f"{prefix}_end_over_plateau": safe_mean(end) / denom - np.sign(plateau_mean),
        f"{prefix}_start_cv": safe_std(start) / max(relative_scale(start), EPS),
        f"{prefix}_plateau_cv": safe_std(plateau) / max(relative_scale(plateau), EPS),
        f"{prefix}_end_cv": safe_std(end) / max(relative_scale(end), EPS),
    }


def robust_spike_fraction(x: np.ndarray, threshold: float = 3.5) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 5:
        return 0.0
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad <= EPS:
        return 0.0
    robust_z = 0.67448975 * np.abs(a - med) / mad
    return float(np.mean(robust_z > threshold))


def mean_pairwise_corr(matrix: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(matrix, dtype=float)
    if x.shape[0] < 3:
        return 0.0, 0.0
    corr = np.corrcoef(x, rowvar=False)
    vals = corr[np.triu_indices_from(corr, k=1)]
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.min(vals))


def find_chord_order(chord_geo: pd.DataFrame) -> pd.DataFrame:
    """尽量按照声道0..4排序；若没有索引列，则保持原顺序。"""
    for col in ["chord_index", "channel_index", "chord_id", "channel", "name"]:
        if col not in chord_geo.columns:
            continue
        parsed = chord_geo[col].astype(str).str.extract(r"(\d+)", expand=False)
        if parsed.notna().sum() >= N_CHORDS:
            out = chord_geo.assign(_order=pd.to_numeric(parsed, errors="coerce")).sort_values("_order")
            return out.drop(columns="_order")
    return chord_geo.copy()


def build_instantaneous_velocities(raw: pd.DataFrame, geo: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = raw.copy()
    velocity_cols = [f"v_inst_{j}" for j in range(N_CHORDS)]

    # 若附件6已经给出逐秒速度列，优先直接使用。
    direct_candidates = [
        [f"chord{j}" for j in range(N_CHORDS)],
        [f"velocity_{j}" for j in range(N_CHORDS)],
        [f"v_{j}" for j in range(N_CHORDS)],
    ]
    for cols in direct_candidates:
        if all(c in out.columns for c in cols):
            for j, col in enumerate(cols):
                out[velocity_cols[j]] = pd.to_numeric(out[col], errors="coerce")
            return out, velocity_cols

    diff_a = [f"diff_ns_{j}" for j in range(N_CHORDS)]
    diff_b = [f"diff_ns_{j + N_CHORDS}" for j in range(N_CHORDS)]
    required = diff_a + diff_b
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            "附件6既没有逐秒速度列，也缺少A/B链时差列。缺失字段: " + ", ".join(missing)
        )

    if "record_type" in geo.columns:
        chord_geo = geo[geo["record_type"].astype(str).eq("chord_mapping")].copy()
    else:
        chord_geo = geo.copy()
    chord_geo = find_chord_order(chord_geo)
    if len(chord_geo) < N_CHORDS:
        raise ValueError(f"附件7中声道映射记录不足5条，实际{len(chord_geo)}条")

    coef_cols = [
        "raw_diff_gain_a_mps_per_ns",
        "raw_diff_gain_b_mps_per_ns",
        "raw_diff_intercept_mps",
    ]
    missing_coef = [c for c in coef_cols if c not in chord_geo.columns]
    if missing_coef:
        raise ValueError("附件7缺少映射系数字段: " + ", ".join(missing_coef))

    chord_geo = chord_geo.iloc[:N_CHORDS]
    gain_a = pd.to_numeric(chord_geo[coef_cols[0]], errors="raise").to_numpy(dtype=float)
    gain_b = pd.to_numeric(chord_geo[coef_cols[1]], errors="raise").to_numpy(dtype=float)
    intercept = pd.to_numeric(chord_geo[coef_cols[2]], errors="raise").to_numpy(dtype=float)

    for j in range(N_CHORDS):
        a = pd.to_numeric(out[diff_a[j]], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(out[diff_b[j]], errors="coerce").to_numpy(dtype=float)
        out[velocity_cols[j]] = gain_a[j] * a + gain_b[j] * b + intercept[j]

    return out, velocity_cols


def add_difference_features(
    row: Dict[str, float],
    t: np.ndarray,
    velocities: np.ndarray,
) -> None:
    ref = np.nanmean(np.abs(velocities), axis=1)
    ref = np.maximum(ref, EPS)
    sequences = {
        "ts_diff_top_bottom": (velocities[:, 0] - velocities[:, 4]) / ref,
        "ts_diff_inner_skew": (velocities[:, 1] - velocities[:, 3]) / ref,
        "ts_diff_center_edge": (velocities[:, 2] - 0.5 * (velocities[:, 0] + velocities[:, 4])) / ref,
        "ts_diff_edge_inner": (0.5 * (velocities[:, 0] + velocities[:, 4]) - 0.5 * (velocities[:, 1] + velocities[:, 3])) / ref,
        "ts_diff_alternating": (velocities[:, 0] - velocities[:, 1] + velocities[:, 3] - velocities[:, 4]) / ref,
    }
    for prefix, seq in sequences.items():
        stats = normalized_stats(t, seq, prefix)
        # 对差值序列，绝对均值可能接近0；保留更直观的绝对统计。
        row[f"{prefix}_mean"] = safe_mean(seq)
        row[f"{prefix}_std"] = safe_std(seq)
        row[f"{prefix}_mad"] = safe_mad(seq)
        row[f"{prefix}_p90span"] = safe_quantile(seq, 0.95) - safe_quantile(seq, 0.05)
        row[f"{prefix}_drift"] = safe_slope(t, seq) * (np.ptp(t) if len(t) >= 2 else 0.0)
        row[f"{prefix}_tv_step"] = safe_mean(np.abs(np.diff(seq))) if len(seq) >= 2 else 0.0
        row[f"{prefix}_zero_cross"] = zero_cross_rate(seq)
        row[f"{prefix}_lag1"] = stats[f"{prefix}_lag1"]


def extract_window_features(
    window: pd.DataFrame,
    velocity_cols: Sequence[str],
    start_frac: float,
    end_frac: float,
) -> Dict[str, float]:
    if "rel_time_s" not in window.columns:
        raise ValueError("附件6缺少 rel_time_s")

    sort_cols = [c for c in ["rel_time_s", "sample_index"] if c in window.columns]
    w = window.sort_values(sort_cols).copy()
    t = pd.to_numeric(w["rel_time_s"], errors="coerce").to_numpy(dtype=float)
    velocities = w[list(velocity_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    valid_t = np.isfinite(t)
    if valid_t.sum() < 3:
        raise ValueError(f"窗口 {w['window_id'].iloc[0]} 有效时间点不足3个")
    # 时间缺失行不参与时序计算；速度的局部缺失保留为NaN，由各统计函数忽略。
    t = t[valid_t]
    velocities = velocities[valid_t]
    n = len(t)
    stage_idx = split_stages(n, start_frac, end_frac)

    row: Dict[str, float] = {"window_id": w["window_id"].iloc[0]}
    for col in META_COLS[1:]:
        if col in w.columns:
            row[col] = w[col].iloc[0]

    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    row.update(
        {
            "ts_sample_count": n,
            "ts_duration_raw_s": float(np.ptp(t)),
            "ts_dt_mean_s": safe_mean(finite_dt),
            "ts_dt_std_s": safe_std(finite_dt),
            "ts_dt_max_s": float(np.max(finite_dt)) if finite_dt.size else np.nan,
            "ts_missing_velocity_frac": float(np.mean(~np.isfinite(velocities))),
        }
    )

    # 1) 单声道统计与分阶段特征。
    channel_cvs: List[float] = []
    channel_plateau_cvs: List[float] = []
    channel_drift_abs: List[float] = []
    channel_tv: List[float] = []
    for j in range(N_CHORDS):
        x = velocities[:, j]
        prefix = f"ts_v{j}"
        basic = normalized_stats(t, x, prefix)
        stages = stage_features(x, prefix, stage_idx)
        row.update(basic)
        row.update(stages)
        channel_cvs.append(basic[f"{prefix}_cv"])
        channel_plateau_cvs.append(stages[f"{prefix}_plateau_cv"])
        channel_drift_abs.append(abs(basic[f"{prefix}_drift_rel"]))
        channel_tv.append(basic[f"{prefix}_tv_step_rel"])

    pair_mean, pair_min = mean_pairwise_corr(velocities)
    row.update(
        {
            "ts_channel_cv_mean": safe_mean(channel_cvs),
            "ts_channel_cv_max": float(np.nanmax(channel_cvs)),
            "ts_channel_plateau_cv_mean": safe_mean(channel_plateau_cvs),
            "ts_channel_plateau_cv_max": float(np.nanmax(channel_plateau_cvs)),
            "ts_channel_drift_abs_mean": safe_mean(channel_drift_abs),
            "ts_channel_drift_abs_max": float(np.nanmax(channel_drift_abs)),
            "ts_channel_tv_mean": safe_mean(channel_tv),
            "ts_pair_corr_mean": pair_mean,
            "ts_pair_corr_min": pair_min,
        }
    )

    # 2) 声道间动态差。
    add_difference_features(row, t, velocities)

    # 3) 瞬时物理流量 q(t)。
    q_m3s = AREA_M2 * (velocities @ W_PHYS6)
    row.update(normalized_stats(t, q_m3s, "ts_q"))
    row.update(stage_features(q_m3s, "ts_q", stage_idx))
    row["ts_q_spike_frac"] = robust_spike_fraction(q_m3s)
    row["ts_q_negative_frac"] = float(np.mean(q_m3s < 0))
    if len(t) >= 2:
        row["ts_volume_trapz_m3"] = float(np.trapezoid(q_m3s, t) if hasattr(np, "trapezoid") else np.trapz(q_m3s, t))
    else:
        row["ts_volume_trapz_m3"] = np.nan

    # 4) 原始传播时间稳定性（如果存在）。只取相对波动，避免将绝对声速尺度过度编码。
    mean_us_cols = [f"mean_us_{j}" for j in range(N_CHORDS)]
    if all(c in w.columns for c in mean_us_cols):
        mean_us = w.loc[valid_t, mean_us_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        cvs = []
        drifts = []
        for j in range(N_CHORDS):
            x = mean_us[:, j]
            basic = normalized_stats(t, x, f"ts_us{j}")
            row[f"ts_us{j}_cv"] = basic[f"ts_us{j}_cv"]
            row[f"ts_us{j}_drift_rel"] = basic[f"ts_us{j}_drift_rel"]
            cvs.append(row[f"ts_us{j}_cv"])
            drifts.append(abs(row[f"ts_us{j}_drift_rel"]))
        row["ts_us_cv_mean"] = safe_mean(cvs)
        row["ts_us_drift_abs_mean"] = safe_mean(drifts)

    return row


COMPACT_FEATURES = [
    # 瞬时物理流量：直接对应窗口内重复性。
    "ts_q_cv",
    "ts_q_mad_rel",
    "ts_q_iqr_rel",
    "ts_q_drift_rel",
    "ts_q_tv_step_rel",
    "ts_q_lag1",
    "ts_q_start_over_plateau",
    "ts_q_end_over_plateau",
    "ts_q_plateau_cv",
    "ts_q_spike_frac",
    # 五声道共同波动与一致性。
    "ts_channel_cv_mean",
    "ts_channel_cv_max",
    "ts_channel_plateau_cv_mean",
    "ts_channel_drift_abs_mean",
    "ts_channel_tv_mean",
    "ts_pair_corr_mean",
    "ts_pair_corr_min",
    # 声道间不对称随时间的变化。
    "ts_diff_top_bottom_std",
    "ts_diff_top_bottom_drift",
    "ts_diff_inner_skew_std",
    "ts_diff_inner_skew_drift",
    "ts_diff_center_edge_std",
    "ts_diff_center_edge_drift",
    "ts_diff_alternating_std",
    "ts_diff_alternating_drift",
]


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].std() <= EPS or pair.iloc[:, 1].std() <= EPS:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))


def build_diagnostics(merged: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if not {"standard_volume_m3", "duration_s"}.issubset(merged.columns):
        return pd.DataFrame()

    chord_cols = [f"chord{j}" for j in range(N_CHORDS)]
    if not all(c in merged.columns for c in chord_cols):
        return pd.DataFrame()

    duration = pd.to_numeric(merged["duration_s"], errors="coerce").to_numpy(dtype=float)
    chords = merged[chord_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    base_volume = AREA_M2 * duration * (chords @ W_PHYS6)
    standard_volume = pd.to_numeric(merged["standard_volume_m3"], errors="coerce").to_numpy(dtype=float)
    target = np.log(standard_volume / base_volume)

    work = merged.copy()
    work["_target"] = target
    group_cols = [c for c in ["date", "flow_point"] if c in work.columns]
    if len(group_cols) != 2:
        group_cols = []

    if group_cols:
        valid_groups = work.groupby(group_cols)["window_id"].transform("size") >= 3
        work_valid = work.loc[valid_groups].copy()
        target_centered = work_valid["_target"] - work_valid.groupby(group_cols)["_target"].transform("mean")
        group_target = work_valid.groupby(group_cols)["_target"].mean()
    else:
        work_valid = work
        target_centered = pd.Series(np.nan, index=work.index)
        group_target = pd.Series(dtype=float)

    rows: List[Dict[str, float]] = []
    for feature in feature_cols:
        x = pd.to_numeric(work[feature], errors="coerce")
        row: Dict[str, float] = {
            "feature": feature,
            "missing_rate": float(x.isna().mean()),
            "unique_count": int(x.nunique(dropna=True)),
            "std": float(x.std(ddof=1)) if x.notna().sum() >= 2 else np.nan,
            "corr_target_pearson": pearson_corr(x, work["_target"]),
            "corr_target_spearman": spearman_corr(x, work["_target"]),
        }

        if group_cols:
            xv = pd.to_numeric(work_valid[feature], errors="coerce")
            x_centered = xv - work_valid.groupby(group_cols)[feature].transform("mean")
            row["corr_within_group"] = pearson_corr(x_centered, target_centered)
            group_x = work_valid.groupby(group_cols)[feature].mean()
            row["corr_between_group"] = pearson_corr(group_x, group_target)
            total_var = float(np.nanvar(xv.to_numpy(dtype=float), ddof=1)) if xv.notna().sum() >= 2 else np.nan
            within_var = float(np.nanvar(x_centered.to_numpy(dtype=float), ddof=1)) if x_centered.notna().sum() >= 2 else np.nan
            row["within_variance_ratio"] = within_var / total_var if np.isfinite(total_var) and total_var > EPS else np.nan
        else:
            row["corr_within_group"] = np.nan
            row["corr_between_group"] = np.nan
            row["within_variance_ratio"] = np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    result["abs_corr_within_group"] = result["corr_within_group"].abs()
    return result.sort_values(
        ["abs_corr_within_group", "corr_target_spearman"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not (0.05 <= args.start_frac <= 0.40 and 0.05 <= args.end_frac <= 0.40):
        raise ValueError("start-frac/end-frac 建议在0.05到0.40之间")
    if args.start_frac + args.end_frac >= 0.80:
        raise ValueError("启动段与结束段比例之和必须小于0.80")

    problem_dir = args.problem_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = problem_dir / "attachment6_window_raw_samples.csv"
    geo_path = problem_dir / "attachment7_meter_geometry.csv"
    window_path = problem_dir / "attachment1_window_data.csv"

    raw = read_csv(raw_path)
    geo = read_csv(geo_path)
    window_data = read_csv(window_path)

    required_raw = ["window_id", "rel_time_s"]
    missing_raw = [c for c in required_raw if c not in raw.columns]
    if missing_raw:
        raise ValueError("附件6缺少字段: " + ", ".join(missing_raw))
    if "window_id" not in window_data.columns:
        raise ValueError("附件1缺少 window_id")

    raw_v, velocity_cols = build_instantaneous_velocities(raw, geo)

    rows: List[Dict[str, float]] = []
    grouped = raw_v.groupby("window_id", sort=False)
    total = grouped.ngroups
    print(f"附件6: {len(raw_v)}条逐秒记录, {total}个窗口")
    for idx, (_, group) in enumerate(grouped, start=1):
        rows.append(
            extract_window_features(
                group,
                velocity_cols=velocity_cols,
                start_frac=args.start_frac,
                end_frac=args.end_frac,
            )
        )
        if idx % 25 == 0 or idx == total:
            print(f"  已处理 {idx}/{total} 个窗口")

    features = pd.DataFrame(rows)
    if features["window_id"].duplicated().any():
        raise RuntimeError("特征结果中出现重复 window_id")

    raw_ids = set(features["window_id"])
    window_ids = set(window_data["window_id"])
    missing_in_raw = sorted(window_ids - raw_ids)
    extra_in_raw = sorted(raw_ids - window_ids)
    if missing_in_raw:
        print(f"警告: 附件1中有 {len(missing_in_raw)} 个窗口未在附件6中找到")
    if extra_in_raw:
        print(f"警告: 附件6中有 {len(extra_in_raw)} 个窗口未在附件1中找到")

    full_path = output_dir / "timeseries_features_full.csv"
    features.to_csv(full_path, index=False, encoding="utf-8-sig")

    compact_cols = [c for c in COMPACT_FEATURES if c in features.columns]
    compact = features[["window_id"] + compact_cols].copy()
    compact_path = output_dir / "timeseries_features_compact.csv"
    compact.to_csv(compact_path, index=False, encoding="utf-8-sig")

    merged = window_data.merge(features, on="window_id", how="left", suffixes=("", "_raw"), validate="one_to_one")
    # 瞬时积分只作为候选特征；同时给出与Phys6窗口体积的相对差，便于检查映射偏差。
    if {"ts_volume_trapz_m3", "duration_s"}.issubset(merged.columns):
        chord_cols = [f"chord{j}" for j in range(N_CHORDS)]
        if all(c in merged.columns for c in chord_cols):
            base = (
                AREA_M2
                * pd.to_numeric(merged["duration_s"], errors="coerce").to_numpy(dtype=float)
                * (merged[chord_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) @ W_PHYS6)
            )
            merged["ts_integral_vs_phys6_pct"] = (
                (pd.to_numeric(merged["ts_volume_trapz_m3"], errors="coerce").to_numpy(dtype=float) - base)
                / base
                * 100.0
            )

    merged_path = output_dir / "merged_window_data_timeseries.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    feature_cols = [c for c in features.columns if c not in META_COLS]
    diagnostics = build_diagnostics(merged, feature_cols)
    diagnostics_path = output_dir / "timeseries_feature_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    manifest = {
        "input": {
            "raw": str(raw_path),
            "geometry": str(geo_path),
            "window_data": str(window_path),
        },
        "n_raw_rows": int(len(raw_v)),
        "n_windows_raw": int(features["window_id"].nunique()),
        "n_windows_attachment1": int(window_data["window_id"].nunique()),
        "n_full_features": int(len(feature_cols)),
        "n_compact_features": int(len(compact_cols)),
        "stage_definition": {
            "start_fraction": args.start_frac,
            "plateau_fraction": 1.0 - args.start_frac - args.end_frac,
            "end_fraction": args.end_frac,
        },
        "compact_features": compact_cols,
        "notes": [
            "特征构造不使用 standard_volume_m3。",
            "诊断文件中的目标相关性只用于筛选，不可代替嵌套LODO评价。",
            "优先关注 corr_within_group 与 within_variance_ratio，以判断特征能否解释组内重复性。",
            "ts_integral_vs_phys6_pct 受附件6到chord近似映射误差影响，使用前需单独消融。",
        ],
    }
    manifest_path = output_dir / "timeseries_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n输出完成:")
    print(f"  全量特征: {full_path}")
    print(f"  紧凑特征: {compact_path} ({len(compact_cols)}维)")
    print(f"  合并数据: {merged_path}")
    print(f"  特征诊断: {diagnostics_path}")
    print(f"  特征说明: {manifest_path}")
    if not diagnostics.empty:
        print("\n组内相关性绝对值最高的10个特征（仅作筛选诊断）:")
        show_cols = ["feature", "corr_within_group", "corr_between_group", "within_variance_ratio", "missing_rate"]
        print(diagnostics[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
