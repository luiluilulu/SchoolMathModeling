"""
A题改进模型 v2
=============
思路：
1. 保留 Phys6 五声道物理积分作为基线；
2. 预测对数相对修正量 log(V_std / V_base)，而不是继续做“类别×流量点”常数查表；
3. 输入仅使用在线可获得特征，不使用 date、disturbance_id、standard_volume_m3；
4. 同时输出：
   - apparent_train：全量拟合后回代，仅用于判断当前数据的可拟合上限；
   - leave_one_date_out：整日留出预测，用于判断跨日期泛化能力。

注意：
apparent_train 的 30/30 不能当作独立验证结果。论文应同时报告 leave_one_date_out。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut


AREA_M2 = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])

CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]

FEATURE_COLS = (
    CHORD_COLS
    + AB_COLS
    + [
        "profile_top_bottom",
        "profile_center_all",
        "profile_edge_inner",
        "profile_inner_skew",
        "profile_ab_abs",
        "profile_swirl",
        "dyn_first_0p1_s",
        "dyn_tail_0p1_s",
        "dyn_start_over_plateau",
        "dyn_end_over_plateau",
        "dyn_plateau_cv",
        "dyn_active_eq_s",
        "zero_rate_med",
        "zero_rate_mad",
        "zero_age_s",
        "base_rate_m3h",
        "duration_s",
        "flow_point",
    ]
)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"window_id", "duration_s", *CHORD_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"输入缺少必要字段: {missing}")
    return df


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    chord = out[CHORD_COLS].astype(float).to_numpy()
    duration = out["duration_s"].astype(float).to_numpy()
    base_velocity = chord @ W_PHYS6
    out["base_volume_m3"] = AREA_M2 * duration * base_velocity
    out["base_rate_m3h"] = out["base_volume_m3"] / duration * 3600.0
    return out


def make_feature_frame(
    df: pd.DataFrame,
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    work = add_base_columns(df)
    missing = sorted(set(FEATURE_COLS) - set(work.columns))
    if missing:
        raise ValueError(f"输入缺少模型特征: {missing}")

    x = work[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median()
    x = x.fillna(medians)
    return x, medians


def make_model(random_state: int = 2026) -> ExtraTreesRegressor:
    # min_samples_leaf=2 防止单棵树逐样本完全记忆；
    # 但全量回代仍是样本内结果，必须配合整日留出评价。
    return ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_residual_model(
    train_df: pd.DataFrame,
    random_state: int = 2026,
) -> tuple[ExtraTreesRegressor, pd.Series]:
    if "standard_volume_m3" not in train_df.columns:
        raise ValueError("训练集必须包含 standard_volume_m3")

    train = add_base_columns(train_df)
    x, medians = make_feature_frame(train)
    y = np.log(
        train["standard_volume_m3"].astype(float).to_numpy()
        / train["base_volume_m3"].astype(float).to_numpy()
    )
    model = make_model(random_state)
    model.fit(x, y)
    return model, medians


def predict_volume(
    model: ExtraTreesRegressor,
    medians: pd.Series,
    df: pd.DataFrame,
    residual_scale: float = 1.0,
) -> np.ndarray:
    """
    residual_scale=1 使用完整数据驱动修正；
    residual_scale<1 可向物理基线收缩，但会降低当前数据上的合格率。
    """
    work = add_base_columns(df)
    x, _ = make_feature_frame(work, medians)
    log_corr = model.predict(x)
    return work["base_volume_m3"].to_numpy() * np.exp(residual_scale * log_corr)


def evaluate(df: pd.DataFrame, pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    if "standard_volume_m3" not in df.columns:
        raise ValueError("评价数据必须包含 standard_volume_m3")

    work = df.copy()
    work["model_volume_m3"] = np.asarray(pred, dtype=float)
    work["error_pct"] = (
        work["model_volume_m3"] / work["standard_volume_m3"].astype(float) - 1.0
    ) * 100.0

    group_rows = []
    for (date, flow_point), group in work.groupby(["date", "flow_point"], sort=True):
        if len(group) < 3:
            continue
        err = group["error_pct"].astype(float)
        mean = float(err.mean())
        sd = float(err.std(ddof=1))
        group_rows.append(
            {
                "date": date,
                "flow_point": flow_point,
                "n": len(group),
                "mean_error_pct": mean,
                "sd_pct": sd,
                "pass_group": abs(mean) <= 0.2 and sd <= 0.040,
                "disturbance_id": group["disturbance_id"].iloc[0]
                if "disturbance_id" in group.columns
                else "",
            }
        )
    groups = pd.DataFrame(group_rows)

    d0 = work[work["disturbance_id"].eq("D0")]
    d0_means = d0.groupby("flow_point")["error_pct"].mean()
    if len(d0_means) > 1:
        u_nor_l = math.sqrt(float(np.sum(d0_means.to_numpy() ** 2)) / (len(d0_means) - 1))
    else:
        u_nor_l = float("nan")

    u_nor_r = float(groups["sd_pct"].max())

    use = work[work["flow_point"].between(40, 100)]
    base_mean = (
        use[use["condition_note"].eq("no_disturbance_reference")]
        .groupby("flow_point")["error_pct"]
        .mean()
    )
    disturbed = (
        use[use["condition_note"].eq("disturbed_test")]
        .groupby(["disturbance_id", "flow_point"])["error_pct"]
        .agg(["mean", "std"])
        .reset_index()
    )
    disturbed["base_mean"] = disturbed["flow_point"].map(base_mean)
    disturbed = disturbed.dropna(subset=["base_mean"])
    disturbed["abs_drift"] = (disturbed["base_mean"] - disturbed["mean"]).abs()

    u_nor_d_c = float(disturbed["abs_drift"].max() / math.sqrt(3))
    u_nor_d_r = float(disturbed["std"].fillna(0).max())
    u_nor_d = math.sqrt(u_nor_d_c**2 + u_nor_d_r**2)

    metrics = {
        "mae_pct": float(work["error_pct"].abs().mean()),
        "mean_error_pct": float(work["error_pct"].mean()),
        "group_pass": int(groups["pass_group"].sum()),
        "group_total": int(len(groups)),
        "max_group_mean_abs_pct": float(groups["mean_error_pct"].abs().max()),
        "u_nor_L_pct": u_nor_l,
        "u_nor_r_pct": u_nor_r,
        "u_nor_d_c_pct": u_nor_d_c,
        "u_nor_d_r_pct": u_nor_d_r,
        "u_nor_d_pct": u_nor_d,
        "target_L_pass": bool(u_nor_l < 0.036),
        "target_r_pass": bool(u_nor_r < 0.040),
        "target_d_pass": bool(u_nor_d < 0.115),
    }
    return metrics, groups


def leave_one_date_out(df: pd.DataFrame, random_state: int = 2026) -> np.ndarray:
    """所有预处理和模型参数都只在训练日期中估计。"""
    pred = np.zeros(len(df), dtype=float)
    groups = df["date"].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, groups=groups)):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        model, medians = fit_residual_model(train, random_state + fold)
        pred[test_idx] = predict_volume(model, medians, test)
    return pred


def save_submission(df: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    out = df[["window_id"]].copy()
    out["model_volume_m3"] = np.asarray(pred, dtype=float)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "attachment1_window_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data)

    # 物理基线
    base = add_base_columns(df)["base_volume_m3"].to_numpy()
    base_metrics, base_groups = evaluate(df, base)

    # 全量训练后回代：只能视为当前数据可拟合上限
    model, medians = fit_residual_model(df)
    apparent_pred = predict_volume(model, medians, df)
    apparent_metrics, apparent_groups = evaluate(df, apparent_pred)

    # 严格整日留出
    lodo_pred = leave_one_date_out(df)
    lodo_metrics, lodo_groups = evaluate(df, lodo_pred)

    save_submission(
        df,
        apparent_pred,
        args.output_dir / "problem4_submission_v2_apparent.csv",
    )
    save_submission(
        df,
        lodo_pred,
        args.output_dir / "problem4_submission_v2_lodo.csv",
    )

    comparison = pd.DataFrame(
        [
            {"evaluation": "phys6_base", **base_metrics},
            {"evaluation": "apparent_train", **apparent_metrics},
            {"evaluation": "leave_one_date_out", **lodo_metrics},
        ]
    )
    comparison.to_csv(
        args.output_dir / "problem4_metrics_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_groups.to_csv(
        args.output_dir / "problem4_groups_v2_phys6.csv",
        index=False,
        encoding="utf-8-sig",
    )
    apparent_groups.to_csv(
        args.output_dir / "problem4_groups_v2_apparent.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lodo_groups.to_csv(
        args.output_dir / "problem4_groups_v2_lodo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "feature_columns": FEATURE_COLS,
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
        "metrics": {
            "phys6_base": base_metrics,
            "apparent_train": apparent_metrics,
            "leave_one_date_out": lodo_metrics,
        },
    }
    with open(
        args.output_dir / "problem4_model_v2_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(comparison.to_string(index=False))
    print(f"\n输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
