"""
A题改进模型 v2: Ridge + 二次多项式残差预测
============================================
1. Phys6 五声道物理积分作为基线 V_base
2. 预测对数修正量 y = log(V_std / V_base)
3. Ridge + PolynomialFeatures(degree=2) , 28维在线特征
4. 同时输出: apparent_train(样本内上限) + leave_one_date_out(折外泛化)
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut


AREA_M2 = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
FEATURE_COLS = (
    CHORD_COLS + AB_COLS + [
        "profile_top_bottom", "profile_center_all", "profile_edge_inner",
        "profile_inner_skew", "profile_ab_abs", "profile_swirl",
        "dyn_first_0p1_s", "dyn_tail_0p1_s",
        "dyn_start_over_plateau", "dyn_end_over_plateau",
        "dyn_plateau_cv", "dyn_active_eq_s",
        "zero_rate_med", "zero_rate_mad", "zero_age_s",
        "base_rate_m3h", "duration_s", "flow_point",
    ]
)


def load_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"window_id", "duration_s", *CHORD_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"缺少字段: {missing}")
    return df


def add_base_columns(df):
    out = df.copy()
    chord = out[CHORD_COLS].astype(float).to_numpy()
    duration = out["duration_s"].astype(float).to_numpy()
    out["base_volume_m3"] = AREA_M2 * duration * (chord @ W_PHYS6)
    out["base_rate_m3h"] = out["base_volume_m3"] / duration * 3600.0
    return out


def make_feature_frame(df, medians=None):
    work = add_base_columns(df)
    x = work[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median()
    return x.fillna(medians), medians


def make_model():
    return ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8,
        random_state=2026, n_jobs=-1,
    )


def fit_residual_model(train_df):
    train = add_base_columns(train_df)
    x, medians = make_feature_frame(train)
    y = np.log(train["standard_volume_m3"].astype(float).to_numpy()
               / train["base_volume_m3"].astype(float).to_numpy())
    model = make_model()
    model.fit(x, y)
    return model, medians


def predict_volume(model, medians, df):
    work = add_base_columns(df)
    x, _ = make_feature_frame(work, medians)
    return work["base_volume_m3"].to_numpy() * np.exp(model.predict(x))


def evaluate(df, pred):
    work = df.copy()
    work["model_volume_m3"] = np.asarray(pred, dtype=float)
    work["error_pct"] = (
        work["model_volume_m3"] / work["standard_volume_m3"].astype(float) - 1.0
    ) * 100.0

    groups = []
    for (date, flow_point), grp in work.groupby(["date", "flow_point"], sort=True):
        if len(grp) < 3:
            continue
        err = grp["error_pct"].astype(float)
        groups.append({
            "date": date, "flow_point": flow_point, "n": len(grp),
            "mean_error_pct": float(err.mean()), "sd_pct": float(err.std(ddof=1)),
            "pass_group": abs(err.mean()) <= 0.2 and err.std(ddof=1) <= 0.040,
        })
    gdf = pd.DataFrame(groups)

    d0 = work[work["disturbance_id"].eq("D0")]
    d0_means = d0.groupby("flow_point")["error_pct"].mean()
    u_l = math.sqrt((d0_means ** 2).sum() / max(len(d0_means) - 1, 1))

    u_r = float(gdf["sd_pct"].max())

    use = work[work["flow_point"].between(40, 100)]
    base = use[use["condition_note"].eq("no_disturbance_reference")].groupby(
        "flow_point")["error_pct"].mean()
    dist = use[use["condition_note"].eq("disturbed_test")].groupby(
        ["disturbance_id", "flow_point"])["error_pct"].agg(["mean", "std"]).reset_index()
    dist["base_mean"] = dist["flow_point"].map(base)
    dist = dist.dropna(subset=["base_mean"])
    dist["abs_drift"] = (dist["base_mean"] - dist["mean"]).abs()
    u_d_c = float(dist["abs_drift"].max() / math.sqrt(3))
    u_d_r = float(dist["std"].fillna(0).max())
    u_d = math.sqrt(u_d_c ** 2 + u_d_r ** 2)

    return {
        "mae_pct": float(work["error_pct"].abs().mean()),
        "mean_error_pct": float(work["error_pct"].mean()),
        "group_pass": int(gdf["pass_group"].sum()) if not gdf.empty else 0,
        "group_total": int(len(gdf)),
        "u_nor_L_pct": u_l, "u_nor_r_pct": u_r,
        "u_nor_d_c_pct": u_d_c, "u_nor_d_r_pct": u_d_r, "u_nor_d_pct": u_d,
        "target_L_pass": bool(u_l < 0.036),
        "target_r_pass": bool(u_r < 0.040),
        "target_d_pass": bool(u_d < 0.115),
    }, gdf


def leave_one_date_out(df):
    pred = np.zeros(len(df), dtype=float)
    dates = df["date"].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(df, groups=dates):
        train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        model, medians = fit_residual_model(train)
        pred[test_idx] = predict_volume(model, medians, test)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).resolve().parent / "attachment1_window_data.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(args.data)

    base_pred = add_base_columns(df)["base_volume_m3"].to_numpy()
    base_m, _ = evaluate(df, base_pred)

    model, medians = fit_residual_model(df)
    apparent_pred = predict_volume(model, medians, df)
    apparent_m, apparent_g = evaluate(df, apparent_pred)

    lodo_pred = leave_one_date_out(df)
    lodo_m, lodo_g = evaluate(df, lodo_pred)

    for label, pred_arr, fname in [
        ("apparent", apparent_pred, "problem4_submission_v2_apparent.csv"),
        ("lodo", lodo_pred, "problem4_submission_v2_lodo.csv"),
    ]:
        out = df[["window_id"]].copy()
        out["model_volume_m3"] = pred_arr
        out.to_csv(args.output_dir / fname, index=False, encoding="utf-8-sig")

    comparison = pd.DataFrame([
        {"evaluation": "phys6_base", **base_m},
        {"evaluation": "apparent_train", **apparent_m},
        {"evaluation": "leave_one_date_out", **lodo_m},
    ])
    comparison.to_csv(args.output_dir / "problem4_metrics_v2.csv",
                      index=False, encoding="utf-8-sig")

    json.dump({
        "features": list(FEATURE_COLS),
        "model": "Ridge + PolynomialFeatures(degree=2)",
        "metrics": {"phys6": base_m, "apparent": apparent_m, "lodo": lodo_m},
        "n_features": len(FEATURE_COLS),
    }, open(args.output_dir / "problem4_model_v2_summary.json", "w", encoding="utf-8"),
       ensure_ascii=False, indent=2)

    print(comparison.to_string(index=False))
    print(f"\n输出: {args.output_dir}")


if __name__ == "__main__":
    main()
