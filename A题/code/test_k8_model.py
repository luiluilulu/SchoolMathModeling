"""测试K=8模型: 每种扰流单独一类，替代K=2。"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs
from evaluate_submission import evaluate

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
FEATURE_NAMES = [
    "norm_chord0", "norm_chord1", "norm_chord2", "norm_chord3", "norm_chord4",
    "ab0", "ab1", "ab2", "ab3", "ab4", "profile_swirl", "profile_ab_abs",
]
DIST_LIST = [f"D{i}" for i in range(1, 9)]


def baserate(df):
    return df[CHORD_COLS].astype(float).values @ W_PHYS6


def build_features(df):
    cs = df[CHORD_COLS].astype(float).sum(axis=1).values
    nc = df[CHORD_COLS].astype(float).values / (cs[:, None] + 1e-12)
    ab = df[AB_COLS].astype(float).values
    pr = df[["profile_swirl", "profile_ab_abs"]].astype(float).values
    return np.hstack([nc, ab, pr])


def fit_k8_model(df):
    """K=8: 每种扰流单独一类，Mahalanobis分类。"""
    from sklearn.covariance import LedoitWolf

    d0 = df[df["disturbance_id"] == "D0"]
    d0f = build_features(d0)
    feat_mean = d0f.mean(axis=0)
    feat_std = d0f.std(axis=0) + 1e-12

    # 每种扰流的标准化均值向量
    X = np.array([(build_features(df[df["disturbance_id"] == d]).mean(0) - feat_mean) / feat_std
                  for d in DIST_LIST])

    # 扰流窗口分类用 Mahalanobis 参数 (每类独立)
    dist_df = df[df["disturbance_id"] != "D0"].copy()
    dist_feat = (build_features(dist_df) - feat_mean) / feat_std
    centers, covs = {}, {}
    for d in DIST_LIST:
        idx = dist_df["disturbance_id"].eq(d).values
        if idx.sum() > 0:
            centers[d] = dist_feat[idx].mean(0)
            if idx.sum() > 1:
                covs[d] = LedoitWolf().fit(dist_feat[idx]).covariance_
            else:
                covs[d] = np.eye(dist_feat.shape[1]) * 0.01

    # 补偿表: 每类×每流量点
    dist_df["base_err"] = (
        (baserate(dist_df) * AREA * dist_df["duration_s"].astype(float)
         - dist_df["standard_volume_m3"]) / dist_df["standard_volume_m3"]
    )
    comp_table = {}
    for d in DIST_LIST:
        comp_table[d] = {}
        for fp in sorted(dist_df["flow_point"].unique()):
            sub = dist_df[(dist_df["disturbance_id"] == d) & (dist_df["flow_point"] == fp)]
            if len(sub) >= 1:
                comp_table[d][fp] = -float(sub["base_err"].mean())

    # D0 流量点基线修正
    d0_df = d0.copy()
    d0_df["base_err"] = (
        (baserate(d0_df) * AREA * d0_df["duration_s"].astype(float)
         - d0_df["standard_volume_m3"]) / d0_df["standard_volume_m3"]
    )
    d0_correction = {}
    for fp in sorted(d0_df["flow_point"].unique()):
        sub = d0_df[d0_df["flow_point"] == fp]
        d0_correction[fp] = -float(sub["base_err"].mean())

    return {
        "feat_mean": feat_mean, "feat_std": feat_std,
        "centers": centers, "covs": covs,
        "comp_table": comp_table, "d0_correction": d0_correction,
    }


def predict_k8(df, tau_ab, tau_sw, params):
    """K=8预测。"""
    V_base = baserate(df) * AREA * df["duration_s"].astype(float).values
    flag = (df["profile_ab_abs"].abs() > tau_ab) | (df["profile_swirl"].abs() > tau_sw)

    feat_mean = params["feat_mean"]
    feat_std = params["feat_std"]
    centers = params["centers"]
    covs = params["covs"]
    comp_table = params["comp_table"]
    d0_corr = params["d0_correction"]

    V_final = V_base.copy()
    for i in range(len(df)):
        fp = int(df.iloc[i]["flow_point"])
        if flag.iloc[i]:
            feat = (build_features(df.iloc[[i]]) - feat_mean) / feat_std
            pc = feat[0]
            dists = {}
            for d in DIST_LIST:
                if d in centers:
                    diff = pc - centers[d]
                    cov_inv = np.linalg.pinv(covs[d])
                    dists[d] = float(np.sqrt(diff @ cov_inv @ diff))
            if dists:
                cls = min(dists, key=dists.get)
                delta = comp_table.get(cls, {}).get(fp, 0)
                V_final[i] *= (1 + delta)
        else:
            delta = d0_corr.get(fp, 0)
            V_final[i] *= (1 + delta)

    return V_final


def fit_detection_thresholds(df):
    d0 = df[df["disturbance_id"] == "D0"]
    tau_ab = float(d0["profile_ab_abs"].abs().max() + 3 * d0["profile_ab_abs"].abs().std())
    tau_sw = float(d0["profile_swirl"].abs().max() + 3 * d0["profile_swirl"].abs().std())
    return tau_ab, tau_sw


def main():
    ensure_dirs()
    df = load_attachment1()

    tau_ab, tau_sw = fit_detection_thresholds(df)
    params = fit_k8_model(df)

    pred = predict_k8(df, tau_ab, tau_sw, params)
    err = (pred - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    mae = err.abs().mean()

    submission = df[["window_id"]].copy()
    submission["model_volume_m3"] = pred
    out_path = RESULTS_DIR / "problem4_k8_submission.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"K=8模型 MAE: {mae:.4f}%")
    print(f"提交文件: {out_path}")

    evaluate(out_path, RESULTS_DIR / "eval_k8")


if __name__ == "__main__":
    main()
