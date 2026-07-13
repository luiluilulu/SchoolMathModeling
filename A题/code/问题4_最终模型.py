"""问题4：最终达标模型 — 全链路级联 (基线→检测→分类→补偿)。

求解方法：Phys6基线 + 双阈值扰流检测 + Mahalanobis在线分类 + 分流量点补偿。
          D0窗口施加流量点基线修正以降 u_nor_L。
输出：output/results/problem4_submission.csv
"""

import numpy as np
import pandas as pd

from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
FEATURE_NAMES = [
    "norm_chord0", "norm_chord1", "norm_chord2", "norm_chord3", "norm_chord4",
    "ab0", "ab1", "ab2", "ab3", "ab4", "profile_swirl", "profile_ab_abs",
]
DIST_LIST = [f"D{i}" for i in range(1, 9)]


# ============================================================
# 离线阶段：参数估计（所有参数在此固定，在线仅查表和计算）
# ============================================================

def baserate(df):
    """Phys6 面平均流速 (m/s)."""
    return df[CHORD_COLS].astype(float).values @ W_PHYS6


def build_features(df):
    """12维在线特征矩阵。"""
    cs = df[CHORD_COLS].astype(float).sum(axis=1).values
    nc = df[CHORD_COLS].astype(float).values / (cs[:, None] + 1e-12)
    ab = df[AB_COLS].astype(float).values
    pr = df[["profile_swirl", "profile_ab_abs"]].astype(float).values
    return np.hstack([nc, ab, pr])


def fit_detection_thresholds(df):
    """双阈值: D0 max(|x|) + 3σ."""
    d0 = df[df["disturbance_id"] == "D0"]
    tau_ab = float(d0["profile_ab_abs"].abs().max() + 3 * d0["profile_ab_abs"].abs().std())
    tau_sw = float(d0["profile_swirl"].abs().max() + 3 * d0["profile_swirl"].abs().std())
    return tau_ab, tau_sw


def fit_offline_params(df):
    """PCA、聚类中心、协方差、补偿表 — 全部离线固定。"""
    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf
    from scipy.cluster.hierarchy import linkage, fcluster

    d0 = df[df["disturbance_id"] == "D0"]
    d0f = build_features(d0)
    feat_mean = d0f.mean(axis=0)
    feat_std = d0f.std(axis=0) + 1e-12

    # 8种扰流的标准化均值向量 → PCA → Ward聚类 (K=2)
    X = np.array([(build_features(df[df["disturbance_id"] == d]).mean(0) - feat_mean) / feat_std
                  for d in DIST_LIST])
    pca = PCA(0.90).fit(X)
    Xp = pca.transform(X)
    Z = linkage(Xp, method="ward")
    raw_labels = fcluster(Z, 2, criterion="maxclust")
    class_map = {d: ("A" if raw_labels[i] == raw_labels[0] else "B")
                 for i, d in enumerate(DIST_LIST)}

    # 扰流窗口分类用 Mahalanobis 参数
    dist_df = df[df["disturbance_id"] != "D0"].copy()
    dist_feat = (build_features(dist_df) - feat_mean) / feat_std
    dist_pc = pca.transform(dist_feat)
    centers, covs = {}, {}
    for cls in ["A", "B"]:
        idx = dist_df["disturbance_id"].map(class_map).eq(cls).values
        centers[cls] = dist_pc[idx].mean(0)
        covs[cls] = LedoitWolf().fit(dist_pc[idx]).covariance_

    # 补偿表: comp_to_zero (类×流量点)
    dist_df["base_err"] = (
        (baserate(dist_df) * AREA * dist_df["duration_s"].astype(float)
         - dist_df["standard_volume_m3"]) / dist_df["standard_volume_m3"]
    )
    dist_df["online_class"] = dist_df["disturbance_id"].map(class_map)
    comp_table = {}
    for cls in ["A", "B"]:
        comp_table[cls] = {}
        for fp in sorted(dist_df["flow_point"].unique()):
            sub = dist_df[(dist_df["online_class"] == cls) & (dist_df["flow_point"] == fp)]
            if len(sub) >= 3:
                comp_table[cls][fp] = -float(sub["base_err"].mean())
            elif len(sub) >= 1:
                comp_table[cls][fp] = -float(sub["base_err"].mean())

    # D0 流量点基线修正表 (独立于扰流补偿)
    d0_df = d0.copy()
    d0_df["base_err"] = (
        (baserate(d0_df) * AREA * d0_df["duration_s"].astype(float)
         - d0_df["standard_volume_m3"]) / d0_df["standard_volume_m3"]
    )
    d0_correction = {}
    for fp in sorted(d0_df["flow_point"].unique()):
        sub = d0_df[d0_df["flow_point"] == fp]
        # 该流量点的平均误差 → 修正系数
        d0_correction[fp] = -float(sub["base_err"].mean())

    return {
        "feat_mean": feat_mean, "feat_std": feat_std, "pca": pca,
        "centers": centers, "covs": covs, "class_map": class_map,
        "comp_table": comp_table, "d0_correction": d0_correction,
    }


# ============================================================
# 在线阶段：逐窗口推理
# ============================================================

def predict(df, tau_ab, tau_sw, params):
    """全量窗口预测，返回 model_volume_m3 数组。"""
    V_base = baserate(df) * AREA * df["duration_s"].astype(float).values
    flag = (df["profile_ab_abs"].abs() > tau_ab) | (df["profile_swirl"].abs() > tau_sw)

    feat_mean = params["feat_mean"]
    feat_std = params["feat_std"]
    pca = params["pca"]
    centers = params["centers"]
    covs = params["covs"]
    comp_table = params["comp_table"]
    d0_corr = params["d0_correction"]

    V_final = V_base.copy()
    for i in range(len(df)):
        fp = int(df.iloc[i]["flow_point"])
        if flag.iloc[i]:
            # 扰流: 分类 + 补偿
            feat = (build_features(df.iloc[[i]]) - feat_mean) / feat_std
            pc = pca.transform(feat)[0]
            dists = {c: float(np.sqrt((pc - centers[c]) @ np.linalg.pinv(covs[c]) @ (pc - centers[c]).T))
                     for c in ["A", "B"]}
            cls = min(dists, key=dists.get)
            delta = comp_table.get(cls, {}).get(fp, 0)
            V_final[i] *= (1 + delta)
        else:
            # D0: 流量点基线修正
            delta = d0_corr.get(fp, 0)
            V_final[i] *= (1 + delta)

    return V_final


def main():
    ensure_dirs()
    df = load_attachment1()

    tau_ab, tau_sw = fit_detection_thresholds(df)
    params = fit_offline_params(df)

    pred = predict(df, tau_ab, tau_sw, params)
    err = (pred - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    mae = err.abs().mean()

    submission = df[["window_id"]].copy()
    submission["model_volume_m3"] = pred
    submission.to_csv(RESULTS_DIR / "problem4_submission.csv", index=False, encoding="utf-8-sig")

    print(f"最终模型 MAE: {mae:.4f}%")
    print(f"扰流检测阈值: ab_abs={tau_ab:.5f}, swirl={tau_sw:.5f}")
    print(f"补偿表: {sorted(params['comp_table']['A'].keys())}")
    print(f"D0修正表: {params['d0_correction']}")
    print(f"提交文件: {RESULTS_DIR / 'problem4_submission.csv'}")


if __name__ == "__main__":
    main()
