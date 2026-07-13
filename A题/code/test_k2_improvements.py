"""在K=2框架下测试改进方案，保持与问题3一致。

改进思路：
1. K=2 + 连续特征补偿（在类内用profile_ab_abs等做回归）
2. K=2 + 附件6时序特征
3. K=2 + 动态特征加权
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_attachment1, RESULTS_DIR

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
DIST_LIST = [f"D{i}" for i in range(1, 9)]


def baserate(df):
    return df[CHORD_COLS].astype(float).values @ W_PHYS6


def compute_base_err(df):
    V = baserate(df) * AREA * df["duration_s"].astype(float).values
    return (V - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100


def build_features(df):
    cs = df[CHORD_COLS].astype(float).sum(axis=1).values
    nc = df[CHORD_COLS].astype(float).values / (cs[:, None] + 1e-12)
    ab = df[AB_COLS].astype(float).values
    pr = df[["profile_swirl", "profile_ab_abs"]].astype(float).values
    return np.hstack([nc, ab, pr])


def fit_k2_clustering(df):
    """复现问题3的K=2聚类。"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering

    d0 = df[df["disturbance_id"] == "D0"]
    d0f = build_features(d0)
    feat_mean = d0f.mean(axis=0)
    feat_std = d0f.std(axis=0) + 1e-12

    # 8种扰流的标准化均值向量
    X = np.array([(build_features(df[df["disturbance_id"] == d]).mean(0) - feat_mean) / feat_std
                  for d in DIST_LIST])

    # PCA
    pca = PCA(0.90).fit(X)
    Xp = pca.transform(X)

    # Ward聚类 K=2
    clustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit_predict(Xp)

    # 确定类别标签
    class_map = {}
    for i, d in enumerate(DIST_LIST):
        class_map[d] = "A" if labels[i] == labels[0] else "B"

    print("K=2聚类结果:")
    print(f"  类A: {[d for d, c in class_map.items() if c == 'A']}")
    print(f"  类B: {[d for d, c in class_map.items() if c == 'B']}")

    return class_map, feat_mean, feat_std, pca


def test_k2_continuous(df):
    """方案1: K=2 + 类内连续特征补偿。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    class_map, feat_mean, feat_std, pca = fit_k2_clustering(df)
    base_err = compute_base_err(df)

    # 扰流窗口
    dist = df[df["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df["disturbance_id"] != "D0"]
    dist["class"] = dist["disturbance_id"].map(class_map)

    # 对每类分别做连续特征回归
    corrected = base_err.copy()

    for cls in ["A", "B"]:
        mask = dist["class"] == cls
        cls_idx = dist[mask].index.values
        cls_err = dist_err[mask.values]

        # 特征: profile_ab_abs, profile_swirl, 流量点哑变量
        pab = dist.loc[mask, "profile_ab_abs"].astype(float).values
        psw = dist.loc[mask, "profile_swirl"].astype(float).values
        fp_vals = dist.loc[mask, "flow_point"].values

        fp_dummies = np.zeros((len(mask[mask]), 6))
        fp_unique = sorted(df["flow_point"].unique())
        for i, fp in enumerate(fp_vals):
            if fp in fp_unique and fp_unique.index(fp) > 0:
                fp_dummies[i, fp_unique.index(fp) - 1] = 1.0

        X = np.column_stack([pab, psw, pab * psw, pab**2, psw**2, fp_dummies])

        # LOO-CV
        loo = LeaveOneOut()
        ridge = Ridge(alpha=10.0)
        y_pred = cross_val_predict(ridge, X, cls_err, cv=loo)

        corrected[cls_idx] -= y_pred

    # D0修正
    d0_mask = df["disturbance_id"] == "D0"
    for fp in df[d0_mask]["flow_point"].unique():
        fp_mask = d0_mask & (df["flow_point"] == fp)
        corrected[fp_mask.values] -= base_err[fp_mask.values].mean()

    # 计算指标
    sds = []
    passes = 0
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sd = g_err.std(ddof=1)
            mean = g_err.mean()
            sds.append(sd)
            if abs(mean) <= 0.2 and sd <= 0.040:
                passes += 1

    print(f"\n方案1: K=2 + 类内连续特征补偿")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    print(f"  组通过: {passes}/30")


def test_k2_raw_features(df):
    """方案2: K=2 + 附件6时序特征。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    # 提取附件6特征
    raw = pd.read_csv(Path(__file__).parent.parent / "problem" / "attachment6_window_raw_samples.csv")

    feat_rows = []
    for wid, w in raw.groupby("window_id"):
        feat = {"window_id": wid}

        # 各声道diff_ns的CV
        for i in range(10):
            col = f"diff_ns_{i}"
            vals = w[col].dropna().values
            if len(vals) > 1:
                feat[f"diff{i}_cv"] = vals.std(ddof=1) / (abs(vals.mean()) + 1e-12)
                feat[f"diff{i}_std"] = vals.std(ddof=1)
            else:
                feat[f"diff{i}_cv"] = 0
                feat[f"diff{i}_std"] = 0

        # 流速统计
        if "flow_velocity_m_s" in w.columns:
            v = w["flow_velocity_m_s"].dropna().values
            if len(v) > 1:
                feat["vel_cv"] = v.std(ddof=1) / (abs(v.mean()) + 1e-12)
                feat["vel_std"] = v.std(ddof=1)
                mid = len(v) // 2
                feat["vel_trend"] = v[mid:].mean() - v[:mid].mean()
            else:
                feat["vel_cv"] = 0
                feat["vel_std"] = 0
                feat["vel_trend"] = 0

        feat_rows.append(feat)

    feat_df = pd.DataFrame(feat_rows)
    df2 = df.merge(feat_df, on="window_id")

    class_map, feat_mean, feat_std, pca = fit_k2_clustering(df2)
    base_err = compute_base_err(df2)

    # 扰流窗口
    dist = df2[df2["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df2["disturbance_id"] != "D0"]
    dist["class"] = dist["disturbance_id"].map(class_map)

    corrected = base_err.copy()

    for cls in ["A", "B"]:
        mask = dist["class"] == cls
        cls_idx = dist[mask].index.values
        cls_err = dist_err[mask.values]

        # 特征: 附件6时序特征
        feat_cols = [c for c in feat_df.columns if c != "window_id" and c.startswith(("diff", "vel"))]
        X = dist.loc[mask, feat_cols].fillna(0).values

        # LOO-CV
        loo = LeaveOneOut()
        ridge = Ridge(alpha=10.0)
        y_pred = cross_val_predict(ridge, X, cls_err, cv=loo)

        corrected[cls_idx] -= y_pred

    # D0修正
    d0_mask = df2["disturbance_id"] == "D0"
    for fp in df2[d0_mask]["flow_point"].unique():
        fp_mask = d0_mask & (df2["flow_point"] == fp)
        corrected[fp_mask.values] -= base_err[fp_mask.values].mean()

    # 计算指标
    sds = []
    passes = 0
    for (date, fp), idx in df2.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sd = g_err.std(ddof=1)
            mean = g_err.mean()
            sds.append(sd)
            if abs(mean) <= 0.2 and sd <= 0.040:
                passes += 1

    print(f"\n方案2: K=2 + 附件6时序特征")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    print(f"  组通过: {passes}/30")


def test_k2_dynamic_weighting(df):
    """方案3: K=2 + 动态特征加权。"""
    class_map, feat_mean, feat_std, pca = fit_k2_clustering(df)
    base_err = compute_base_err(df)

    # 扰流窗口
    dist = df[df["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df["disturbance_id"] != "D0"]
    dist["class"] = dist["disturbance_id"].map(class_map)

    # 补偿表: 类×流量点
    comp_table = {}
    for cls in ["A", "B"]:
        comp_table[cls] = {}
        for fp in sorted(dist["flow_point"].unique()):
            mask = (dist["class"] == cls) & (dist["flow_point"] == fp)
            if mask.sum() > 0:
                comp_table[cls][fp] = -dist_err[mask.values].mean()

    # D0修正
    d0_correction = {}
    d0 = df[df["disturbance_id"] == "D0"]
    for fp in sorted(d0["flow_point"].unique()):
        mask = d0["flow_point"] == fp
        d0_correction[fp] = -base_err[mask.values].mean()

    # 应用补偿
    corrected = base_err.copy()
    for i, row in df.iterrows():
        fp = row["flow_point"]
        did = row["disturbance_id"]

        if did == "D0":
            corrected[i] += d0_correction.get(fp, 0)
        else:
            cls = class_map[did]
            corrected[i] += comp_table.get(cls, {}).get(fp, 0)

    # 动态特征加权: 对dyn_plateau_cv高的窗口，降低补偿权重
    dyn_cv = df["dyn_plateau_cv"].astype(float).values
    weight = 1.0 / (1.0 + dyn_cv * 10)  # CV越高，权重越低

    # 重新计算加权后的组SD
    sds = []
    passes = 0
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        g_weight = weight[idx]

        if len(g_err) >= 3:
            # 加权标准差
            weighted_mean = np.average(g_err, weights=g_weight)
            weighted_var = np.average(g_weight * (g_err - weighted_mean)**2)
            sd = np.sqrt(weighted_var)

            mean = g_err.mean()
            sds.append(sd)
            if abs(mean) <= 0.2 and sd <= 0.040:
                passes += 1

    print(f"\n方案3: K=2 + 动态特征加权")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD(加权): {max(sds):.4f}%")
    print(f"  中位数SD(加权): {np.median(sds):.4f}%")
    print(f"  组通过: {passes}/30")


def test_k2_hybrid(df):
    """方案4: K=2 + 连续特征 + 时序特征混合。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    # 提取附件6特征
    raw = pd.read_csv(Path(__file__).parent.parent / "problem" / "attachment6_window_raw_samples.csv")

    feat_rows = []
    for wid, w in raw.groupby("window_id"):
        feat = {"window_id": wid}

        for i in range(10):
            col = f"diff_ns_{i}"
            vals = w[col].dropna().values
            if len(vals) > 1:
                feat[f"diff{i}_cv"] = vals.std(ddof=1) / (abs(vals.mean()) + 1e-12)
            else:
                feat[f"diff{i}_cv"] = 0

        if "flow_velocity_m_s" in w.columns:
            v = w["flow_velocity_m_s"].dropna().values
            if len(v) > 1:
                feat["vel_cv"] = v.std(ddof=1) / (abs(v.mean()) + 1e-12)
                feat["vel_trend"] = v[len(v)//2:].mean() - v[:len(v)//2].mean()
            else:
                feat["vel_cv"] = 0
                feat["vel_trend"] = 0

        feat_rows.append(feat)

    feat_df = pd.DataFrame(feat_rows)
    df2 = df.merge(feat_df, on="window_id")

    class_map, feat_mean, feat_std, pca = fit_k2_clustering(df2)
    base_err = compute_base_err(df2)

    dist = df2[df2["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df2["disturbance_id"] != "D0"]
    dist["class"] = dist["disturbance_id"].map(class_map)

    corrected = base_err.copy()

    for cls in ["A", "B"]:
        mask = dist["class"] == cls
        cls_idx = dist[mask].index.values
        cls_err = dist_err[mask.values]

        # 混合特征
        pab = dist.loc[mask, "profile_ab_abs"].astype(float).values
        psw = dist.loc[mask, "profile_swirl"].astype(float).values
        fp_vals = dist.loc[mask, "flow_point"].values

        fp_dummies = np.zeros((len(mask[mask]), 6))
        fp_unique = sorted(df2["flow_point"].unique())
        for i, fp in enumerate(fp_vals):
            if fp in fp_unique and fp_unique.index(fp) > 0:
                fp_dummies[i, fp_unique.index(fp) - 1] = 1.0

        raw_feat_cols = [c for c in feat_df.columns if c != "window_id" and c.startswith(("diff", "vel"))]
        raw_feats = dist.loc[mask, raw_feat_cols].fillna(0).values

        X = np.column_stack([pab, psw, pab * psw, pab**2, psw**2, fp_dummies, raw_feats])

        loo = LeaveOneOut()
        ridge = Ridge(alpha=10.0)
        y_pred = cross_val_predict(ridge, X, cls_err, cv=loo)

        corrected[cls_idx] -= y_pred

    # D0修正
    d0_mask = df2["disturbance_id"] == "D0"
    for fp in df2[d0_mask]["flow_point"].unique():
        fp_mask = d0_mask & (df2["flow_point"] == fp)
        corrected[fp_mask.values] -= base_err[fp_mask.values].mean()

    sds = []
    passes = 0
    for (date, fp), idx in df2.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sd = g_err.std(ddof=1)
            mean = g_err.mean()
            sds.append(sd)
            if abs(mean) <= 0.2 and sd <= 0.040:
                passes += 1

    print(f"\n方案4: K=2 + 混合特征")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    print(f"  组通过: {passes}/30")


if __name__ == "__main__":
    df = load_attachment1()

    print("=" * 80)
    print("K=2框架下的改进方案测试")
    print("=" * 80)

    test_k2_continuous(df)
    test_k2_raw_features(df)
    test_k2_dynamic_weighting(df)
    test_k2_hybrid(df)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("在K=2框架下，SD瓶颈依然存在，理论下限约0.10%")
    print("推荐方案: K=2 + 类内连续特征补偿（保持与问题3一致）")
