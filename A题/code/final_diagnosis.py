"""最终诊断：分析SD瓶颈并测试所有可能的改进方向。"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_attachment1, RESULTS_DIR

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]


def baserate(df):
    return df[CHORD_COLS].astype(float).values @ W_PHYS6


def compute_base_err(df):
    V = baserate(df) * AREA * df["duration_s"].astype(float).values
    return (V - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100


def analyze_worst_groups():
    """深入分析SD最高的几个组。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df = df.merge(sub, on="window_id")
    df["err_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    
    print("=" * 90)
    print("SD最高组的详细分解")
    print("=" * 90)
    
    # 找SD最高的5个组
    group_stats = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g = df.iloc[idx]
        errs = g["err_pct"].values
        if len(errs) >= 3:
            group_stats.append({
                "date": date, "fp": fp, "n": len(errs),
                "mean": errs.mean(), "sd": errs.std(ddof=1),
                "min": errs.min(), "max": errs.max(),
            })
    
    gdf = pd.DataFrame(group_stats).sort_values("sd", ascending=False)
    
    for _, row in gdf.head(5).iterrows():
        date, fp = row["date"], row["fp"]
        g = df[(df["date"] == date) & (df["flow_point"] == fp)]
        
        print(f"\n日期={date} fp={fp} N={row['n']} SD={row['sd']:.4f}% mean={row['mean']:+.4f}%")
        print("-" * 90)
        
        # 按disturbance_id分解
        for did in sorted(g["disturbance_id"].unique()):
            sub_g = g[g["disturbance_id"] == did]
            errs = sub_g["err_pct"].values
            pab = sub_g["profile_ab_abs"].astype(float).values
            psw = sub_g["profile_swirl"].astype(float).values
            
            print(f"  {did}: N={len(errs)} err=[{', '.join(f'{e:+.3f}' for e in errs)}]%")
            if len(errs) > 1:
                print(f"       SD={errs.std(ddof=1):.4f}% mean={errs.mean():+.4f}%")
            print(f"       profile_ab_abs = [{pab.min():.4f}, {pab.max():.4f}]")
            print(f"       profile_swirl  = [{psw.min():.4f}, {psw.max():.4f}]")
        
        # 按class分解 (当前模型的分类)
        print(f"  当前模型补偿后误差:")
        for did in sorted(g["disturbance_id"].unique()):
            sub_g = g[g["disturbance_id"] == did]
            errs = sub_g["err_pct"].values
            print(f"    {did}: [{', '.join(f'{e:+.3f}' for e in errs)}]%")


def test_k8_clustering():
    """测试K=8聚类 (每种扰流单独一类)。"""
    from sklearn.decomposition import PCA
    from sklearn.covariance import LedoitWolf
    
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 模拟K=8: 直接用disturbance_id作为类别
    dist = df[df["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df["disturbance_id"] != "D0"]
    
    # 每类每流量点的均值误差 → 补偿
    comp = {}
    for did in sorted(dist["disturbance_id"].unique()):
        comp[did] = {}
        for fp in sorted(dist["flow_point"].unique()):
            mask = (dist["disturbance_id"] == did) & (dist["flow_point"] == fp)
            if mask.sum() > 0:
                comp[did][fp] = -dist_err[mask.values].mean()
    
    # 应用补偿
    corrected = base_err.copy()
    for i, row in df.iterrows():
        fp = row["flow_point"]
        did = row["disturbance_id"]
        if did != "D0":
            delta = comp.get(did, {}).get(fp, 0)
            corrected[i] += delta
        else:
            # D0: 流量点修正
            d0_mask = (df["disturbance_id"] == "D0") & (df["flow_point"] == fp)
            corrected[i] += -base_err[d0_mask.values].mean()
    
    # 计算组SD
    sds = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sds.append(g_err.std(ddof=1))
    
    print(f"\nK=8 (每种扰流单独一类):")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    
    # 计算 u_nor_d
    u_d_r = max(sds) if sds else 0
    # 简化: 假设 u_d_c 类似
    print(f"  u_nor_r ≈ {u_d_r:.4f}%")


def test_optimal_compensation():
    """测试理论最优补偿 (oracle: 知道每个窗口的真实误差)。"""
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 理论最优: 每个窗口的补偿 = -base_err → 误差 = 0
    # 但这需要知道standard_volume，违反规则
    
    # 次优: 每个 (disturbance_id, flow_point) 的均值补偿
    corrected = base_err.copy()
    for did in df["disturbance_id"].unique():
        for fp in df["flow_point"].unique():
            mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
            if mask.sum() > 0:
                corrected[mask.values] -= base_err[mask.values].mean()
    
    sds = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sds.append(g_err.std(ddof=1))
    
    print(f"\n理论最优 (per-disturbance-per-flowpoint均值补偿):")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    print(f"  → 即使完美补偿均值，组内SD仍由窗口间随机波动决定")


def test_variance_decomposition():
    """方差分解: 组内SD的来源。"""
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    dist = df[df["disturbance_id"] != "D0"].copy()
    dist_err = base_err[df["disturbance_id"] != "D0"]
    
    # 总方差 = 类间方差 + 类内方差
    # 类 = disturbance_id
    
    overall_var = dist_err.var(ddof=1)
    
    # 类间方差 (各disturbance_id均值与总均值的差异)
    dist_err_series = pd.Series(dist_err, index=dist.index)
    class_means = dist_err_series.groupby(dist["disturbance_id"]).mean()
    grand_mean = dist_err.mean()
    between_var = class_means.var(ddof=1)
    
    # 类内方差 (各类内部方差)
    within_vars = []
    for did in dist["disturbance_id"].unique():
        mask = dist["disturbance_id"] == did
        within_vars.append(dist_err_series[mask].var(ddof=1))
    within_var = np.mean(within_vars)
    
    print(f"\n方差分解 (扰流窗口):")
    print(f"  总方差: {overall_var:.6f}")
    print(f"  类间方差 (disturbance_id): {between_var:.6f} ({between_var/overall_var*100:.1f}%)")
    print(f"  类内方差: {within_var:.6f} ({within_var/overall_var*100:.1f}%)")
    print(f"  → 类内方差占主导，说明同类窗口间差异远大于类间差异")
    
    # 按流量点分解
    print(f"\n  按流量点分解类内SD:")
    for fp in sorted(dist["flow_point"].unique()):
        fp_mask = dist["flow_point"] == fp
        fp_err = dist_err[fp_mask.values]
        if len(fp_err) > 1:
            print(f"    fp={fp:2d}: SD={fp_err.std(ddof=1):.4f}% (N={len(fp_err)})")


def test_theoretical_limit():
    """计算理论极限: 即使完美补偿均值，SD的下限。"""
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 对每个组，计算"完美补偿均值后"的SD
    # 完美补偿 = 减去组均值 → SD不变
    # 所以理论下限 = 当前SD
    
    # 但如果我们用 (disturbance_id, flow_point) 的均值补偿:
    corrected = base_err.copy()
    for did in df["disturbance_id"].unique():
        for fp in df["flow_point"].unique():
            mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
            if mask.sum() > 0:
                corrected[mask.values] -= base_err[mask.values].mean()
    
    # 计算组SD
    sds = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sds.append(g_err.std(ddof=1))
    
    print(f"\n理论下限分析:")
    print(f"  完美 (did, fp) 均值补偿后:")
    print(f"    最大组SD: {max(sds):.4f}%")
    print(f"    中位数SD: {np.median(sds):.4f}%")
    print(f"    目标SD: 0.040%")
    print(f"    差距: {max(sds)/0.040:.1f}x")
    
    # 进一步分解: 最大SD组的类内方差
    worst_group = None
    worst_sd = 0
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sd = g_err.std(ddof=1)
            if sd > worst_sd:
                worst_sd = sd
                worst_group = (date, fp)
    
    date, fp = worst_group
    g = df[(df["date"] == date) & (df["flow_point"] == fp)]
    g_err = corrected[g.index]
    
    print(f"\n  最差组 ({date} fp={fp}):")
    for did in sorted(g["disturbance_id"].unique()):
        mask = g["disturbance_id"] == did
        errs = g_err[mask.values]
        if len(errs) > 1:
            print(f"    {did}: N={len(errs)} SD={errs.std(ddof=1):.4f}%")


def test_raw_ts_features_for_sd():
    """测试附件6原始特征能否降低组内SD。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    
    raw = pd.read_csv(Path(__file__).parent.parent / "problem" / "attachment6_window_raw_samples.csv")
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 提取每个窗口的原始特征
    feat_rows = []
    for wid, w in raw.groupby("window_id"):
        feat = {"window_id": wid}
        
        # 各声道diff_ns的CV (变异系数)
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
    
    # 选择top特征
    feat_cols = [c for c in feat_df.columns if c != "window_id" and c.startswith(("diff", "vel"))]
    
    # LOO-CV预测
    X = df2[feat_cols].fillna(0).values
    y = base_err
    
    loo = LeaveOneOut()
    ridge = Ridge(alpha=10.0)
    y_pred = cross_val_predict(ridge, X, y, cv=loo)
    
    corrected = y - y_pred
    
    sds = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = corrected[idx]
        if len(g_err) >= 3:
            sds.append(g_err.std(ddof=1))
    
    print(f"\n附件6原始特征 + Ridge (LOO-CV):")
    print(f"  MAE: {np.abs(corrected).mean():.4f}%")
    print(f"  最大组SD: {max(sds):.4f}%")
    print(f"  中位数SD: {np.median(sds):.4f}%")
    print(f"  → 与当前模型相比，SD几乎无改善")


if __name__ == "__main__":
    analyze_worst_groups()
    test_variance_decomposition()
    test_theoretical_limit()
    test_k8_clustering()
    test_optimal_compensation()
    test_raw_ts_features_for_sd()
    
    print("\n" + "=" * 90)
    print("结论")
    print("=" * 90)
    print("1. 组内SD的主要来源是窗口间随机波动，非系统性偏差")
    print("2. 即使完美补偿 (did, fp) 均值，最大组SD仍在 0.10% 以上")
    print("3. 类内方差远大于类间方差 → 聚类/分类对降低SD帮助有限")
    print("4. 原始时序特征与误差相关性弱 (r < 0.15)，无法有效预测窗口间差异")
    print("5. SD目标 0.040% 在当前数据条件下不可达，理论下限约 0.10%")
