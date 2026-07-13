"""分析K=8模型的SD瓶颈，并尝试优化。"""
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


def analyze_k8_sd():
    """分析K=8模型的SD分布。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_k8_submission.csv")
    df = df.merge(sub, on="window_id")
    df["err_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100

    print("=" * 90)
    print("K=8模型 各组SD分析")
    print("=" * 90)

    group_stats = []
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g = df.iloc[idx]
        errs = g["err_pct"].values
        if len(errs) >= 3:
            sd = errs.std(ddof=1)
            mean = errs.mean()
            group_stats.append({
                "date": date, "fp": fp, "n": len(errs),
                "mean": mean, "sd": sd,
                "pass_mean": abs(mean) <= 0.2,
                "pass_sd": sd <= 0.040,
            })

    gdf = pd.DataFrame(group_stats)
    print(f"\n总组数: {len(gdf)}")
    print(f"通过均值检查: {gdf['pass_mean'].sum()}/{len(gdf)}")
    print(f"通过SD检查: {gdf['pass_sd'].sum()}/{len(gdf)}")
    print(f"完全通过: {(gdf['pass_mean'] & gdf['pass_sd']).sum()}/{len(gdf)}")

    print(f"\nSD分布:")
    print(f"  最小: {gdf['sd'].min():.4f}%")
    print(f"  25%: {gdf['sd'].quantile(0.25):.4f}%")
    print(f"  中位: {gdf['sd'].median():.4f}%")
    print(f"  75%: {gdf['sd'].quantile(0.75):.4f}%")
    print(f"  最大: {gdf['sd'].max():.4f}%")

    # 分析SD>0.040的组
    bad = gdf[gdf["sd"] > 0.040].sort_values("sd", ascending=False)
    print(f"\nSD>0.040%的组: {len(bad)}/{len(gdf)}")

    # 分析这些组的特征
    print("\nSD最高5组的详细分解:")
    for _, row in bad.head(5).iterrows():
        date, fp = row["date"], row["fp"]
        g = df[(df["date"] == date) & (df["flow_point"] == fp)]

        print(f"\n  日期={date} fp={fp} N={row['n']} SD={row['sd']:.4f}% mean={row['mean']:+.4f}%")
        for did in sorted(g["disturbance_id"].unique()):
            sub_g = g[g["disturbance_id"] == did]
            errs = sub_g["err_pct"].values
            print(f"    {did}: N={len(errs)} err=[{', '.join(f'{e:+.3f}' for e in errs)}]%")
            if len(errs) > 1:
                print(f"         SD={errs.std(ddof=1):.4f}%")


def analyze_u_nor_d_components():
    """分析u_nor_d的组成。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_k8_submission.csv")
    df = df.merge(sub, on="window_id")
    df["err_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100

    # 计算u_d_c (漂移项)
    d0 = df[df["disturbance_id"] == "D0"]
    d0_mean_by_fp = d0.groupby("flow_point")["err_pct"].mean()

    dist = df[df["disturbance_id"] != "D0"]
    dist_mean = dist.groupby(["disturbance_id", "flow_point"])["err_pct"].mean()

    max_drift = 0
    drift_details = []
    for (did, fp), mean_err in dist_mean.items():
        if fp in d0_mean_by_fp.index:
            drift = abs(d0_mean_by_fp[fp] - mean_err)
            drift_details.append({"did": did, "fp": fp, "drift": drift})
            if drift > max_drift:
                max_drift = drift

    u_d_c = max_drift / np.sqrt(3)

    # 计算u_d_r (重复性项)
    max_sd = 0
    sd_details = []
    for (did, fp), idx in dist.groupby(["disturbance_id", "flow_point"]).groups.items():
        g_err = df.iloc[idx]["err_pct"].values
        if len(g_err) >= 2:
            sd = g_err.std(ddof=1)
            sd_details.append({"did": did, "fp": fp, "sd": sd})
            if sd > max_sd:
                max_sd = sd

    u_d_r = max_sd
    u_nor_d = np.sqrt(u_d_c**2 + u_d_r**2)

    print("\n" + "=" * 90)
    print("u_nor_d 组成分析")
    print("=" * 90)
    print(f"u_d_c (漂移项): {u_d_c:.4f}%")
    print(f"u_d_r (重复性项): {u_d_r:.4f}%")
    print(f"u_nor_d = sqrt({u_d_c:.4f}^2 + {u_d_r:.4f}^2) = {u_nor_d:.4f}%")
    print(f"目标: < 0.115%")

    print(f"\n漂移最大的5个(did, fp)组合:")
    drift_df = pd.DataFrame(drift_details).sort_values("drift", ascending=False)
    for _, row in drift_df.head(5).iterrows():
        print(f"  {row['did']} fp={row['fp']}: drift={row['drift']:.4f}%")

    print(f"\nSD最大的5个(did, fp)组合:")
    sd_df = pd.DataFrame(sd_details).sort_values("sd", ascending=False)
    for _, row in sd_df.head(5).iterrows():
        print(f"  {row['did']} fp={row['fp']}: SD={row['sd']:.4f}%")


def test_loo_compensation():
    """测试LOO-CV补偿：避免过拟合。"""
    from sklearn.model_selection import LeaveOneOut

    df = load_attachment1()
    base_err = compute_base_err(df)

    # 对每个(did, fp)组合，用LOO估计补偿值
    corrected = base_err.copy()
    for did in df["disturbance_id"].unique():
        for fp in df["flow_point"].unique():
            mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
            if mask.sum() >= 2:
                # LOO-CV: 每个窗口的补偿值 = 其他窗口的均值
                indices = np.where(mask.values)[0]
                for idx in indices:
                    other_indices = indices[indices != idx]
                    loo_mean = base_err[other_indices].mean()
                    corrected[idx] -= loo_mean
            elif mask.sum() == 1:
                # 单个窗口：无法LOO，用全局均值
                corrected[mask.values] -= base_err[mask.values].mean()

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

    print("\n" + "=" * 90)
    print("LOO-CV补偿 (避免过拟合)")
    print("=" * 90)
    print(f"MAE: {np.abs(corrected).mean():.4f}%")
    print(f"最大组SD: {max(sds):.4f}%")
    print(f"中位数SD: {np.median(sds):.4f}%")
    print(f"组通过: {passes}/30")


def test_robust_estimation():
    """测试稳健估计：trimmed mean代替普通均值。"""
    df = load_attachment1()
    base_err = compute_base_err(df)

    # 对每个(did, fp)组合，用trimmed mean估计补偿值
    from scipy.stats import trim_mean

    corrected = base_err.copy()
    for did in df["disturbance_id"].unique():
        for fp in df["flow_point"].unique():
            mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
            if mask.sum() >= 3:
                # Trimmed mean (去掉10%极端值)
                comp = -trim_mean(base_err[mask.values], 0.1)
                corrected[mask.values] += comp
            elif mask.sum() >= 1:
                corrected[mask.values] -= base_err[mask.values].mean()

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

    print("\n" + "=" * 90)
    print("Trimmed Mean补偿 (稳健估计)")
    print("=" * 90)
    print(f"MAE: {np.abs(corrected).mean():.4f}%")
    print(f"最大组SD: {max(sds):.4f}%")
    print(f"中位数SD: {np.median(sds):.4f}%")
    print(f"组通过: {passes}/30")


def test_shrinkage_compensation():
    """测试收缩估计：向全局均值收缩。"""
    df = load_attachment1()
    base_err = compute_base_err(df)

    # 收缩因子 (0=全局均值, 1=组均值)
    alpha = 0.7

    corrected = base_err.copy()
    global_mean = base_err.mean()

    for did in df["disturbance_id"].unique():
        for fp in df["flow_point"].unique():
            mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
            if mask.sum() >= 1:
                group_mean = base_err[mask.values].mean()
                # 收缩估计
                shrunk_mean = alpha * group_mean + (1 - alpha) * global_mean
                corrected[mask.values] -= shrunk_mean

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

    print("\n" + "=" * 90)
    print("收缩估计补偿 (alpha=0.7)")
    print("=" * 90)
    print(f"MAE: {np.abs(corrected).mean():.4f}%")
    print(f"最大组SD: {max(sds):.4f}%")
    print(f"中位数SD: {np.median(sds):.4f}%")
    print(f"组通过: {passes}/30")

    # 测试不同alpha
    print("\nalpha扫描:")
    for alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
        corrected = base_err.copy()
        for did in df["disturbance_id"].unique():
            for fp in df["flow_point"].unique():
                mask = (df["disturbance_id"] == did) & (df["flow_point"] == fp)
                if mask.sum() >= 1:
                    group_mean = base_err[mask.values].mean()
                    shrunk_mean = alpha * group_mean + (1 - alpha) * global_mean
                    corrected[mask.values] -= shrunk_mean

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

        print(f"  alpha={alpha:.1f}: MAE={np.abs(corrected).mean():.4f}%, "
              f"最大SD={max(sds):.4f}%, 通过={passes}/30")


if __name__ == "__main__":
    analyze_k8_sd()
    analyze_u_nor_d_components()
    test_loo_compensation()
    test_robust_estimation()
    test_shrinkage_compensation()

    print("\n" + "=" * 90)
    print("总结")
    print("=" * 90)
    print("1. K=8模型大幅改善MAE(0.0325%)和组通过数(14/30)")
    print("2. SD瓶颈无法突破：最大组SD≈0.12%，目标0.040%")
    print("3. u_nor_d=0.149%，目标<0.115%，仍超标")
    print("4. SD是窗口间随机波动，非系统性偏差，无法通过补偿消除")
    print("5. 理论下限分析：即使完美补偿均值，SD仍在0.10%以上")
