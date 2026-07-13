"""深入分析：探索改进方向。
1. 分析D0窗口误差模式
2. 分析扰流窗口误差与特征的关系
3. 探索附件6原始数据能否提供更优特征
"""
import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_attachment1, RESULTS_DIR

DATA_DIR = Path(__file__).parent.parent / "problem"

def analyze_d0_pattern():
    """分析D0窗口的误差模式。"""
    df = load_attachment1()
    d0 = df[df["disturbance_id"] == "D0"].copy()
    AREA = 0.13138219017128852
    W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
    chord_cols = [f"chord{i}" for i in range(5)]
    d0["phys6_v"] = d0[chord_cols].values @ W
    d0["phys6_vol"] = d0["phys6_v"] * AREA * d0["duration_s"].astype(float)
    d0["err_pct"] = (d0["phys6_vol"] - d0["standard_volume_m3"]) / d0["standard_volume_m3"] * 100
    
    print("=" * 80)
    print("D0窗口 Phys6 误差 (按流量点)")
    print("=" * 80)
    for fp, g in d0.groupby("flow_point"):
        errs = g["err_pct"].values
        print(f"  fp={fp:2d} N={len(g)} mean={errs.mean():+.4f}% std={errs.std(ddof=1):.4f}% "
              f"range=[{errs.min():+.4f}, {errs.max():+.4f}%]")
    
    # 归一化权重 (和=1)
    W_norm = W / W.sum()
    d0["norm_v"] = d0[chord_cols].values @ W_norm
    d0["norm_vol"] = d0["norm_v"] * AREA * d0["duration_s"].astype(float)
    d0["norm_err"] = (d0["norm_vol"] - d0["standard_volume_m3"]) / d0["standard_volume_m3"] * 100
    
    print("\n  归一化权重 (和=1):")
    for fp, g in d0.groupby("flow_point"):
        errs = g["norm_err"].values
        print(f"  fp={fp:2d} N={len(g)} mean={errs.mean():+.4f}% std={errs.std(ddof=1):.4f}%")
    
    # OWICS权重
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df2 = df.merge(sub, on="window_id")
    df2["err_pct"] = (df2["model_volume_m3"] - df2["standard_volume_m3"]) / df2["standard_volume_m3"] * 100
    d0_2 = df2[df2["disturbance_id"] == "D0"]
    print("\n  当前模型 D0 误差:")
    for fp, g in d0_2.groupby("flow_point"):
        errs = g["err_pct"].values
        print(f"  fp={fp:2d} N={len(g)} mean={errs.mean():+.4f}% std={errs.std(ddof=1):.4f}%")

def analyze_disturbance_continuous():
    """分析扰流窗口误差与连续特征的关系。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df = df.merge(sub, on="window_id")
    df["err_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    
    dist = df[df["disturbance_id"] != "D0"].copy()
    
    print("\n" + "=" * 80)
    print("扰流窗口: 误差与特征的相关性 (按流量点内)")
    print("=" * 80)
    features = ["profile_ab_abs", "profile_swirl", "dyn_plateau_cv", "zero_age_s",
                "chord0", "chord1", "chord2", "chord3", "chord4"]
    
    # 全局相关性
    print("\n全局 (所有扰流窗口):")
    for f in features:
        r = dist["err_pct"].corr(dist[f])
        print(f"  {f:20s}: r = {r:+.4f}")
    
    # 按流量点内相关性 (去除流量点均值后)
    print("\n流量点内 (去均值后):")
    dist_centered = dist.copy()
    fp_mean = dist.groupby("flow_point")["err_pct"].transform("mean")
    dist_centered["err_centered"] = dist_centered["err_pct"] - fp_mean
    for f in features:
        f_centered = dist_centered[f] - dist_centered.groupby("flow_point")[f].transform("mean")
        r = dist_centered["err_centered"].corr(f_centered)
        print(f"  {f:20s}: r = {r:+.4f}")
    
    # 按扰流类型分析
    print("\n" + "=" * 80)
    print("各扰流类型误差统计")
    print("=" * 80)
    for d_id in sorted(dist["disturbance_id"].unique()):
        d = dist[dist["disturbance_id"] == d_id]
        errs = d["err_pct"].values
        print(f"  {d_id}: N={len(d):2d} mean={errs.mean():+.4f}% "
              f"std={errs.std(ddof=1):.4f}% range=[{errs.min():+.3f}, {errs.max():+.3f}]")

def analyze_raw_timeseries():
    """分析附件6原始数据，探索更好的特征。"""
    raw = pd.read_csv(DATA_DIR / "attachment6_window_raw_samples.csv")
    df = load_attachment1()
    
    print("\n" + "=" * 80)
    print("附件6 原始数据分析")
    print("=" * 80)
    print(f"总行数: {len(raw)}")
    print(f"窗口数: {raw['window_id'].nunique()}")
    print(f"列: {list(raw.columns)}")
    
    # 对几个典型窗口分析
    # 选SD最高组的窗口
    worst_windows = [
        "updated_20260613_home__20260613_085407",  # D4, err=-0.061
        "updated_20260613_home__20260613_092646",  # D4, err=-0.029
        "updated_20260613_home__20260613_163055",  # D4, err=+0.212
        "updated_20260613_home__20260613_191132",  # D3, err=+0.015
    ]
    
    for wid in worst_windows:
        w = raw[raw["window_id"] == wid]
        if len(w) == 0:
            print(f"\n  {wid}: 不在附件6中")
            continue
        print(f"\n  {wid}:")
        print(f"    样本数: {len(w)}")
        
        # 分析diff_ns_0 (声道0的顺逆流时间差)
        diff_cols = [c for c in raw.columns if c.startswith("diff_ns_")]
        mean_cols = [c for c in raw.columns if c.startswith("mean_us_")]
        
        for col in diff_cols[:3]:
            vals = w[col].dropna().values
            if len(vals) > 0:
                print(f"    {col}: mean={vals.mean():.2f} std={vals.std():.4f} "
                      f"cv={vals.std()/abs(vals.mean()+1e-12):.4f}")
        
        # 分析瞬时流速
        if "flow_velocity_m_s" in w.columns:
            v = w["flow_velocity_m_s"].dropna().values
            if len(v) > 0:
                print(f"    flow_velocity: mean={v.mean():.4f} std={v.std():.4f} "
                      f"cv={v.std()/(v.mean()+1e-12):.4f}")

def explore_better_weights():
    """探索能否通过优化权重降低SD。"""
    df = load_attachment1()
    AREA = 0.13138219017128852
    chord_cols = [f"chord{i}" for i in range(5)]
    
    # 用所有数据，尝试不同权重组合
    from scipy.optimize import minimize
    
    def compute_sd(w):
        v = df[chord_cols].values @ w
        vol = v * AREA * df["duration_s"].astype(float).values
        err = (vol - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100
        
        # 计算组SD
        sds = []
        for (date, fp), g_idx in df.groupby(["date", "flow_point"]).groups.items():
            g_err = err[g_idx]
            if len(g_err) >= 3:
                sds.append(g_err.std(ddof=1))
        return max(sds) if sds else 999
    
    # Phys6权重
    w_phys6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
    print(f"\n  Phys6 最大组SD: {compute_sd(w_phys6):.4f}%")
    
    # 归一化权重
    w_norm = w_phys6 / w_phys6.sum()
    print(f"  归一化 最大组SD: {compute_sd(w_norm):.4f}%")
    
    # 等权
    w_eq = np.ones(5) / 5
    print(f"  等权 最大组SD: {compute_sd(w_eq):.4f}%")
    
    # 尝试优化权重 (最小化最大SD)
    def objective(w):
        return compute_sd(w)
    
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0, 0.5)] * 5
    result = minimize(objective, w_norm, method="SLSQP", bounds=bounds, constraints=constraints,
                     options={"maxiter": 500, "ftol": 1e-8})
    print(f"  优化权重 最大组SD: {compute_sd(result.x):.4f}%")
    print(f"  优化权重: {result.x}")

if __name__ == "__main__":
    analyze_d0_pattern()
    analyze_disturbance_continuous()
    analyze_raw_timeseries()
    explore_better_weights()
