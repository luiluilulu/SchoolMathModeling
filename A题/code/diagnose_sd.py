"""诊断SD过高的根因：分析各组误差分布和特征差异。"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_attachment1, RESULTS_DIR

def analyze_group_sd():
    """分析每组的SD和误差分布。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df = df.merge(sub, on="window_id")
    df["error_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    
    # 按日期+流量点分组
    groups = []
    for (date, fp), g in df.groupby(["date", "flow_point"]):
        if len(g) < 3:
            continue
        errors = g["error_pct"].values
        mean_err = errors.mean()
        sd = errors.std(ddof=1)
        range_err = errors.max() - errors.min()
        
        # 分析组内特征差异
        chord_vars = [g[f"chord{i}"].var() for i in range(5)]
        ab_vars = [g[f"ab{i}"].var() for i in range(5)]
        profile_vars = {
            "profile_ab_abs_var": g["profile_ab_abs"].var(),
            "profile_swirl_var": g["profile_swirl"].var(),
            "dyn_plateau_cv_var": g["dyn_plateau_cv"].var(),
            "zero_age_var": g["zero_age_s"].var(),
        }
        
        groups.append({
            "date": date,
            "flow_point": fp,
            "n": len(g),
            "mean_error": mean_err,
            "sd": sd,
            "range": range_err,
            "pass": abs(mean_err) <= 0.2 and sd <= 0.040,
            "chord_var_mean": np.mean(chord_vars),
            "ab_var_mean": np.mean(ab_vars),
            **profile_vars,
        })
    
    gdf = pd.DataFrame(groups)
    
    print("=" * 80)
    print("SD过高的组 (SD > 0.040%)")
    print("=" * 80)
    bad = gdf[gdf["sd"] > 0.040].sort_values("sd", ascending=False)
    for _, row in bad.iterrows():
        print(f"日期={row['date']} 流量点={row['flow_point']:2d} "
              f"N={row['n']} SD={row['sd']:.4f}% "
              f"均值={row['mean_error']:+.4f}% "
              f"极差={row['range']:.4f}%")
    
    print(f"\n未通过组数: {len(bad)}/{len(gdf)}")
    print(f"SD最大值: {gdf['sd'].max():.4f}%")
    print(f"SD中位数: {gdf['sd'].median():.4f}%")
    
    print("\n" + "=" * 80)
    print("特征方差与SD的相关性")
    print("=" * 80)
    for col in ["chord_var_mean", "ab_var_mean", "profile_ab_abs_var", 
                "profile_swirl_var", "dyn_plateau_cv_var", "zero_age_var"]:
        corr = gdf["sd"].corr(gdf[col])
        print(f"{col:25s}: r = {corr:+.3f}")
    
    return gdf

def analyze_window_outliers():
    """分析组内异常窗口。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df = df.merge(sub, on="window_id")
    df["error_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    
    print("\n" + "=" * 80)
    print("SD最高组内的窗口详情")
    print("=" * 80)
    
    # 找到SD最高的组
    worst_group = None
    worst_sd = 0
    for (date, fp), g in df.groupby(["date", "flow_point"]):
        if len(g) < 3:
            continue
        sd = g["error_pct"].std(ddof=1)
        if sd > worst_sd:
            worst_sd = sd
            worst_group = (date, fp)
    
    date, fp = worst_group
    g = df[(df["date"] == date) & (df["flow_point"] == fp)]
    print(f"\n日期={date} 流量点={fp} SD={worst_sd:.4f}%")
    print("-" * 80)
    
    cols = ["window_id", "error_pct", "disturbance_id", "profile_ab_abs", 
            "profile_swirl", "dyn_plateau_cv", "zero_age_s"]
    print(g[cols].to_string(index=False))
    
    # 分析该组内窗口的特征差异
    print(f"\n组内特征范围:")
    for col in ["chord0", "chord1", "chord2", "chord3", "chord4", 
                "profile_ab_abs", "profile_swirl", "dyn_plateau_cv"]:
        vals = g[col].values
        print(f"  {col:20s}: [{vals.min():.4f}, {vals.max():.4f}] "
              f"range={vals.max()-vals.min():.4f}")

if __name__ == "__main__":
    gdf = analyze_group_sd()
    analyze_window_outliers()
