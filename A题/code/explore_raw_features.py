"""从附件6原始时序数据中提取窗口级稳定性特征，探索能否降低SD。"""
import numpy as np
import pandas as pd
from pathlib import Path
from utils import load_attachment1, RESULTS_DIR

DATA_DIR = Path(__file__).parent.parent / "problem"
AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
DIFF_COLS = [f"diff_ns_{i}" for i in range(10)]
MEAN_COLS = [f"mean_us_{i}" for i in range(10)]

def extract_raw_features():
    """从附件6提取每个窗口的稳定性特征。"""
    raw = pd.read_csv(DATA_DIR / "attachment6_window_raw_samples.csv")
    
    rows = []
    for wid, w in raw.groupby("window_id"):
        w = w.sort_values("rel_time_s")
        feat = {"window_id": wid}
        
        # 1. 各声道diff_ns的统计量
        for i in range(10):
            col = f"diff_ns_{i}"
            vals = w[col].dropna().values
            if len(vals) > 1:
                feat[f"diff{i}_mean"] = vals.mean()
                feat[f"diff{i}_std"] = vals.std(ddof=1)
                feat[f"diff{i}_cv"] = vals.std(ddof=1) / (abs(vals.mean()) + 1e-12)
                feat[f"diff{i}_range"] = vals.max() - vals.min()
                feat[f"diff{i}_skew"] = pd.Series(vals).skew()
                feat[f"diff{i}_kurt"] = pd.Series(vals).kurtosis()
            else:
                feat[f"diff{i}_mean"] = np.nan
                feat[f"diff{i}_std"] = 0
                feat[f"diff{i}_cv"] = 0
                feat[f"diff{i}_range"] = 0
                feat[f"diff{i}_skew"] = 0
                feat[f"diff{i}_kurt"] = 0
        
        # 2. 流速统计量
        if "flow_velocity_m_s" in w.columns:
            v = w["flow_velocity_m_s"].dropna().values
            if len(v) > 1:
                feat["vel_mean"] = v.mean()
                feat["vel_std"] = v.std(ddof=1)
                feat["vel_cv"] = v.std(ddof=1) / (abs(v.mean()) + 1e-12)
                feat["vel_range"] = v.max() - v.min()
                feat["vel_skew"] = pd.Series(v).skew()
                # 前后半段差异 (趋势)
                mid = len(v) // 2
                feat["vel_trend"] = v[mid:].mean() - v[:mid].mean()
            else:
                feat["vel_mean"] = np.nan
                feat["vel_std"] = 0
                feat["vel_cv"] = 0
                feat["vel_range"] = 0
                feat["vel_skew"] = 0
                feat["vel_trend"] = 0
        
        # 3. 水温统计量
        if "water_temp_c" in w.columns:
            t = w["water_temp_c"].dropna().values
            if len(t) > 1:
                feat["temp_std"] = t.std(ddof=1)
                feat["temp_range"] = t.max() - t.min()
            else:
                feat["temp_std"] = 0
                feat["temp_range"] = 0
        
        # 4. 声道间差异的稳定性
        # A组 (0-4) vs B组 (5-9) 的diff_ns差异
        a_means = [w[f"diff_ns_{i}"].dropna().mean() for i in range(5)]
        b_means = [w[f"diff_ns_{i+5}"].dropna().mean() for i in range(5)]
        feat["ab_diff_mean"] = np.mean([abs(a - b) for a, b in zip(a_means, b_means)])
        feat["ab_diff_std"] = np.std([abs(a - b) for a, b in zip(a_means, b_means)])
        
        rows.append(feat)
    
    return pd.DataFrame(rows)

def analyze_raw_feature_correlation():
    """分析原始特征与误差的相关性。"""
    df = load_attachment1()
    sub = pd.read_csv(RESULTS_DIR / "problem4_submission.csv")
    df = df.merge(sub, on="window_id")
    df["err_pct"] = (df["model_volume_m3"] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    
    raw_feat = extract_raw_features()
    df2 = df.merge(raw_feat, on="window_id")
    
    # 只看扰流窗口
    dist = df2[df2["disturbance_id"] != "D0"].copy()
    
    # 去流量点均值后的误差
    fp_mean = dist.groupby("flow_point")["err_pct"].transform("mean")
    dist["err_centered"] = dist["err_pct"] - fp_mean
    
    print("=" * 80)
    print("原始时序特征与误差的相关性 (扰流窗口, 流量点内)")
    print("=" * 80)
    
    # 收集所有数值特征
    feat_cols = [c for c in raw_feat.columns if c != "window_id"]
    
    corrs = []
    for f in feat_cols:
        valid = dist[[f, "err_centered"]].dropna()
        if len(valid) > 10 and valid[f].std() > 1e-10:
            r = valid["err_centered"].corr(valid[f])
            corrs.append((f, r))
    
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'特征':30s} {'相关系数':>10s}")
    print("-" * 45)
    for f, r in corrs[:30]:
        print(f"{f:30s} {r:+.4f}")
    
    print(f"\n共 {len(corrs)} 个有效特征")
    
    # 也看全局相关性
    print("\n" + "=" * 80)
    print("原始时序特征与误差的全局相关性 (扰流窗口)")
    print("=" * 80)
    corrs_global = []
    for f in feat_cols:
        valid = dist[[f, "err_pct"]].dropna()
        if len(valid) > 10 and valid[f].std() > 1e-10:
            r = valid["err_pct"].corr(valid[f])
            corrs_global.append((f, r))
    
    corrs_global.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'特征':30s} {'相关系数':>10s}")
    print("-" * 45)
    for f, r in corrs_global[:30]:
        print(f"{f:30s} {r:+.4f}")
    
    return df2, dist

def test_regression_reduction(df2):
    """测试回归模型能否降低SD。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    
    dist = df2[df2["disturbance_id"] != "D0"].copy()
    fp_mean = dist.groupby("flow_point")["err_pct"].transform("mean")
    dist["err_centered"] = dist["err_pct"] - fp_mean
    
    # 选top特征
    feat_cols = [c for c in df2.columns if c.startswith("diff") or c.startswith("vel") or c.startswith("temp") or c.startswith("ab_diff")]
    
    # 去流量点均值后的特征
    for c in feat_cols:
        dist[f"{c}_c"] = dist[c] - dist.groupby("flow_point")[c].transform("mean")
    
    feat_cols_c = [f"{c}_c" for c in feat_cols]
    
    X = dist[feat_cols_c].fillna(0).values
    y = dist["err_centered"].values
    
    # 用Ridge回归 + LOO-CV预测
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    ridge = Ridge(alpha=1.0)
    y_pred = cross_val_predict(ridge, X, y, cv=loo)
    
    resid = y - y_pred
    print(f"\n回归前 err_centered std: {y.std(ddof=1):.4f}%")
    print(f"回归后 residual std: {resid.std(ddof=1):.4f}%")
    print(f"SD降低: {(1 - resid.std(ddof=1)/y.std(ddof=1))*100:.1f}%")
    
    # 也测试不同alpha
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        y_pred = cross_val_predict(ridge, X, y, cv=loo)
        resid = y - y_pred
        print(f"  alpha={alpha:6.2f}: residual std = {resid.std(ddof=1):.4f}%")

if __name__ == "__main__":
    df2, dist = analyze_raw_feature_correlation()
    test_regression_reduction(df2)
