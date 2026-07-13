"""测试多种改进方案，找出最优策略。"""
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


def compute_group_sd(errors, df):
    """计算所有组的SD，返回最大SD和通过数。"""
    sds = []
    passes = 0
    for (date, fp), idx in df.groupby(["date", "flow_point"]).groups.items():
        g_err = errors[idx]
        if len(g_err) >= 3:
            sd = g_err.std(ddof=1)
            mean = g_err.mean()
            sds.append(sd)
            if abs(mean) <= 0.2 and sd <= 0.040:
                passes += 1
    return max(sds) if sds else 0, passes, np.median(sds) if sds else 0


def test_no_compensation():
    """测试无补偿的基线。"""
    df = load_attachment1()
    err = compute_base_err(df)
    max_sd, passes, med_sd = compute_group_sd(err, df)
    print(f"无补偿 (Phys6基线):")
    print(f"  MAE: {np.abs(err).mean():.4f}%")
    print(f"  最大组SD: {max_sd:.4f}%, 中位数SD: {med_sd:.4f}%")
    print(f"  组通过: {passes}/30")
    return err


def test_continuous_compensation():
    """测试连续特征补偿。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 扰流窗口
    dist_mask = df["disturbance_id"] != "D0"
    dist_idx = np.where(dist_mask)[0]
    
    # 特征: profile_ab_abs, profile_swirl, 流量点哑变量
    pab = df["profile_ab_abs"].astype(float).values
    psw = df["profile_swirl"].astype(float).values
    fp_vals = df["flow_point"].values
    
    # 构建特征矩阵 (所有窗口)
    fp_dummies = np.zeros((len(df), 6))  # 6个哑变量 (7个流量点, drop first)
    fp_unique = sorted(df["flow_point"].unique())
    for i, fp in enumerate(fp_vals):
        if fp in fp_unique and fp_unique.index(fp) > 0:
            fp_dummies[i, fp_unique.index(fp) - 1] = 1.0
    
    X = np.column_stack([pab, psw, pab * psw, pab**2, psw**2, fp_dummies])
    y = base_err
    
    # LOO-CV预测 (避免过拟合)
    loo = LeaveOneOut()
    ridge = Ridge(alpha=10.0)
    y_pred_cv = cross_val_predict(ridge, X, y, cv=loo)
    
    # 补偿后的误差
    err_corrected = y - y_pred_cv
    max_sd, passes, med_sd = compute_group_sd(err_corrected, df)
    
    print(f"\n连续补偿 (Ridge, LOO-CV):")
    print(f"  MAE: {np.abs(err_corrected).mean():.4f}%")
    print(f"  最大组SD: {max_sd:.4f}%, 中位数SD: {med_sd:.4f}%")
    print(f"  组通过: {passes}/30")
    
    # 也测试不同alpha
    print("\n  alpha扫描:")
    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        y_pred_cv = cross_val_predict(ridge, X, y, cv=loo)
        err_c = y - y_pred_cv
        max_sd, passes, med_sd = compute_group_sd(err_c, df)
        print(f"    alpha={alpha:6.1f}: MAE={np.abs(err_c).mean():.4f}%, "
              f"最大SD={max_sd:.4f}%, 通过={passes}/30")
    
    return err_corrected


def test_separate_d0_dist():
    """测试D0和扰流分开建模。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # D0: 流量点均值修正
    err_corrected = base_err.copy()
    d0_mask = df["disturbance_id"] == "D0"
    for fp in df[d0_mask]["flow_point"].unique():
        mask = d0_mask & (df["flow_point"] == fp)
        err_corrected[mask] -= base_err[mask].mean()
    
    # 扰流: 连续补偿
    dist_mask = ~d0_mask
    dist_idx = np.where(dist_mask)[0]
    
    pab = df["profile_ab_abs"].astype(float).values[dist_idx]
    psw = df["profile_swirl"].astype(float).values[dist_idx]
    fp_vals = df["flow_point"].values[dist_idx]
    
    fp_dummies = np.zeros((len(dist_idx), 6))
    fp_unique = sorted(df["flow_point"].unique())
    for i, fp in enumerate(fp_vals):
        if fp in fp_unique and fp_unique.index(fp) > 0:
            fp_dummies[i, fp_unique.index(fp) - 1] = 1.0
    
    X = np.column_stack([pab, psw, pab * psw, pab**2, psw**2, fp_dummies])
    y = base_err[dist_idx]
    
    loo = LeaveOneOut()
    ridge = Ridge(alpha=10.0)
    y_pred_cv = cross_val_predict(ridge, X, y, cv=loo)
    
    err_corrected[dist_idx] -= y_pred_cv
    
    max_sd, passes, med_sd = compute_group_sd(err_corrected, df)
    print(f"\n分开建模 (D0流量点修正 + 扰流连续补偿):")
    print(f"  MAE: {np.abs(err_corrected).mean():.4f}%")
    print(f"  最大组SD: {max_sd:.4f}%, 中位数SD: {med_sd:.4f}%")
    print(f"  组通过: {passes}/30")
    
    return err_corrected


def test_more_features():
    """测试加入更多特征。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict, LeaveOneOut
    
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    # 特征集1: 基础 (profile_ab_abs, profile_swirl)
    pab = df["profile_ab_abs"].astype(float).values
    psw = df["profile_swirl"].astype(float).values
    
    # 特征集2: 加入动态特征
    dyn_cv = df["dyn_plateau_cv"].astype(float).values
    dyn_eq = df["dyn_active_eq_s"].astype(float).values
    
    # 特征集3: 加入零点特征
    zero_med = df["zero_rate_med"].astype(float).values
    zero_mad = df["zero_rate_mad"].astype(float).values
    zero_age = df["zero_age_s"].astype(float).values
    
    # 特征集4: 加入chord和ab
    chord_vals = df[CHORD_COLS].astype(float).values
    ab_vals = df[[f"ab{i}" for i in range(5)]].astype(float).values
    
    # 流量点哑变量
    fp_vals = df["flow_point"].values
    fp_dummies = np.zeros((len(df), 6))
    fp_unique = sorted(df["flow_point"].unique())
    for i, fp in enumerate(fp_vals):
        if fp in fp_unique and fp_unique.index(fp) > 0:
            fp_dummies[i, fp_unique.index(fp) - 1] = 1.0
    
    # 测试不同特征组合
    feature_sets = {
        "基础 (pab, psw)": np.column_stack([pab, psw, fp_dummies]),
        "+ 动态特征": np.column_stack([pab, psw, dyn_cv, dyn_eq, fp_dummies]),
        "+ 零点特征": np.column_stack([pab, psw, zero_med, zero_mad, zero_age, fp_dummies]),
        "+ chord/ab": np.column_stack([pab, psw, *chord_vals.T, *ab_vals.T, fp_dummies]),
        "全部特征": np.column_stack([pab, psw, dyn_cv, dyn_eq, zero_med, zero_mad, zero_age,
                                    *chord_vals.T, *ab_vals.T, fp_dummies]),
    }
    
    print("\n特征组合测试 (Ridge, alpha=10, LOO-CV):")
    loo = LeaveOneOut()
    for name, X in feature_sets.items():
        ridge = Ridge(alpha=10.0)
        y_pred_cv = cross_val_predict(ridge, X, base_err, cv=loo)
        err_c = base_err - y_pred_cv
        max_sd, passes, med_sd = compute_group_sd(err_c, df)
        print(f"  {name:20s}: MAE={np.abs(err_c).mean():.4f}%, "
              f"最大SD={max_sd:.4f}%, 通过={passes}/30")


def test_xgboost():
    """测试XGBoost模型。"""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("\nXGBoost未安装，跳过")
        return
    
    from sklearn.model_selection import cross_val_predict, KFold
    
    df = load_attachment1()
    base_err = compute_base_err(df)
    
    pab = df["profile_ab_abs"].astype(float).values
    psw = df["profile_swirl"].astype(float).values
    fp_vals = df["flow_point"].values
    
    fp_dummies = np.zeros((len(df), 6))
    fp_unique = sorted(df["flow_point"].unique())
    for i, fp in enumerate(fp_vals):
        if fp in fp_unique and fp_unique.index(fp) > 0:
            fp_dummies[i, fp_unique.index(fp) - 1] = 1.0
    
    X = np.column_stack([pab, psw, pab * psw, pab**2, psw**2, fp_dummies])
    y = base_err
    
    # XGBoost with strong regularization
    xgb = XGBRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        reg_alpha=1.0,
        reg_lambda=10.0,
        random_state=42,
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(xgb, X, y, cv=kf)
    
    err_c = y - y_pred_cv
    max_sd, passes, med_sd = compute_group_sd(err_c, df)
    
    print(f"\nXGBoost (5折CV):")
    print(f"  MAE: {np.abs(err_c).mean():.4f}%")
    print(f"  最大组SD: {max_sd:.4f}%, 中位数SD: {med_sd:.4f}%")
    print(f"  组通过: {passes}/30")


if __name__ == "__main__":
    print("=" * 80)
    print("改进方案测试")
    print("=" * 80)
    
    test_no_compensation()
    test_continuous_compensation()
    test_separate_d0_dist()
    test_more_features()
    test_xgboost()
