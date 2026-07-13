"""问题4 改进模型：连续特征补偿替代类别查表。

核心改进：
- 用 profile_ab_abs, profile_swirl 等连续特征的线性回归做逐窗口补偿
- 每个窗口获得不同的补偿值 → 降低组内SD
- D0 仍用流量点修正
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]


def baserate(df):
    return df[CHORD_COLS].astype(float).values @ W_PHYS6


def compute_base_err(df):
    """Phys6 基线相对误差 (%)。"""
    V_base = baserate(df) * AREA * df["duration_s"].astype(float).values
    return (V_base - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100


def fit_continuous_model(df):
    """拟合连续补偿模型。

    策略：对扰流窗口，用 profile_ab_abs, profile_swirl 及交互项
    做线性回归预测 Phys6 误差，然后补偿。
    对 D0，用流量点修正。
    """
    d0 = df[df["disturbance_id"] == "D0"].copy()
    dist = df[df["disturbance_id"] != "D0"].copy()

    # D0 流量点修正
    d0_err = compute_base_err(d0)
    d0_correction = {}
    for fp in sorted(d0["flow_point"].unique()):
        mask = d0["flow_point"].values == fp
        d0_correction[fp] = -float(d0_err[mask].mean())

    # 扰流窗口: 构建特征矩阵
    # 关键特征: profile_ab_abs, profile_swirl, 以及它们的交互
    pab = dist["profile_ab_abs"].astype(float).values
    psw = dist["profile_swirl"].astype(float).values
    err = compute_base_err(dist)

    # 特征工程: 利用物理直觉
    # profile_ab_abs 反映 A/B 差异大小 → 与误差正相关
    # profile_swirl 反映旋流强度 → 与误差正相关
    # 加流量点哑变量 (不同流量点基线不同)
    fp_dummies = pd.get_dummies(dist["flow_point"], prefix="fp", drop_first=False).values.astype(float)

    # 基础特征
    X_base = np.column_stack([
        pab,                    # A/B差异幅度
        psw,                    # 旋流强度
        pab * psw,              # 交互
        pab**2,                 # 非线性
        psw**2,                 # 非线性
        fp_dummies,             # 流量点
    ])

    # 用 Ridge 回归 (alpha 控制正则化)
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)

    # 选择 alpha: 用 LOO-CV
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    best_alpha, best_score = 1.0, -1e10
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        scores = cross_val_score(ridge, X_scaled, err, cv=min(5, len(err)),
                                 scoring="neg_mean_squared_error")
        score = scores.mean()
        if score > best_score:
            best_alpha, best_score = alpha, score

    print(f"  最优 alpha: {best_alpha}")
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, err)

    # 训练集拟合效果
    pred = ridge.predict(X_scaled)
    resid = err - pred
    print(f"  训练集: err std={err.std(ddof=1):.4f}%, residual std={resid.std(ddof=1):.4f}%")
    print(f"  系数: {dict(zip(['pab','psw','pab*psw','pab^2','psw^2'], ridge.coef_[:5]))}")

    return {
        "d0_correction": d0_correction,
        "scaler": scaler,
        "ridge": ridge,
        "best_alpha": best_alpha,
    }


def predict_continuous(df, tau_ab, tau_sw, params):
    """连续补偿预测。"""
    V_base = baserate(df) * AREA * df["duration_s"].astype(float).values
    flag = (df["profile_ab_abs"].abs() > tau_ab) | (df["profile_swirl"].abs() > tau_sw)

    d0_corr = params["d0_correction"]
    scaler = params["scaler"]
    ridge = params["ridge"]

    V_final = V_base.copy()
    for i in range(len(df)):
        fp = int(df.iloc[i]["flow_point"])
        if flag.iloc[i]:
            # 扰流: 连续特征回归补偿
            pab = float(df.iloc[i]["profile_ab_abs"])
            psw = float(df.iloc[i]["profile_swirl"])
            fp_dum = np.zeros(7)
            fp_idx = [20, 30, 40, 50, 60, 70, 80].index(fp) if fp in [20,30,40,50,60,70,80] else -1
            if fp_idx >= 0:
                fp_dum[fp_idx] = 1.0

            X = np.array([[pab, psw, pab*psw, pab**2, psw**2, *fp_dum]])
            X_s = scaler.transform(X)
            delta_pct = float(ridge.predict(X_s)[0])
            delta = delta_pct / 100.0  # 转为比例
            V_final[i] *= (1 + delta)
        else:
            # D0: 流量点修正
            delta = d0_corr.get(fp, 0) / 100.0
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
    params = fit_continuous_model(df)

    pred = predict_continuous(df, tau_ab, tau_sw, params)
    err = (pred - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    mae = err.abs().mean()

    submission = df[["window_id"]].copy()
    submission["model_volume_m3"] = pred
    out_path = RESULTS_DIR / "problem4_v5_continuous.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n连续补偿模型 MAE: {mae:.4f}%")
    print(f"提交文件: {out_path}")

    # 快速评价
    from evaluate_submission import evaluate
    evaluate(out_path, RESULTS_DIR / "eval_v5")


if __name__ == "__main__":
    main()
