"""问题2：无扰流条件下的多声道流量估计模型。

求解方法：volume = duration_s × area × Σ(wᵢ·chordᵢ)，Ridge 拟合声道权重 + 留一日期CV。
输出：output/results/problem2_results.csv, output/figures/problem2_weights.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

AREA = 0.13138219017128852  # 等效截面积 m²
CHORD_COLS = ["chord0", "chord1", "chord2", "chord3", "chord4"]


def build_xy(df):
    """volume_rate = Σ wᵢ·chordᵢ + ε, 归一化到流速量纲。"""
    rate = df["standard_volume_m3"].astype(float) / df["duration_s"].astype(float)
    y = rate.values / AREA  # 面平均流速 ≈ m/s
    X = df[CHORD_COLS].astype(float).values
    return X, y


def cv_weights(df, alphas=(0.01, 0.1, 1, 10, 100)):
    """留一日期 CV 选 α，返回最优权重。"""
    logo = LeaveOneGroupOut()
    dates = df["date"].values
    X, y = build_xy(df)
    best_alpha, best_mae = alphas[0], float("inf")
    best_w, best_intercept = np.zeros(5), 0.0

    for alpha in alphas:
        pred = np.zeros(len(y))
        for train_idx, test_idx in logo.split(X, y, groups=dates):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X[train_idx], y[train_idx])
            pred[test_idx] = model.predict(X[test_idx])
        mae = np.abs((pred * AREA * df["duration_s"].astype(float).values
                       - df["standard_volume_m3"].astype(float).values)
                      / df["standard_volume_m3"].astype(float).values).mean() * 100
        print(f"  α={alpha:.2f}, CV MAE={mae:.4f}%")
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha
            # 全量拟合一次取权重
            final = Ridge(alpha=alpha, fit_intercept=True)
            final.fit(X, y)
            best_w = final.coef_
            best_intercept = final.intercept_

    print(f"最优 α={best_alpha:.2f}, 权重和={best_w.sum():.4f}, 截距={best_intercept:.4f}")
    return best_w, best_intercept, best_mae


def predict(df, weights, intercept):
    rate = df[CHORD_COLS].astype(float).values @ weights + intercept
    rate *= AREA
    return df["duration_s"].astype(float).values * rate


def evaluate(df, pred):
    df = df.copy()
    df["model_volume_m3"] = pred
    df["error_pct"] = (
        (df["model_volume_m3"] - df["standard_volume_m3"])
        / df["standard_volume_m3"] * 100
    )
    groups = []
    for (date, fp), grp in df.groupby(["date", "flow_point"]):
        if len(grp) < 3:
            continue
        e = grp["error_pct"]
        groups.append({
            "date": date, "flow_point": fp, "n": len(grp),
            "mean_error_pct": e.mean(), "sd_pct": e.std(ddof=1),
        })
    groups_df = pd.DataFrame(groups)
    if not groups_df.empty:
        groups_df["pass_group"] = (
            groups_df["mean_error_pct"].abs() <= 0.2) & (groups_df["sd_pct"] <= 0.040
        )
    return df["error_pct"].abs().mean(), groups_df, df


def plot_weights(fitted_w):
    phys6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
    lagrange = np.array([0.266260, 0.096121, 0.275238, 0.096121, 0.266260])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    w_bar = 0.25
    ax.bar(x - w_bar, phys6, w_bar, color="#2c7bb6", edgecolor="none", label="Phys6")
    ax.bar(x, lagrange, w_bar, color="#fdae61", edgecolor="none", label="Lagrange")
    ax.bar(x + w_bar, fitted_w, w_bar, color="#d7191c", edgecolor="none", label="Ridge拟合")
    ax.set_xticks(x)
    ax.set_xticklabels(["chord0", "chord1", "chord2", "chord3", "chord4"])
    ax.set_ylabel("权重")
    ax.set_title("五声道权重对比 (Ridge α 经 CV 选定)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem2_weights.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    df = load_attachment1()

    # 基线
    lagrange_mae = np.abs(
        (df["lagrange_volume_m3"] - df["standard_volume_m3"])
        / df["standard_volume_m3"] * 100
    ).mean()

    print(f"Lagrange 全量 MAE: {lagrange_mae:.4f}%")
    print("\n留一日期 CV 选 α:")
    w, intercept, cv_mae = cv_weights(df)

    print(f"\n拟合权重: {np.round(w, 4)}, 截距: {intercept:.4f}")
    print(f"CV MAE: {cv_mae:.4f}%")

    pred = predict(df, w, intercept)
    full_mae, groups, result = evaluate(df, pred)
    print(f"全量 MAE: {full_mae:.4f}%")
    if not groups.empty:
        n_pass = groups["pass_group"].sum()
        print(f"组通过: {n_pass}/{len(groups)}")
        # 分无扰流/扰流
        d0_groups = groups[groups["date"] == 20260607]
        if not d0_groups.empty:
            print(f"  无扰流(D0)组通过: {d0_groups['pass_group'].sum()}/{len(d0_groups)}")

    result[["window_id", "model_volume_m3"]].to_csv(
        RESULTS_DIR / "problem2_results.csv", index=False, encoding="utf-8-sig"
    )
    groups.to_csv(RESULTS_DIR / "problem2_groups.csv", index=False, encoding="utf-8-sig")

    plot_weights(w)


if __name__ == "__main__":
    main()
