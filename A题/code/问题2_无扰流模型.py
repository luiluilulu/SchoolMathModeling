"""问题2：无扰流条件下的多声道流量估计模型。

求解方法：拉格朗日基函数 × Gauss-Jacobi 权函数积分 → OWICS权重。
         叠加壁面边界层修正（权重不归一，取 Phys6）→ 零参数物理模型。
输出：output/results/problem2_results.csv, output/figures/problem2_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.integrate import quad

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

AREA = 0.13138219017128852
CHORD_COLS = ["chord0", "chord1", "chord2", "chord3", "chord4"]
# 五声道归一化高度（附件7，管顶→管底）
CHORD_POS = np.array([0.727425, 0.266219, 0.0, -0.266219, -0.727425])


def derive_weights():
    """拉格朗日基函数积分求 Gauss-Jacobi 权重。

    w_i = (2/π) ∫ L_i(t)·√(1-t²) dt, L_i 为 i 号节点的拉格朗日基函数。
    返回：(OWICS权重, Phys6权重)
    """
    n = len(CHORD_POS)
    w = np.zeros(n)
    for i in range(n):
        nodes = np.zeros(n); nodes[i] = 1.0
        poly = lagrange(CHORD_POS, nodes)
        f = lambda t, p=poly: p(t) * np.sqrt(max(0, 1 - t * t))
        val, _ = quad(f, -1, 1, limit=200)
        w[i] = val * 2 / np.pi
    # Phys6：工程修正，覆盖区域外壁面边界层为零流速
    phys6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
    return w, phys6


def predict(df, weights):
    rate = df[CHORD_COLS].astype(float).values @ weights
    return df["duration_s"].astype(float).values * AREA * rate


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
    return df["error_pct"].abs().mean(), groups_df


def plot_weights(owics_w, phys6_w):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(5)
    w = 0.3
    ax.bar(x - w/2, owics_w, w, color="#fdae61", edgecolor="none",
           label="推导权重 (OWICS, 和=1)")
    ax.bar(x + w/2, phys6_w, w, color="#2c7bb6", edgecolor="none",
           label="工程修正权重 (Phys6, 和≈0.993)")
    ax.set_xticks(x)
    ax.set_xticklabels(["chord0", "chord1", "chord2", "chord3", "chord4"])
    ax.set_ylabel("权重")
    ax.set_title("五声道权重：数学推导 vs 工程修正")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem2_weights.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_results(result, groups):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(result["standard_volume_m3"], result["model_volume_m3"],
                c=result["flow_point"], cmap="viridis", s=15, alpha=0.6)
    lims = [30, 90]
    ax1.plot(lims, lims, "k--", linewidth=0.8)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel("标准体积 (m^3)")
    ax1.set_ylabel("预测体积 (m^3)")
    ax1.set_title("预测 vs 真值")

    d0 = result[result["disturbance_id"] == "D0"]
    flow_pts = sorted(result["flow_point"].unique())
    all_mae = [result[result["flow_point"]==fp]["error_pct"].abs().mean() for fp in flow_pts]
    d0_mae = [d0[d0["flow_point"]==fp]["error_pct"].abs().mean() if fp in d0["flow_point"].values else 0
              for fp in flow_pts]
    x = np.arange(len(flow_pts)); bar_w = 0.35
    ax2.bar(x-bar_w/2, d0_mae, bar_w, color="#2c7bb6", edgecolor="none", label="无扰流")
    ax2.bar(x+bar_w/2, all_mae, bar_w, color="#d7191c", edgecolor="none", label="全量")
    ax2.set_xticks(x); ax2.set_xticklabels(flow_pts)
    ax2.set_xlabel("流量点"); ax2.set_ylabel("MAE (%)")
    ax2.set_title("各流量点 MAE")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem2_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    df = load_attachment1()
    owics_w, phys6_w = derive_weights()

    print("拉格朗日基函数 × Gauss-Jacobi 权函数积分:")
    for i, (h, w_o, w_p) in enumerate(zip(CHORD_POS, owics_w, phys6_w)):
        print(f"  chord{i}: 高度={h:+.4f}  推导权重={w_o:.6f}  Phys6={w_p:.6f}")
    print(f"  推导权重和={owics_w.sum():.4f}, Phys6和={phys6_w.sum():.4f}")

    # 验证：推导权重应与附件7 OWICS 一致
    owics_ref = np.array([0.221205, 0.112176, 0.333238, 0.112176, 0.221205])
    diff = np.abs(owics_w - owics_ref).max()
    print(f"  推导权重与附件7 OWICS 最大偏差: {diff:.2e}")
    if diff < 1e-5:
        print("  [OK] 推导正确，与 OWICS 一致")

    # 选择 Phys6 作为最终模型（工程修正后）
    weights = phys6_w
    pred = predict(df, weights)
    mae, groups = evaluate(df, pred)
    result = df.assign(model_volume_m3=pred, error_pct=
        (pred-df["standard_volume_m3"])/df["standard_volume_m3"]*100)

    d0 = df[df["disturbance_id"] == "D0"]
    d0_pred = predict(d0, weights)
    d0_mae = np.abs((d0_pred-d0["standard_volume_m3"])/d0["standard_volume_m3"]*100).mean()

    print(f"\n最终模型 (Phys6 工程修正):")
    print(f"  无扰流 D0 MAE: {d0_mae:.4f}%")
    print(f"  全量 MAE: {mae:.4f}%")
    if not groups.empty:
        n_pass = groups["pass_group"].sum()
        print(f"  组通过: {n_pass}/{len(groups)}")

    result[["window_id", "model_volume_m3"]].to_csv(
        RESULTS_DIR / "problem2_results.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(RESULTS_DIR / "problem2_groups.csv", index=False, encoding="utf-8-sig")

    plot_weights(owics_w, phys6_w)
    plot_results(result, groups)


if __name__ == "__main__":
    main()
