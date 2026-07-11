"""问题1：四套基础积分方案的误差对比。

求解方法：直接计算附件1中 phys6/owics/lagrange/equal_weight 四列相对误差。
输出：output/results/problem1_error_summary.csv, output/figures/problem1_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

METHODS = {
    "phys6": "phys6_volume_m3",
    "owics": "owics_volume_m3",
    "lagrange": "lagrange_volume_m3",
    "equal_weight": "equal_weight_volume_m3",
}


def calc_errors(df):
    """逐窗口相对误差 (pct)。"""
    out = df[["window_id", "date", "flow_point", "standard_volume_m3",
              "condition_note", "disturbance_id"]].copy()
    for name, col in METHODS.items():
        out[f"error_{name}_pct"] = (
            (df[col] - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
        )
    return out


def summary_table(errors):
    """全窗口 MAE / 均值误差 / 最大误差。"""
    rows = []
    for name in METHODS:
        e = errors[f"error_{name}_pct"]
        rows.append({
            "method": name,
            "mae_pct": e.abs().mean(),
            "mean_error_pct": e.mean(),
            "max_abs_error_pct": e.abs().max(),
        })
    return pd.DataFrame(rows)


def by_flow_table(errors):
    """按流量点分组。"""
    rows = []
    for flow, grp in errors.groupby("flow_point"):
        for name in METHODS:
            e = grp[f"error_{name}_pct"]
            rows.append({
                "flow_point": flow,
                "method": name,
                "n": len(grp),
                "mae_pct": e.abs().mean(),
                "mean_error_pct": e.mean(),
            })
    return pd.DataFrame(rows)


def plot_comparison(summary, by_flow):
    """全窗口平均绝对误差柱状 + 分流量点折线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    methods = list(METHODS.keys())
    labels = ["Phys6", "OWICS", "Lagrange", "等权"]
    colors = ["#2c7bb6", "#d7191c", "#fdae61", "#5e3c99"]

    mae_vals = [summary.loc[summary["method"] == m, "mae_pct"].values[0] for m in methods]
    bars = ax1.bar(labels, mae_vals, color=colors, edgecolor="none", width=0.6)
    ax1.set_ylabel("平均绝对误差（%）")
    ax1.set_title("全窗口平均绝对误差")
    ax1.set_ylim(0, max(mae_vals) * 1.15)
    for bar, val in zip(bars, mae_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}%", ha="center", va="bottom", fontsize=9)

    for name, color, label in zip(methods, colors, labels):
        sub = by_flow[by_flow["method"] == name]
        ax2.plot(sub["flow_point"], sub["mae_pct"], "o-", color=color, label=label, linewidth=1.8)
    ax2.set_xlabel("流量点")
    ax2.set_ylabel("平均绝对误差（%）")
    ax2.set_title("各流量点平均绝对误差")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem1_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()



CHORD_COLS = ["chord0", "chord1", "chord2", "chord3", "chord4"]
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
AREA = 0.13138219017128852


def verify_model_formula(df):
    """验证 volume = duration × area × Σ(w·chord) 与附件1 phys6_volume_m3 一致。"""
    rate = df[CHORD_COLS].astype(float).values @ W_PHYS6
    computed = df["duration_s"].astype(float).values * AREA * rate
    ref = df["phys6_volume_m3"].astype(float).values
    diff = np.abs(computed - ref)
    print(f"\n模型公式验证: vol = A × T × Σ(w·chord)")
    print(f"  与 phys6_volume_m3 最大差异: {diff.max():.2e} m³")
    if diff.max() < 1e-6:
        print("  [OK] 公式正确，与附件1一致")


def main():
    ensure_dirs()
    df = load_attachment1()
    verify_model_formula(df)
    errors = calc_errors(df)

    summary = summary_table(errors)
    by_flow = by_flow_table(errors)

    summary.to_csv(RESULTS_DIR / "problem1_error_summary.csv", index=False, encoding="utf-8-sig")
    by_flow.to_csv(RESULTS_DIR / "problem1_error_by_flow.csv", index=False, encoding="utf-8-sig")

    print(summary.round(4).to_string(index=False))
    print(f"\n总窗口数: {len(df)}")

    plot_comparison(summary, by_flow)


if __name__ == "__main__":
    main()
