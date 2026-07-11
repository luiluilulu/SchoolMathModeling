"""问题2：无扰流条件下的多声道流量估计模型。

求解方法：Phys6物理权重——0参数模型，权重来源于声道几何位置和Gauss-Jacobi积分。
验证：对比无扰流(10窗)与全量(159窗)表现，分析扰流导致的退化。
输出：output/results/problem2_results.csv, output/figures/problem2_flow_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils import load_attachment1, RESULTS_DIR, FIGURES_DIR, ensure_dirs

AREA = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = ["chord0", "chord1", "chord2", "chord3", "chord4"]


def predict(df):
    """volume = duration_s × area × Σ(wᵢ·chordᵢ)。"""
    rate = df[CHORD_COLS].astype(float).values @ W_PHYS6 * AREA
    return df["duration_s"].astype(float).values * rate


def eval_subset(df, label):
    """评估一个子集，返回 MAE 和分流量点统计。"""
    pred = predict(df)
    err = (pred - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    mae = err.abs().mean()
    by_flow = err.groupby(df["flow_point"]).agg(["mean", "count"])
    return mae, err.to_numpy(), by_flow


def evaluate_groups(df):
    """按日期-流量点分组统计。"""
    pred = predict(df)
    err = (pred - df["standard_volume_m3"]) / df["standard_volume_m3"] * 100
    groups = []
    for (date, fp), grp in df.groupby(["date", "flow_point"]):
        if len(grp) < 3:
            continue
        e = err[grp.index]
        groups.append({
            "date": date, "flow_point": fp, "n": len(grp),
            "mean_error_pct": e.mean(), "sd_pct": e.std(ddof=1),
        })
    groups_df = pd.DataFrame(groups)
    if not groups_df.empty:
        groups_df["pass_group"] = (
            groups_df["mean_error_pct"].abs() <= 0.2) & (groups_df["sd_pct"] <= 0.040
        )
    return groups_df


def plot_flow_comparison(d0_by_flow, all_by_flow, groups):
    """流量点MAE对比：无扰流 vs 全量。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    flow_pts = sorted(set(d0_by_flow.index) | set(all_by_flow.index))
    d0_mae = [d0_by_flow.loc[fp, "mean"].abs() if fp in d0_by_flow.index else np.nan
              for fp in flow_pts]
    all_mae = [all_by_flow.loc[fp, "mean"].abs() if fp in all_by_flow.index else np.nan
               for fp in flow_pts]
    x = np.arange(len(flow_pts))
    w = 0.35
    ax1.bar(x - w/2, d0_mae, w, color="#2c7bb6", edgecolor="none", label="无扰流(10窗)")
    ax1.bar(x + w/2, all_mae, w, color="#d7191c", edgecolor="none", label="全量(159窗)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(flow_pts)
    ax1.set_xlabel("流量点")
    ax1.set_ylabel("绝对平均误差（%）")
    ax1.set_title("各流量点误差：无扰流 vs 全量")
    ax1.legend(fontsize=9)

    dates = sorted(groups["date"].unique())
    date_pass = [
        groups[groups["date"] == d]["pass_group"].sum() for d in dates
    ]
    date_total = [len(groups[groups["date"] == d]) for d in dates]
    ax2.barh(range(len(dates)), date_pass, color="#2c7bb6", edgecolor="none")
    ax2.set_yticks(range(len(dates)))
    ax2.set_yticklabels([str(d) for d in dates], fontsize=7)
    ax2.set_xlabel("通过组数")
    ax2.set_title(f"各日期组通过数 (共{len(groups)}组)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem2_flow_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    df = load_attachment1()
    d0 = df[df["disturbance_id"] == "D0"]

    # Phys6: 0参数模型
    d0_mae, d0_err, d0_by_flow = eval_subset(d0, "无扰流")
    all_mae, all_err, all_by_flow = eval_subset(df, "全量")
    groups = evaluate_groups(df)

    print(f"Phys6物理权重: {np.round(W_PHYS6, 4)}")
    print(f"权重和: {W_PHYS6.sum():.4f}")
    print(f"无扰流(10窗) MAE: {d0_mae:.4f}%")
    print(f"全量(159窗) MAE: {all_mae:.4f}%")
    if not groups.empty:
        n_pass = groups["pass_group"].sum()
        print(f"组通过: {n_pass}/{len(groups)}")

    result = df[["window_id"]].copy()
    result["model_volume_m3"] = predict(df)
    result.to_csv(RESULTS_DIR / "problem2_results.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(RESULTS_DIR / "problem2_groups.csv", index=False, encoding="utf-8-sig")

    plot_flow_comparison(d0_by_flow, all_by_flow, groups)


if __name__ == "__main__":
    main()
