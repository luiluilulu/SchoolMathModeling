"""问题3：扰流剖面识别与补偿。

求解方法：Cohen's d 特征筛选，Ward 聚类，PCA-Mahalanobis 在线分类，分流量点补偿。
输出：output/results/problem3_*.csv, output/figures/problem3_*.png。
"""

import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import f_oneway
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import FIGURES_DIR, RESULTS_DIR, ensure_dirs, load_attachment1

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

AREA_M2 = 0.13138219017128852
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
DIST_LIST = [f"D{i}" for i in range(1, 9)]
FEATURE_NAMES = [
    "norm_chord0", "norm_chord1", "norm_chord2", "norm_chord3", "norm_chord4",
    "ab0", "ab1", "ab2", "ab3", "ab4", "profile_swirl", "profile_ab_abs",
]
DISCRIMINATION_FEATURES = [
    "profile_swirl", "profile_ab_abs", "profile_edge_inner", "profile_center_all",
    "profile_top_bottom", "profile_inner_skew", "ab0", "ab1", "ab2", "ab3", "ab4",
    "zero_rate_med", "zero_rate_mad", "zero_age_s", "dyn_plateau_cv", "dyn_active_eq_s",
]


def calc_base_volume(df):
    """Phys6 基线体积，返回 ndarray."""
    velocity_m_s = df[CHORD_COLS].astype(float).values @ W_PHYS6
    return velocity_m_s * df["duration_s"].astype(float).values * AREA_M2


def relative_error_pct(model_volume, standard_volume):
    """相对误差，单位 pct."""
    return (model_volume - standard_volume.astype(float)) / standard_volume.astype(float) * 100


def build_feature_matrix(df):
    """在线特征矩阵，返回 ndarray."""
    chord_sum = df[CHORD_COLS].astype(float).sum(axis=1).values
    norm_chord = df[CHORD_COLS].astype(float).values / (chord_sum[:, None] + 1e-12)
    ab_values = df[AB_COLS].astype(float).values
    profile_values = df[["profile_swirl", "profile_ab_abs"]].astype(float).values
    return np.hstack([norm_chord, ab_values, profile_values])


def cohens_d(v0, v1):
    """两组数据的 Cohen's d."""
    denom = math.sqrt((v0.var(ddof=1) + v1.var(ddof=1)) / 2)
    return abs(v1.mean() - v0.mean()) / denom if denom > 0 else np.nan


def feature_discrimination(df):
    """D0 与扰流样本的特征判别力表."""
    d0 = df[df["disturbance_id"].eq("D0")]
    disturbed = df[~df["disturbance_id"].eq("D0")]
    rows = []
    for feature in DISCRIMINATION_FEATURES:
        v0 = d0[feature].astype(float)
        v1 = disturbed[feature].astype(float)
        rows.append({
            "feature": feature,
            "d0_mean": v0.mean(),
            "d0_sd": v0.std(ddof=1),
            "disturbed_mean": v1.mean(),
            "disturbed_sd": v1.std(ddof=1),
            "d0_min": v0.min(),
            "d0_max": v0.max(),
            "disturbed_min": v1.min(),
            "disturbed_max": v1.max(),
            "cohens_d": cohens_d(v0, v1),
            "range_separated": bool(v0.max() < v1.min() or v1.max() < v0.min()),
        })
    return pd.DataFrame(rows).sort_values("cohens_d", ascending=False)


def detection_thresholds(df):
    """D0 最大绝对值加 3 倍标准差阈值."""
    d0 = df[df["disturbance_id"].eq("D0")]
    ab_abs = d0["profile_ab_abs"].abs()
    swirl = d0["profile_swirl"].abs()
    return {
        "profile_ab_abs": float(ab_abs.max() + 3 * ab_abs.std(ddof=1)),
        "profile_swirl": float(swirl.max() + 3 * swirl.std(ddof=1)),
    }


def detect_disturbance(df, thresholds):
    """双阈值 OR 扰流检测."""
    return (
        df["profile_ab_abs"].abs().gt(thresholds["profile_ab_abs"])
        | df["profile_swirl"].abs().gt(thresholds["profile_swirl"])
    )


def fit_cluster_model(df):
    """按扰流编号均值聚类，返回 PCA 和聚类信息."""
    d0 = df[df["disturbance_id"].eq("D0")]
    d0_feat = build_feature_matrix(d0)
    feat_mean = d0_feat.mean(axis=0)
    feat_std = d0_feat.std(axis=0, ddof=0) + 1e-12
    dist_means = np.vstack([
        (build_feature_matrix(df[df["disturbance_id"].eq(dist)]).mean(axis=0) - feat_mean) / feat_std
        for dist in DIST_LIST
    ])
    pca = PCA(n_components=0.90).fit(dist_means)
    pc_means = pca.transform(dist_means)
    tree = linkage(pc_means, method="ward")
    raw_labels = fcluster(tree, 2, criterion="maxclust")
    class_map = {dist: ("A" if label == raw_labels[0] else "B")
                 for dist, label in zip(DIST_LIST, raw_labels)}
    return feat_mean, feat_std, pca, pc_means, tree, raw_labels, class_map


def silhouette_table(pc_means, tree):
    """不同 K 的轮廓系数."""
    rows = []
    for k in range(2, min(6, len(DIST_LIST) + 1)):
        labels = fcluster(tree, k, criterion="maxclust")
        groups = {}
        for dist, label in zip(DIST_LIST, labels):
            groups.setdefault(int(label), []).append(dist)
        rows.append({
            "k": k,
            "silhouette": silhouette_score(pc_means, labels),
            "groups": "; ".join("/".join(v) for v in groups.values()),
        })
    return pd.DataFrame(rows)


def fit_online_classifier(df, feat_mean, feat_std, pca, class_map):
    """用离线类标签拟合在线 Mahalanobis 分类器."""
    disturbed = df[~df["disturbance_id"].eq("D0")].copy()
    features = (build_feature_matrix(disturbed) - feat_mean) / feat_std
    pc_values = pca.transform(features)
    disturbed["true_class"] = disturbed["disturbance_id"].map(class_map)
    centers, covs = {}, {}
    for class_name, group in disturbed.groupby("true_class", sort=True):
        values = pc_values[disturbed["true_class"].eq(class_name).values]
        centers[class_name] = values.mean(axis=0)
        covs[class_name] = LedoitWolf().fit(values).covariance_
    return centers, covs


def mahalanobis_distance(row, center, cov):
    """Mahalanobis 距离."""
    diff = row - center
    return float(np.sqrt(diff @ np.linalg.pinv(cov) @ diff.T))


def classify_rows(df, disturbance_flag, feat_mean, feat_std, pca, centers, covs):
    """在线分类，D0 输出 D0，扰流输出 A/B."""
    pc_values = pca.transform((build_feature_matrix(df) - feat_mean) / feat_std)
    class_rows = []
    for idx, row in enumerate(pc_values):
        distances = {
            name: mahalanobis_distance(row, centers[name], covs[name])
            for name in sorted(centers)
        }
        predicted = min(distances, key=distances.get) if disturbance_flag.iloc[idx] else "D0"
        class_rows.append({
            "online_class": predicted,
            "mahalanobis_A": distances.get("A", np.nan),
            "mahalanobis_B": distances.get("B", np.nan),
            "pc1": row[0],
            "pc2": row[1] if len(row) > 1 else 0.0,
        })
    return pd.DataFrame(class_rows, index=df.index)


def d0_target_errors(df):
    """D0 各流量点基线误差；缺失点取最近流量点."""
    d0 = df[df["disturbance_id"].eq("D0")]
    target = d0.groupby("flow_point")["base_error_pct"].mean().to_dict()
    for flow_point in sorted(df["flow_point"].unique()):
        if flow_point not in target:
            nearest = min(target, key=lambda value: abs(value - flow_point))
            target[flow_point] = target[nearest]
    return target


def anova_by_class(df):
    """扰流误差随流量点变化的 ANOVA."""
    rows = []
    for class_name, group in df[df["disturbance_flag"]].groupby("online_class"):
        samples = [sub["base_error_pct"].values for _, sub in group.groupby("flow_point") if len(sub) >= 2]
        stat, p_value = f_oneway(*samples) if len(samples) >= 2 else (np.nan, np.nan)
        rows.append({"online_class": class_name, "anova_f": stat, "anova_p": p_value})
    return pd.DataFrame(rows)


def compensation_params(df, target_errors):
    """按在线类和流量点估计补偿参数."""
    disturbed = df[df["disturbance_flag"]].copy()
    target_pct = disturbed["flow_point"].map(target_errors).astype(float)
    disturbed["target_delta"] = (
        (1 + target_pct / 100) * disturbed["standard_volume_m3"] / disturbed["base_volume_m3"] - 1
    )
    disturbed["zero_delta"] = disturbed["standard_volume_m3"] / disturbed["base_volume_m3"] - 1
    params = (
        disturbed.groupby(["online_class", "flow_point"])
        .agg(
            n=("window_id", "count"),
            base_error_mean_pct=("base_error_pct", "mean"),
            base_error_sd_pct=("base_error_pct", "std"),
            disturbance_delta_to_d0=("target_delta", "mean"),
            total_delta_to_zero=("zero_delta", "mean"),
        )
        .reset_index()
    )
    params["d0_target_error_pct"] = params["flow_point"].map(target_errors).astype(float)
    return params.merge(anova_by_class(df), on="online_class", how="left")


def apply_compensation(df, params, delta_col):
    """扰流补偿预测体积."""
    lookup = params.set_index(["online_class", "flow_point"])[delta_col]
    out = df["base_volume_m3"].copy()
    for idx, row in df[df["disturbance_flag"]].iterrows():
        delta = lookup.loc[(row["online_class"], row["flow_point"])]
        out.loc[idx] = row["base_volume_m3"] * (1 + delta)
    return out


def group_summary(df, error_col):
    """同日期同流量点分组指标."""
    rows = []
    for (date, flow_point), group in df.groupby(["date", "flow_point"], sort=True):
        if len(group) < 3:
            continue
        errors = group[error_col].astype(float)
        rows.append({
            "date": date,
            "flow_point": flow_point,
            "n": len(group),
            "mean_error_pct": errors.mean(),
            "sd_pct": errors.std(ddof=1),
            "pass_group": abs(errors.mean()) <= 0.2 and errors.std(ddof=1) <= 0.040,
        })
    return pd.DataFrame(rows)


def disturbance_u_d(df, error_col):
    """扰流综合指标."""
    use = df[df["flow_point"].between(40, 100)]
    base = use[use["condition_note"].eq("no_disturbance_reference")]
    base_mean = base.groupby("flow_point")[error_col].mean()
    disturbed = use[use["condition_note"].eq("disturbed_test")]
    detail = disturbed.groupby(["disturbance_id", "flow_point"])[error_col].agg(["mean", "std"]).reset_index()
    detail["base_mean"] = detail["flow_point"].map(base_mean)
    detail["abs_drift"] = (detail["base_mean"] - detail["mean"]).abs()
    u_d_c = detail["abs_drift"].max() / math.sqrt(3)
    u_d_r = detail["std"].fillna(0).max()
    return math.sqrt(u_d_c * u_d_c + u_d_r * u_d_r)


def evaluation_summary(df):
    """补偿前后评价汇总."""
    rows = []
    models = [
        ("base_phys6", "base_error_pct"),
        ("comp_to_d0", "model_error_pct"),
        ("comp_to_zero", "zero_bias_error_pct"),
    ]
    for label, error_col in models:
        groups = group_summary(df, error_col)
        rows.append({
            "model": label,
            "mae_pct": df[error_col].abs().mean(),
            "disturbed_mae_pct": df[df["disturbance_flag"]][error_col].abs().mean(),
            "mean_error_pct": df[error_col].mean(),
            "group_pass": f"{int(groups['pass_group'].sum())}/{len(groups)}",
            "max_group_sd_pct": groups["sd_pct"].max(),
            "u_nor_d_pct": disturbance_u_d(df, error_col),
        })
    return pd.DataFrame(rows)


def save_params_json(thresholds, pca, class_map, centers, covs):
    """保存在线模型参数."""
    payload = {
        "thresholds": thresholds,
        "feature_names": FEATURE_NAMES,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "disturbance_class_map": class_map,
        "mahalanobis_centers": {key: val.tolist() for key, val in centers.items()},
        "mahalanobis_covariances": {key: val.tolist() for key, val in covs.items()},
    }
    with open(RESULTS_DIR / "problem3_params.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def plot_discrimination(stats):
    """特征判别力图."""
    top = stats.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#b2182b" if value > 2 else "#ef8a62" for value in top["cohens_d"]]
    ax.barh(top["feature"], top["cohens_d"], color=colors, edgecolor="none")
    ax.set_xlabel("Cohen's d")
    ax.set_title("D0 与扰流状态的特征判别力")
    for i, value in enumerate(top["cohens_d"]):
        ax.text(value + 0.4, i, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_feature_discrimination.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution(df, thresholds):
    """双阈值特征分布图."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = df["disturbance_id"].eq("D0").map({True: "#2166ac", False: "#b2182b"})
    ax.scatter(df["profile_ab_abs"], df["profile_swirl"], c=colors, s=30, alpha=0.8)
    ax.axvline(thresholds["profile_ab_abs"], color="#4d4d4d", linestyle="--", linewidth=1)
    ax.axhline(thresholds["profile_swirl"], color="#4d4d4d", linestyle="--", linewidth=1)
    ax.set_xlabel("profile_ab_abs")
    ax.set_ylabel("profile_swirl")
    ax.set_title("扰流检测双特征分布")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_feature_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(cluster_df):
    """扰流编号 PCA 分布图."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for class_name, group in cluster_df.groupby("online_class"):
        ax.scatter(group["pc1"], group["pc2"], s=70, label=f"类{class_name}")
        for _, row in group.iterrows():
            ax.text(row["pc1"] + 0.08, row["pc2"] + 0.08, row["disturbance_id"], fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("D1-D8 扰流状态 PCA 聚类")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(tree):
    """Ward 聚类谱系图."""
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(tree, labels=DIST_LIST, ax=ax, color_threshold=None)
    ax.set_ylabel("Ward distance")
    ax.set_title("D1-D8 层次聚类谱系")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compensation_heatmap(params):
    """补偿参数热力图."""
    pivot = params.pivot(index="online_class", columns="flow_point", values="disturbance_delta_to_d0") * 100
    fig, ax = plt.subplots(figsize=(8, 3.5))
    image = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"类{v}" for v in pivot.index])
    ax.set_xlabel("流量点")
    ax.set_title("扰流补偿系数：对齐 D0 基线误差 (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_compensation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(df, stats, cluster_df, silhouette_df, params, summary):
    """写出问题3结果表."""
    stats.to_csv(RESULTS_DIR / "problem3_feature_discrimination.csv", index=False, encoding="utf-8-sig")
    cluster_df.to_csv(RESULTS_DIR / "problem3_cluster_assignment.csv", index=False, encoding="utf-8-sig")
    silhouette_df.to_csv(RESULTS_DIR / "problem3_silhouette.csv", index=False, encoding="utf-8-sig")
    params.to_csv(RESULTS_DIR / "problem3_compensation_params.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / "problem3_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    df[[
        "window_id", "disturbance_id", "flow_point", "disturbance_flag",
        "true_is_disturbed", "online_class", "true_class", "class_correct",
        "mahalanobis_A", "mahalanobis_B",
    ]].to_csv(RESULTS_DIR / "problem3_classification.csv", index=False, encoding="utf-8-sig")
    df[[
        "window_id", "profile_ab_abs", "profile_swirl", "disturbance_flag",
        "true_is_disturbed", "detection_correct",
    ]].to_csv(RESULTS_DIR / "problem3_detection_flags.csv", index=False, encoding="utf-8-sig")
    df[[
        "window_id", "base_volume_m3", "model_volume_m3",
        "zero_bias_volume_m3", "base_error_pct", "model_error_pct",
        "zero_bias_error_pct", "online_class", "flow_point",
    ]].to_csv(RESULTS_DIR / "problem3_compensated_results.csv", index=False, encoding="utf-8-sig")
    df[["window_id", "model_volume_m3"]].to_csv(
        RESULTS_DIR / "problem3_d0_aligned_submission.csv", index=False, encoding="utf-8-sig"
    )
    df[["window_id", "zero_bias_volume_m3"]].rename(
        columns={"zero_bias_volume_m3": "model_volume_m3"}
    ).to_csv(RESULTS_DIR / "problem3_zero_bias_submission.csv", index=False, encoding="utf-8-sig")


def main():
    ensure_dirs()
    df = load_attachment1()
    df["base_volume_m3"] = calc_base_volume(df)
    df["base_error_pct"] = relative_error_pct(df["base_volume_m3"], df["standard_volume_m3"])

    stats = feature_discrimination(df)
    thresholds = detection_thresholds(df)
    df["disturbance_flag"] = detect_disturbance(df, thresholds)
    df["true_is_disturbed"] = ~df["disturbance_id"].eq("D0")
    df["detection_correct"] = df["disturbance_flag"].eq(df["true_is_disturbed"])

    feat_mean, feat_std, pca, pc_means, tree, raw_labels, class_map = fit_cluster_model(df)
    silhouette_df = silhouette_table(pc_means, tree)
    centers, covs = fit_online_classifier(df, feat_mean, feat_std, pca, class_map)
    class_df = classify_rows(df, df["disturbance_flag"], feat_mean, feat_std, pca, centers, covs)
    df = pd.concat([df, class_df], axis=1)
    df["true_class"] = df["disturbance_id"].map(class_map).fillna("D0")
    df["class_correct"] = df["online_class"].eq(df["true_class"])

    target_errors = d0_target_errors(df)
    params = compensation_params(df, target_errors)
    df["model_volume_m3"] = apply_compensation(df, params, "disturbance_delta_to_d0")
    df["zero_bias_volume_m3"] = apply_compensation(df, params, "total_delta_to_zero")
    df["model_error_pct"] = relative_error_pct(df["model_volume_m3"], df["standard_volume_m3"])
    df["zero_bias_error_pct"] = relative_error_pct(
        df["zero_bias_volume_m3"], df["standard_volume_m3"]
    )
    summary = evaluation_summary(df)

    cluster_df = pd.DataFrame({
        "disturbance_id": DIST_LIST,
        "raw_cluster": raw_labels,
        "online_class": [class_map[dist] for dist in DIST_LIST],
        "pc1": pc_means[:, 0],
        "pc2": pc_means[:, 1] if pc_means.shape[1] > 1 else np.zeros(len(DIST_LIST)),
        "n_windows": [int(df["disturbance_id"].eq(dist).sum()) for dist in DIST_LIST],
    })

    write_outputs(df, stats, cluster_df, silhouette_df, params, summary)
    save_params_json(thresholds, pca, class_map, centers, covs)
    plot_discrimination(stats)
    plot_feature_distribution(df, thresholds)
    plot_pca_scatter(cluster_df)
    plot_dendrogram(tree)
    plot_compensation_heatmap(params)

    print("=== 问题3：扰流识别与补偿 ===")
    print(f"检测阈值: profile_ab_abs={thresholds['profile_ab_abs']:.5f}, "
          f"profile_swirl={thresholds['profile_swirl']:.5f}")
    print(f"扰流检测正确: {int(df['detection_correct'].sum())}/{len(df)}")
    disturbed = df[df["true_is_disturbed"]]
    print(f"在线分类正确: {int(disturbed['class_correct'].sum())}/{len(disturbed)}")
    print("聚类分组:")
    for class_name, group in cluster_df.groupby("online_class"):
        print(f"  类{class_name}: {'/'.join(group['disturbance_id'])}")
    print("\n评价汇总:")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
