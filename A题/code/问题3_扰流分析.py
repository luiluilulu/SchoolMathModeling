"""问题3子问题(1)(2)：扰流敏感特征识别 + 扰流状态聚类。

求解方法：Cohen's d 效应量评估特征判别力，Ward 层次聚类 + 轮廓系数选 K。
输出：output/figures/problem3_feature_discrimination.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from utils import load_attachment1, FIGURES_DIR, ensure_dirs

DIST_LIST = [f"D{i}" for i in range(1, 9)]
FEATURES = [
    "profile_swirl", "profile_ab_abs", "profile_edge_inner", "profile_center_all",
    "profile_top_bottom", "profile_inner_skew",
    "ab0", "ab1", "ab2", "ab3", "ab4",
    "zero_rate_med", "zero_rate_mad", "zero_age_s",
    "dyn_plateau_cv", "dyn_active_eq_s",
]


def cohens_d(v0, v1):
    """两组数据的 Cohen's d 效应量。"""
    return abs(v1.mean() - v0.mean()) / np.sqrt((v1.var() + v0.var()) / 2)


def feature_discrimination(df):
    """子问题(1)：各特征对 D0 vs 扰流的判别力。"""
    d0 = df[df["disturbance_id"] == "D0"]
    d1 = df[df["disturbance_id"] != "D0"]
    rows = []
    for f in FEATURES:
        d = cohens_d(d0[f].astype(float), d1[f].astype(float))
        rows.append({"feature": f, "cohens_d": d})
    return pd.DataFrame(rows).sort_values("cohens_d", ascending=False)


def cluster_disturbances(df):
    """子问题(2)：D1–D8 Ward 层次聚类 (10D特征→PCA→聚类)。"""
    chord_cols = ["chord0", "chord1", "chord2", "chord3", "chord4"]
    ab_cols = ["ab0", "ab1", "ab2", "ab3", "ab4"]
    d0 = df[df["disturbance_id"] == "D0"]

    # 逐窗口归一化chord + ab + swirl + ab_abs
    def extract_feats(sub):
        cs = sub[chord_cols].sum(axis=1).values
        norm_c = sub[chord_cols].values / (cs[:, None] + 1e-10)
        return np.hstack([norm_c, sub[ab_cols].values,
                          sub["profile_swirl"].values.reshape(-1, 1),
                          sub["profile_ab_abs"].values.reshape(-1, 1)])

    d0_feats = extract_feats(d0)
    feat_mean = d0_feats.mean(axis=0)
    feat_std = d0_feats.std(axis=0) + 1e-10

    data, labels_list = [], []
    for d in DIST_LIST:
        sub = df[df["disturbance_id"] == d]
        vec = (extract_feats(sub).mean(axis=0) - feat_mean) / feat_std
        data.append(vec)
        labels_list.append(d)

    X = np.array(data)
    pca = PCA(0.90).fit(X)
    X_pca = pca.transform(X)
    print(f"  PCA: {X.shape[1]}维 → {X_pca.shape[1]}维, 解释方差={pca.explained_variance_ratio_.sum():.1%}")

    Z = linkage(X_pca, method="ward")
    best_k, best_s = 2, -1
    for k in range(2, min(6, len(DIST_LIST) + 1)):
        cl = fcluster(Z, k, criterion="maxclust")
        s = silhouette_score(X_pca, cl)
        groups = {}
        for i, lab in enumerate(labels_list):
            groups.setdefault(cl[i], []).append(lab)
        print(f"  K={k}: 轮廓系数={s:.3f}  {list(groups.values())}")
        if s > best_s:
            best_k, best_s = k, s
    return best_k, best_s, X_pca, labels_list, Z


def plot_discrimination(stats):
    """特征判别力水平柱状图。"""
    top = stats.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d7191c" if v > 2 else "#fdae61" if v > 0.8 else "#abd9e9"
              for v in top["cohens_d"].values]
    ax.barh(range(len(top)), top["cohens_d"].values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.set_xlabel("Cohen's d")
    ax.set_title("特征对 D0 vs 扰流的判别力")
    for i, v in enumerate(top["cohens_d"].values):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "problem3_feature_discrimination.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()
    df = load_attachment1()

    # 子问题(1)
    stats = feature_discrimination(df)
    print("=== 子问题(1): 特征判别力 (Cohen's d) ===")
    for _, r in stats.iterrows():
        tag = "★★★" if r["cohens_d"] > 2 else ("★★" if r["cohens_d"] > 0.8 else "")
        print(f"  {r['feature']:25s}  d={r['cohens_d']:6.1f}  {tag}")

    # 子问题(2)
    print("\n=== 子问题(2): D1-D8 聚类 ===")
    best_k, best_s, X, labels, Z = cluster_disturbances(df)
    print(f"\n最优 K={best_k} (轮廓系数={best_s:.3f})")

    plot_discrimination(stats)


if __name__ == "__main__":
    main()
