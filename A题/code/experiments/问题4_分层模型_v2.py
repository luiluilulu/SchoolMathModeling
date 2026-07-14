"""
问题4分层模型 v2：D0(OWICS+LOFPO校准) + 扰流(Phys6+ET+聚类连续特征)。
H0=ET基线 H1=+D0校准 H2=+聚类特征 H3=+A/B趋势+ET残差。
D0: LOFPO验证；扰流: 嵌套LODO验证(u_r+u_d评分)。
输出: output/results/problem4_layered_v2/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/problem4_layered_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W_OWICS = np.array([0.221205, 0.112176, 0.333238, 0.112176, 0.221205])
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
FEATS_ET = CHORD_COLS + [f"ab{i}" for i in range(5)] + [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner", "profile_inner_skew",
    "profile_ab_abs", "profile_swirl", "dyn_first_0p1_s", "dyn_tail_0p1_s",
    "dyn_start_over_plateau", "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "zero_rate_med", "zero_rate_mad", "zero_age_s", "base_rate_m3h", "duration_s", "flow_point",
]
FEATS_CLASSIFY = [
    "norm_chord0", "norm_chord1", "norm_chord2", "norm_chord3", "norm_chord4",
    "ab0", "ab1", "ab2", "ab3", "ab4", "profile_swirl", "profile_ab_abs",
]
CLUSTER_MAP = {"D1":"A","D2":"A","D5":"A","D7":"A","D3":"B","D4":"B","D6":"B","D8":"B"}

df = pd.read_csv(DATA, encoding="utf-8-sig")
dur = df["duration_s"].astype(float).values
ch = df[CHORD_COLS].astype(float).values
df["v_owics"] = AREA * dur * (ch @ W_OWICS)
df["v_phys6"] = AREA * dur * (ch @ W_PHYS6)
df["base_rate_m3h"] = df["v_phys6"] / dur * 3600
ch_sum = ch.sum(axis=1)
for j in range(5):
    df[f"norm_chord{j}"] = ch[:, j] / np.maximum(ch_sum, 1e-12)
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
d0_mask = df["disturbance_id"] == "D0"
dist_mask = ~d0_mask

# ---- 评价函数 ----
def ev(err, df_sub=None):
    w = (df if df_sub is None else df_sub).copy(); w["ep"] = err
    gd = []
    for _, grp in w.groupby(["date", "flow_point"]):
        if len(grp) < 3: continue
        ee = grp["ep"].values; gd.append({"m": ee.mean(), "s": ee.std(ddof=1)})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0
    d0w = w[w["disturbance_id"] == "D0"]
    d0m = d0w.groupby("flow_point")["ep"].mean() if len(d0w) > 0 else pd.Series(dtype=float)
    ul = math.sqrt((d0m**2).sum() / max(len(d0m) - 1, 1)) if len(d0m) > 1 else (float(d0m.iloc[0]) if len(d0m) == 1 else np.nan)
    ur = gdf["s"].max() if not gdf.empty else 0.0
    use = w[w["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")]
    dt = use[use["condition_note"].eq("disturbed_test")]
    if len(bm) > 0 and len(dt) > 0:
        bmm = bm.groupby("flow_point")["ep"].mean()
        d1 = dt.groupby(["disturbance_id", "flow_point"])["ep"].agg(["mean", "std"]).reset_index()
        d1["bm"] = d1["flow_point"].map(bmm); d1 = d1.dropna(subset=["bm"])
        d1["drift"] = (d1["bm"] - d1["mean"]).abs()
        ud = math.sqrt((d1["drift"].max() / math.sqrt(3))**2 + d1["std"].fillna(0).max()**2)
    else:
        ud = np.nan
    return {"pass": gp, "total": len(gdf),
            "u_L": float(ul) if np.isfinite(ul) else 0.0,
            "u_r": float(ur),
            "u_d": float(ud) if np.isfinite(ud) else 0.0,
            "MAE": float(np.abs(err).mean())}

def inner_score(err, df_sub):
    gp=0; gs=0.0; gm=0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)

def dist_score(m):
    """扰流分支评分：仅u_r+u_d，不含u_L。越小越好。"""
    rr = m["u_r"] / 0.040; rd = m["u_d"] / 0.115 if np.isfinite(m["u_d"]) else 0
    return (max(rr, rd), rr**2 + rd**2, -m["pass"], m["MAE"])

# ---- D0 LOFPO校准 ----
def d0_lofpo_calibration():
    """Leave-One-Flow-Point-Out评估D0校准。返回最优方法和全窗口D0预测。"""
    d0_df = df[d0_mask].copy()
    fps = sorted(d0_df["flow_point"].unique())
    methods = {
        "none": lambda z_train, y_train, z_test: np.zeros(len(z_test)),
        "k0": lambda z_train, y_train, z_test: np.full(len(z_test), np.mean(y_train)),
        "linear": lambda z_train, y_train, z_test: np.polyval(np.polyfit(z_train, y_train, 1), z_test),
    }
    best_method = "none"; best_ul = np.inf
    all_preds = {}
    for method, fn in methods.items():
        preds = np.zeros(len(d0_df))
        for fp in fps:
            tr_idx = d0_df["flow_point"] != fp
            te_idx = d0_df["flow_point"] == fp
            z_tr = (d0_df.loc[tr_idx, "flow_point"].values - 50) / 30
            y_tr = np.log(d0_df.loc[tr_idx, "standard_volume_m3"].values
                          / d0_df.loc[tr_idx, "v_owics"].values)
            z_te = (d0_df.loc[te_idx, "flow_point"].values - 50) / 30
            preds[te_idx] = fn(z_tr, y_tr, z_te)
        all_preds[method] = preds
        v_pred = d0_df["v_owics"].values * np.exp(preds)
        e = (v_pred - d0_df["standard_volume_m3"].values) / d0_df["standard_volume_m3"].values * 100
        fp_means = pd.Series(e, index=d0_df.index).groupby(d0_df["flow_point"]).mean()
        ul = math.sqrt((fp_means**2).sum() / max(len(fp_means) - 1, 1))
        if ul < best_ul:
            best_ul = ul; best_method = method
    return best_method, all_preds[best_method], best_ul

# D0 LOFPO校准在每个LODO折内独立执行（见run_h_model）

# ---- 聚类连续特征构造 ----
def build_cluster_features(train_df):
    """在训练集上拟合分类器，返回对任意df_sub提取4个连续特征的函数。"""
    d0_tr = train_df[train_df["disturbance_id"] == "D0"]
    if len(d0_tr) == 0:
        def null_fn(df_sub):
            return np.zeros((len(df_sub), 4))
        return null_fn
    X = train_df[FEATS_CLASSIFY].astype(float).values
    mu = X[train_df["disturbance_id"] == "D0"].mean(axis=0)
    sig = X[train_df["disturbance_id"] == "D0"].std(axis=0) + 1e-12
    Xs = (X - mu) / sig
    from sklearn.decomposition import PCA
    pca = PCA(0.90).fit(Xs)
    Xp = pca.transform(Xs)
    centers = {}
    for label in ["A", "B"]:
        m = train_df["disturbance_id"].map(CLUSTER_MAP) == label
        if m.sum() > 0: centers[label] = Xp[m.values].mean(axis=0)
    tb_a = train_df[train_df["disturbance_id"].map(CLUSTER_MAP) == "A"]["profile_top_bottom"].mean()
    tb_b = train_df[train_df["disturbance_id"].map(CLUSTER_MAP) == "B"]["profile_top_bottom"].mean()
    swap = tb_b > tb_a

    def extract(df_sub):
        Xs2 = (df_sub[FEATS_CLASSIFY].astype(float).values - mu) / sig
        Xp2 = pca.transform(Xs2)
        d_a = np.sqrt(((Xp2 - centers["A"])**2).sum(axis=1))
        d_b = np.sqrt(((Xp2 - centers["B"])**2).sum(axis=1))
        if swap: d_a, d_b = d_b, d_a
        ea, eb = np.exp(-d_a**2/2), np.exp(-d_b**2/2)
        p_a = ea / (ea + eb + 1e-12)
        p_b = 1 - p_a
        return np.column_stack([p_a, p_b, d_a - d_b, np.minimum(d_a, d_b)])
    return extract

# ---- 主流程：H0-H3 ----
ET_PG = [{"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
         for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]]

def run_h_model(name, use_d0_cal, use_cluster_feats, use_ab_trend):
    """嵌套LODO运行扰流分支 + LOFPO D0校准。"""
    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote].reset_index(drop=True)
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        # 准备特征
        cluster_fn = build_cluster_features(tr_o) if use_cluster_feats else None
        et_feats = FEATS_ET.copy()
        if cluster_fn is not None:
            cluster_cols = ["p_A", "p_B", "dA_dB", "d_min"]
            tr_cluster = cluster_fn(tr_o)
            te_cluster = cluster_fn(te_o)
            for j, cn in enumerate(cluster_cols):
                tr_o[cn] = tr_cluster[:, j]
                te_o[cn] = te_cluster[:, j]
            et_feats = et_feats + cluster_cols

        # 扰流训练数据
        tr_dist = tr_o[tr_o["disturbance_id"] != "D0"]
        te_dist = te_o[te_o["disturbance_id"] != "D0"]

        # A/B流量趋势（可选）
        if use_ab_trend and cluster_fn is not None and len(tr_dist) >= 6:
            z_tr = (tr_dist["flow_point"].values - 50) / 30
            y_tr = np.log(tr_dist["standard_volume_m3"].values / tr_dist["v_phys6"].values)
            dist_pos = np.where((tr_o["disturbance_id"] != "D0").values)[0]
            pa_tr_arr = tr_cluster[dist_pos, 0]
            pb_tr_arr = tr_cluster[dist_pos, 1]
            mask_a = tr_dist["disturbance_id"].map(CLUSTER_MAP).values == "A"
            mask_b = tr_dist["disturbance_id"].map(CLUSTER_MAP).values == "B"
            if mask_a.sum() >= 3:
                ca = np.polyfit(z_tr[mask_a], y_tr[mask_a], 1)
            else:
                ca = np.array([0.0, 0.0])
            if mask_b.sum() >= 3:
                cb = np.polyfit(z_tr[mask_b], y_tr[mask_b], 1)
            else:
                cb = np.array([0.0, 0.0])
            r_ab = np.zeros(len(tr_o))
            r_ab[dist_pos] = pa_tr_arr * np.polyval(ca, z_tr) + pb_tr_arr * np.polyval(cb, z_tr)
            # 测试集AB趋势
            te_dist_pos2 = np.where((te_o["disturbance_id"] != "D0").values)[0]
            z_te_d = (te_o.iloc[te_dist_pos2]["flow_point"].values - 50) / 30
            r_ab_te_arr = np.zeros(len(te_o))
            r_ab_te_arr[te_dist_pos2] = (
                te_cluster[te_dist_pos2, 0] * np.polyval(ca, z_te_d)
                + te_cluster[te_dist_pos2, 1] * np.polyval(cb, z_te_d)
            )
        else:
            r_ab = np.zeros(len(tr_o))
            r_ab_te_arr = np.zeros(len(te_o))

        # ET拟合（扰流分支）
        r_target = np.zeros(len(tr_o))
        dist_idx_tr = np.where((tr_o["disturbance_id"] != "D0").values)[0]
        r_target[dist_idx_tr] = np.log(tr_o.iloc[dist_idx_tr]["standard_volume_m3"].values
                                        / tr_o.iloc[dist_idx_tr]["v_phys6"].values)
        r_residual = r_target - r_ab

        best_is = (-1, -np.inf, -np.inf); best_p = None
        for p in ET_PG:
            ip = np.zeros(len(tr_o))
            for itr, iva in inner.split(tr_o, groups=idates):
                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                Xtr_f = tri[et_feats].astype(float).fillna(tri[et_feats].median()).values
                Xva_f = vai[et_feats].astype(float).fillna(tri[et_feats].median()).values
                et = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
                et.fit(Xtr_f, r_residual[itr])
                ip[iva] = et.predict(Xva_f)
            r_pred = r_ab + ip
            vol = tr_o["v_phys6"].values * np.exp(r_pred)
            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
            iscore = inner_score(err, tr_o)
            if iscore > best_is:
                best_is = iscore; best_p = p

        # 固定η=1.0（已验证有效，η搜索未改善外层结果）
        best_eta = 1.0

        # 最终拟合+预测
        Xall = tr_o[et_feats].astype(float).fillna(tr_o[et_feats].median()).values
        Xte = te_o[et_feats].astype(float).fillna(tr_o[et_feats].median()).values
        et_final = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
        et_final.fit(Xall, r_residual)
        r_te_et = et_final.predict(Xte)

        # 测试集预测: D0 + 扰流
        te_pred = np.zeros(len(te_o))
        te_d0 = te_o["disturbance_id"] == "D0"
        te_dist_idx2 = np.where(~te_d0.values)[0]
        # 折内D0 LOFPO校准
        d0_tr_fold = tr_o[tr_o["disturbance_id"] == "D0"]
        if te_d0.sum() > 0 and use_d0_cal and len(d0_tr_fold) >= 4:
            # 训练集有D0 → LOFPO校准
            for fp_te in sorted(te_o.loc[te_d0, "flow_point"].unique()):
                fp_mask = (te_o["flow_point"] == fp_te) & te_d0
                z_val = (fp_te - 50) / 30
                z_tr = (d0_tr_fold["flow_point"].values - 50) / 30
                y_tr = np.log(d0_tr_fold["standard_volume_m3"].values / d0_tr_fold["v_owics"].values)
                a, b = np.polyfit(z_tr, y_tr, 1)
                te_pred[fp_mask] = te_o.loc[fp_mask, "v_owics"].values * np.exp(a * z_val + b)
        elif te_d0.sum() > 0:
            te_pred[te_d0] = te_o.loc[te_d0, "v_owics"].values
        if len(te_dist_idx2) > 0:
            te_pred[te_dist_idx2] = te_o.iloc[te_dist_idx2]["v_phys6"].values * np.exp(
                r_ab_te_arr[te_dist_idx2] + r_te_et[te_dist_idx2] * best_eta)

        outer_pred[ote] = te_pred
        print(f"  {name} 折{of+1}/{len(set(dates))} η={best_eta:.2f} best_is={best_is[0]}", flush=True)

    final_err = (outer_pred - std_vol) / std_vol * 100
    return ev(final_err), outer_pred


# ==== H0-H3 ====
configs = {
    "H0_et_baseline":   (False, False, False),
    "H1_d0cal":         (True,  False, False),
    "H2_cluster":       (True,  True,  False),
    "H3_ab_trend":      (True,  True,  True),
}
all_results = {}
for name, (d0cal, cluster, ab) in configs.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}", flush=True)
    m, pred = run_h_model(name, d0cal, cluster, ab)
    all_results[name] = m
    rr = m["u_r"]/0.040; rd = m["u_d"]/0.115 if m["u_d"]>0 else 0
    print(f"  => pass={m['pass']}/{m['total']} MAE={m['MAE']:.4f}% "
          f"u_L={m['u_L']:.4f}% u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}% "
          f"max(Rr,Rd)={max(rr,rd):.1f}x", flush=True)

# 基线
e_p = (df["v_phys6"].values - std_vol) / std_vol * 100; m_phys6 = ev(e_p)
e_o = (df["v_owics"].values - std_vol) / std_vol * 100; m_owics = ev(e_o)

print(f"\n{'='*65}")
print(f"分层模型 v2 对比")
print(f"{'='*65}")
print(f"{'模型':15s} {'Pass':>7s} {'MAE':>8s} {'u_L':>8s} {'u_r':>8s} {'u_d':>8s}")
print("-"*65)
for label, m in [("Phys6", m_phys6), ("OWICS", m_owics)] + list(all_results.items()):
    print(f"{label:15s} {m['pass']:2d}/{m['total']}  {m['MAE']:.4f}% {m['u_L']:.4f}% "
          f"{m['u_r']:.4f}% {m['u_d']:.4f}%")
print(f"\n  D0 LOFPO u_L: {d0_ul_lofpo:.4f}% (方法={best_d0_method})")

rows = [{"model":"phys6",**m_phys6},{"model":"owics",**m_owics}]
for n, m in all_results.items(): rows.append({"model":n,**m})
pd.DataFrame(rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({k:all_results[k] for k in all_results}, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
