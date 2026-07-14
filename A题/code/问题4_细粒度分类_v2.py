"""
H1 + 细粒度分类 v2：训练集计算子类补偿，测试集应用。
D1/D2和D3/D4拆分后用训练集子类均值修正，不做测试集归零。
嵌套LODO，严格验证。
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/problem4_finegrained_v2"
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

df = pd.read_csv(DATA, encoding="utf-8-sig")
dur = df["duration_s"].astype(float).values; ch = df[CHORD_COLS].astype(float).values
df["v_owics"] = AREA * dur * (ch @ W_OWICS)
df["v_phys6"] = AREA * dur * (ch @ W_PHYS6)
df["base_rate_m3h"] = df["v_phys6"] / dur * 3600
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values

# D0 LOFPO校准
d0_df = df[df["disturbance_id"] == "D0"].copy()
fps = sorted(d0_df["flow_point"].unique())
d0_preds = np.zeros(len(d0_df))
for fp in fps:
    tr = d0_df["flow_point"] != fp; te = d0_df["flow_point"] == fp
    z_tr = (d0_df.loc[tr, "flow_point"].values - 50) / 30
    y_tr = np.log(d0_df.loc[tr, "standard_volume_m3"].values / d0_df.loc[tr, "v_owics"].values)
    z_te = (d0_df.loc[te, "flow_point"].values - 50) / 30
    d0_preds[te.values] = np.polyval(np.polyfit(z_tr, y_tr, 1), z_te)

# 细粒度分类器工厂
def build_fine_classifier(train_df):
    """返回(classify_fn, subgroup_corrections_dict)"""
    corrections = {}
    avail = {"d1d2": False, "d3d4": False}

    # D1 vs D2: profile_inner_skew (D1 > D2)
    tr_d1 = train_df[train_df["disturbance_id"] == "D1"]
    tr_d2 = train_df[train_df["disturbance_id"] == "D2"]
    th_d1d2 = 0
    if len(tr_d1) >= 2:
        avail["d1d2"] = True
        if len(tr_d2) >= 2:
            th_d1d2 = (tr_d1["profile_inner_skew"].mean() + tr_d2["profile_inner_skew"].mean()) / 2
        else:
            th_d1d2 = tr_d1["profile_inner_skew"].mean() - tr_d1["profile_inner_skew"].std()
        # 训练集子类补偿: 各自Phys6残差均值
        r_d1 = np.log(tr_d1["standard_volume_m3"].values / tr_d1["v_phys6"].values).mean()
        if len(tr_d2) >= 2:
            r_d2 = np.log(tr_d2["standard_volume_m3"].values / tr_d2["v_phys6"].values).mean()
        else:
            r_d2 = r_d1  # fallback
        corrections["D1"] = r_d1; corrections["D2"] = r_d2

    # D3 vs D4: profile_center_all + profile_top_bottom (D3 > D4 on center)
    tr_d3 = train_df[train_df["disturbance_id"] == "D3"]
    tr_d4 = train_df[train_df["disturbance_id"] == "D4"]
    th_d3d4_f1 = 0; th_d3d4_f2 = 0; ref_d3d4 = 1.0
    if len(tr_d3) >= 2 and len(tr_d4) >= 2:
        avail["d3d4"] = True
        th_d3d4_f1 = (tr_d3["profile_center_all"].mean() + tr_d4["profile_center_all"].mean()) / 2
        th_d3d4_f2 = (tr_d3["profile_top_bottom"].mean() + tr_d4["profile_top_bottom"].mean()) / 2
        ref_d3d4 = abs(tr_d3["profile_center_all"].mean() - tr_d4["profile_center_all"].mean()) + 1e-6
        r_d3 = np.log(tr_d3["standard_volume_m3"].values / tr_d3["v_phys6"].values).mean()
        r_d4 = np.log(tr_d4["standard_volume_m3"].values / tr_d4["v_phys6"].values).mean()
        corrections["D3"] = r_d3; corrections["D4"] = r_d4
    elif len(tr_d3) >= 2:
        avail["d3d4"] = True
        th_d3d4_f1 = tr_d3["profile_center_all"].mean() - tr_d3["profile_center_all"].std()
        th_d3d4_f2 = tr_d3["profile_top_bottom"].mean() - tr_d3["profile_top_bottom"].std()
        ref_d3d4 = tr_d3["profile_center_all"].std() + 1e-6
        r_d3 = np.log(tr_d3["standard_volume_m3"].values / tr_d3["v_phys6"].values).mean()
        corrections["D3"] = r_d3; corrections["D4"] = r_d3  # D4 fallback

    # 未细分的扰流: 用全部扰流均值
    all_dist = train_df[train_df["disturbance_id"] != "D0"]
    r_all = np.log(all_dist["standard_volume_m3"].values / all_dist["v_phys6"].values).mean()
    corrections["_default"] = r_all

    def classify(df_sub):
        """返回细粒度标签，用于查corrections表。"""
        labels = np.full(len(df_sub), "_default", dtype=object)
        ab = np.where(df_sub["profile_top_bottom"].values > -0.02, "A", "B")
        if avail["d1d2"]:
            mask_a = (ab == "A")
            inner = df_sub.loc[mask_a, "profile_inner_skew"].values
            labels[mask_a] = np.where(inner > th_d1d2, "D1", "D2")
        if avail["d3d4"]:
            mask_b = (ab == "B")
            f1 = df_sub.loc[mask_b, "profile_center_all"].values
            f2 = df_sub.loc[mask_b, "profile_top_bottom"].values
            score = (f1 - th_d3d4_f1) / ref_d3d4 + (f2 - th_d3d4_f2) / ref_d3d4
            labels[mask_b] = np.where(score > 0, "D3", "D4")
        return labels

    return classify, corrections


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
    ul = math.sqrt((d0m**2).sum() / max(len(d0m) - 1, 1)) if len(d0m) > 1 else 0.0
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
    else: ud = 0.0
    return {"pass": gp, "total": len(gdf), "u_L": float(ul), "u_r": float(ur),
            "u_d": float(ud), "MAE": float(np.abs(err).mean())}

def inner_score(err, df_sub):
    gp=0; gs=0.0; gm=0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


ET_PG = [{"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
         for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]]

outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote].reset_index(drop=True)
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 细粒度分类器 + 训练集子类补偿
    fine_cls, corrections = build_fine_classifier(tr_o)
    tr_dist_labels = fine_cls(tr_o[tr_o["disturbance_id"] != "D0"])

    # ET选参（内层LODO）
    best_is = (-1, -np.inf, -np.inf); best_p = None
    for p in ET_PG:
        ip = np.zeros(len(tr_o))
        for itr, iva in inner.split(tr_o, groups=idates):
            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
            Xtr_f = tri[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
            Xva_f = vai[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
            r_tr = np.log(tri["standard_volume_m3"].values / tri["v_phys6"].values)
            et = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
            et.fit(Xtr_f, r_tr)
            ip[iva] = et.predict(Xva_f)
        vol = tr_o["v_phys6"].values * np.exp(ip)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
        iscore = inner_score(err, tr_o)
        if iscore > best_is: best_is = iscore; best_p = p

    # 最终拟合ET
    Xall = tr_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    Xte = te_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    r_tr_all = np.log(tr_o["standard_volume_m3"].values / tr_o["v_phys6"].values)
    et_final = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
    et_final.fit(Xall, r_tr_all)
    r_te_et = et_final.predict(Xte)

    # 测试集预测
    te_pred = np.zeros(len(te_o))
    te_d0 = te_o["disturbance_id"] == "D0"
    te_dist_mask = ~te_d0

    # D0校准
    if te_d0.sum() > 0:
        te_pred[te_d0] = te_o.loc[te_d0, "v_owics"].values * np.exp(
            d0_preds[df.index.get_indexer(te_o[te_d0].index)])

    # 扰流: Phys6 * exp(ET残差 + 子类补偿)
    if te_dist_mask.sum() > 0:
        te_labels = fine_cls(te_o[te_dist_mask])
        # ET残差: 用训练集子类均值替换ET对个别窗口的过度预测
        # 策略: ET预测全残差，细粒度子类修正只加训练集子类偏移
        sub_corr = np.array([corrections.get(lbl, corrections["_default"])
                             for lbl in te_labels])
        # 混合: ET预测 + 子类偏移(训练集均值)
        r_final = r_te_et[te_dist_mask] * 0.5 + sub_corr * 0.5  # 各取一半
        te_pred[te_dist_mask] = te_o.loc[te_dist_mask, "v_phys6"].values * np.exp(r_final)

    outer_pred[ote] = te_pred
    test_date = str(df.iloc[ote[0]]["date"])
    # 算当日混合组SD
    mixed_sds = []
    for (dt, fp), grp in te_o.groupby(["date", "flow_point"]):
        if len(grp) < 3 or grp["disturbance_id"].nunique() < 2: continue
        idx = grp.index.values
        e = (te_pred[idx] - te_o.loc[idx, "standard_volume_m3"].values) / te_o.loc[idx, "standard_volume_m3"].values * 100
        mixed_sds.append(e.std(ddof=1))
    max_mix = max(mixed_sds) if mixed_sds else 0
    print(f"  折{of+1}/{len(set(dates))} 日期={test_date} best_is={best_is[0]} max_mixed_SD={max_mix:.4f}%", flush=True)

final_err = (outer_pred - std_vol) / std_vol * 100
m = ev(final_err)
e_p = (df["v_phys6"].values - std_vol) / std_vol * 100; m_p = ev(e_p)

print(f"\n{'='*55}")
print(f"Phys6:        pass={m_p['pass']}/30 u_L={m_p['u_L']:.4f}% u_r={m_p['u_r']:.4f}% u_d={m_p['u_d']:.4f}%")
print(f"H1(之前):     pass=9/30     u_L=0.0330% u_r=0.1179% u_d=0.2457%")
print(f"H1+细粒度v2:  pass={m['pass']}/30 u_L={m['u_L']:.4f}% u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}%")

pd.DataFrame([{"model":"phys6",**m_p},{"model":"h1_fine_v2",**m}]).to_csv(
    OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
print(f"输出: {OUT_DIR}")
