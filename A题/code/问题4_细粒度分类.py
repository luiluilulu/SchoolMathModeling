"""
H1 + 细粒度分类：A类内分D1/D2，B类内分D3/D4。
仅对混合组中同大类的窗口做拆分，各自补偿以消除between-SD。
嵌套LODO验证。输出: output/results/problem4_finegrained/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/problem4_finegrained"
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
d0_mask = df["disturbance_id"] == "D0"
dist_mask = ~d0_mask

# ---- D0 LOFPO校准（复用已验证结果）----
def d0_lofpo_calibration():
    d0_df = df[d0_mask].copy()
    fps = sorted(d0_df["flow_point"].unique())
    best_ul = np.inf; best_preds = None
    for method in ["none", "k0", "linear"]:
        preds = np.zeros(len(d0_df))
        for fp in fps:
            tr = d0_df["flow_point"] != fp; te = d0_df["flow_point"] == fp
            z_tr = (d0_df.loc[tr, "flow_point"].values - 50) / 30
            y_tr = np.log(d0_df.loc[tr, "standard_volume_m3"].values / d0_df.loc[tr, "v_owics"].values)
            z_te = (d0_df.loc[te, "flow_point"].values - 50) / 30
            if method == "none": preds[te.values] = 0
            elif method == "k0": preds[te.values] = np.mean(y_tr)
            else: preds[te.values] = np.polyval(np.polyfit(z_tr, y_tr, 1), z_te)
        v = d0_df["v_owics"].values * np.exp(preds)
        e = (v - d0_df["standard_volume_m3"].values) / d0_df["standard_volume_m3"].values * 100
        fp_m = pd.Series(e, index=d0_df.index).groupby(d0_df["flow_point"]).mean()
        ul = math.sqrt((fp_m**2).sum() / max(len(fp_m) - 1, 1))
        if ul < best_ul: best_ul = ul; best_preds = preds
    return best_preds

d0_cal = d0_lofpo_calibration()

# ---- 细粒度分类器 ----
def build_fine_classifier(train_df):
    """在训练集上构建D1/D2和D3/D4分类器。返回(分类函数, 可用标志)。"""
    avail = {"d1d2": False, "d3d4": False}
    th_d1d2 = 0; th_d3d4_f1 = 0; th_d3d4_f2 = 0; ref_d3d4 = 0

    # D1 vs D2: 需要D1在训练集中
    tr_d1 = train_df[train_df["disturbance_id"] == "D1"]
    tr_d2 = train_df[train_df["disturbance_id"] == "D2"]
    if len(tr_d1) >= 2:
        avail["d1d2"] = True
        if len(tr_d2) >= 2:
            th_d1d2 = (tr_d1["profile_inner_skew"].mean() + tr_d2["profile_inner_skew"].mean()) / 2
        else:
            # D2不在训练集，用D1均值-1σ做阈值（D2的inner_skew更低）
            th_d1d2 = tr_d1["profile_inner_skew"].mean() - tr_d1["profile_inner_skew"].std()

    # D3 vs D4: 需要D3或D4在训练集中
    tr_d3 = train_df[train_df["disturbance_id"] == "D3"]
    tr_d4 = train_df[train_df["disturbance_id"] == "D4"]
    if len(tr_d3) >= 2 and len(tr_d4) >= 2:
        avail["d3d4"] = True
        th_d3d4_f1 = (tr_d3["profile_center_all"].mean() + tr_d4["profile_center_all"].mean()) / 2
        th_d3d4_f2 = (tr_d3["profile_top_bottom"].mean() + tr_d4["profile_top_bottom"].mean()) / 2
        ref_d3d4 = abs(tr_d3["profile_center_all"].mean() - tr_d4["profile_center_all"].mean()) + 1e-6
    elif len(tr_d3) >= 2:
        avail["d3d4"] = True
        th_d3d4_f1 = tr_d3["profile_center_all"].mean() - tr_d3["profile_center_all"].std()
        th_d3d4_f2 = tr_d3["profile_top_bottom"].mean() - tr_d3["profile_top_bottom"].std()
        ref_d3d4 = tr_d3["profile_center_all"].std() + 1e-6

    def classify(df_sub):
        """对df_sub中的扰流窗口做细粒度分类。返回更细的类别标签。"""
        labels = np.full(len(df_sub), "none", dtype=object)
        # 先用A/B分类确定大类方向（简化：用profile_top_bottom正负）
        ab_label = np.where(df_sub["profile_top_bottom"].values > -0.02, "A", "B")

        if avail["d1d2"]:
            mask_a = (ab_label == "A")
            inner = df_sub.loc[mask_a, "profile_inner_skew"].values
            labels[mask_a] = np.where(inner > th_d1d2, "D1", "D2")

        if avail["d3d4"]:
            mask_b = (ab_label == "B")
            f1 = df_sub.loc[mask_b, "profile_center_all"].values
            f2 = df_sub.loc[mask_b, "profile_top_bottom"].values
            score = (f1 - th_d3d4_f1) / ref_d3d4 + (f2 - th_d3d4_f2) / ref_d3d4
            labels[mask_b] = np.where(score > 0, "D3", "D4")

        return labels

    return classify, avail


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


# ---- 主流程 ----
ET_PG = [{"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
         for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]]

print("H1 + 细粒度分类 嵌套LODO", flush=True)
outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))
fold_groups = []  # 记录每折混合组SD变化

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote].reset_index(drop=True)
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 细粒度分类器
    fine_cls, fine_avail = build_fine_classifier(tr_o)
    te_labels = fine_cls(te_o) if fine_avail["d1d2"] or fine_avail["d3d4"] else np.full(len(te_o), "none")

    # 为每个细粒度子类拟合ET（在A/B分类之后）
    # 简化：在A/B大类基础上，细粒度子类各自补偿
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

    # 最终拟合
    Xall = tr_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    Xte = te_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    r_tr = np.log(tr_o["standard_volume_m3"].values / tr_o["v_phys6"].values)
    et_final = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
    et_final.fit(Xall, r_tr)
    r_te = et_final.predict(Xte)

    # 细粒度补偿修正：对混合组中的细粒度子类做均值归零
    te_dist = te_o[te_o["disturbance_id"] != "D0"]
    if len(te_dist) > 0 and (fine_avail["d1d2"] or fine_avail["d3d4"]):
        # 对ET预测后的残差再按细粒度标签做组内归零
        r_te_dist = r_te[te_o["disturbance_id"] != "D0"]
        for lbl in np.unique(te_labels[te_labels != "none"]):
            mask = (te_labels == lbl)
            if mask.sum() >= 2:
                r_te[mask] -= r_te[mask].mean()  # 同一细粒度子类内归零

    # 预测
    te_pred = np.zeros(len(te_o))
    te_d0 = te_o["disturbance_id"] == "D0"
    if te_d0.sum() > 0:
        te_pred[te_d0] = te_o.loc[te_d0, "v_owics"].values * np.exp(
            d0_cal[df.index.get_indexer(te_o[te_d0].index)])
    te_pred[~te_d0] = te_o.loc[~te_d0, "v_phys6"].values * np.exp(r_te[~te_d0])
    outer_pred[ote] = te_pred

    # 记录混合组SD
    test_date = str(df.iloc[ote[0]]["date"])
    mixed_sds = []
    for (dt, fp), grp in te_o.groupby(["date", "flow_point"]):
        if len(grp) < 3 or grp["disturbance_id"].nunique() < 2: continue
        idx = grp.index.values
        e = (te_pred[idx] - te_o.loc[idx, "standard_volume_m3"].values) / te_o.loc[idx, "standard_volume_m3"].values * 100
        mixed_sds.append(e.std(ddof=1))
    max_mixed_sd = max(mixed_sds) if mixed_sds else 0
    print(f"  折{of+1}/{len(set(dates))} 日期={test_date} "
          f"best_is={best_is[0]} fine_avail={fine_avail} max_mixed_SD={max_mixed_sd:.4f}%", flush=True)

final_err = (outer_pred - std_vol) / std_vol * 100
m = ev(final_err)
print(f"\n=> pass={m['pass']}/{m['total']} MAE={m['MAE']:.4f}% "
      f"u_L={m['u_L']:.4f}% u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}%")

# 对照H1
e_p = (df["v_phys6"].values - std_vol) / std_vol * 100
m_p = ev(e_p)
print(f"\nPhys6:  pass={m_p['pass']}/{m_p['total']} u_r={m_p['u_r']:.4f}% u_d={m_p['u_d']:.4f}%")
print(f"H1:     pass=9/30 u_r=0.1179% u_d=0.2457% (之前)")
print(f"本模型: pass={m['pass']}/{m['total']} u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}%")

pd.DataFrame([{"model":"phys6",**m_p},{"model":"h1_finegrained",**m}]).to_csv(
    OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
