"""
H1 + 组内中位数收缩：对同(date,flow_point)组内的ET残差向组中位数收缩。
γ由训练集内层LODO选择。嵌套LODO验证。
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/problem4_shrink"
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

# D0 LOFPO
d0_df = df[df["disturbance_id"] == "D0"].copy()
fps = sorted(d0_df["flow_point"].unique())
d0_preds = np.zeros(len(d0_df))
for fp in fps:
    tr = d0_df["flow_point"] != fp; te = d0_df["flow_point"] == fp
    z_tr = (d0_df.loc[tr, "flow_point"].values - 50) / 30
    y_tr = np.log(d0_df.loc[tr, "standard_volume_m3"].values / d0_df.loc[tr, "v_owics"].values)
    z_te = (d0_df.loc[te, "flow_point"].values - 50) / 30
    d0_preds[te.values] = np.polyval(np.polyfit(z_tr, y_tr, 1), z_te)

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

def apply_shrink(r_hat, df_sub, gamma):
    """对df_sub中每组(date,flow_point)的r_hat做中位数收缩。"""
    result = r_hat.copy()
    for (dt, fp), grp in df_sub.groupby(["date", "flow_point"]):
        if len(grp) < 2: continue
        idx = grp.index.values
        med = np.median(r_hat[idx])
        result[idx] = (1 - gamma) * r_hat[idx] + gamma * med
    return result

ET_PG = [{"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
         for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]]
GAMMAS = [0.0, 0.25, 0.5, 0.75]

outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))
fold_gammas = []

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote].reset_index(drop=True)
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # ET选参
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

    # 最优ET重跑内层LODO获取ip_best，用于γ选择
    ip_best = np.zeros(len(tr_o))
    for itr, iva in inner.split(tr_o, groups=idates):
        tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
        Xtr_f = tri[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        Xva_f = vai[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        et = ExtraTreesRegressor(**best_p, random_state=2026, n_jobs=-1)
        et.fit(Xtr_f, np.log(tri["standard_volume_m3"].values / tri["v_phys6"].values))
        ip_best[iva] = et.predict(Xva_f)

    # γ选择（内层LODO + 训练集评价）
    best_gamma = 0.0; best_g_score = (np.inf, np.inf, -np.inf, np.inf)
    for gamma in GAMMAS:
        ip_shrunk = apply_shrink(ip_best, tr_o, gamma)
        vol = tr_o["v_phys6"].values * np.exp(ip_shrunk)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
        m_inner = ev(err, tr_o)
        sc = (m_inner["u_r"] / 0.040, m_inner["u_d"] / 0.115 if m_inner["u_d"] > 0 else 0,
              -m_inner["pass"], m_inner["MAE"])
        if sc < best_g_score: best_g_score = sc; best_gamma = gamma

    # 最终拟合+预测
    Xall = tr_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    Xte = te_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    r_tr = np.log(tr_o["standard_volume_m3"].values / tr_o["v_phys6"].values)
    et_final = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
    et_final.fit(Xall, r_tr)
    r_te = et_final.predict(Xte)
    r_te_shrunk = apply_shrink(r_te, te_o, best_gamma)

    te_pred = np.zeros(len(te_o))
    te_d0 = te_o["disturbance_id"] == "D0"
    if te_d0.sum() > 0:
        te_pred[te_d0] = te_o.loc[te_d0, "v_owics"].values * np.exp(
            d0_preds[df.index.get_indexer(te_o[te_d0].index)])
    te_pred[~te_d0] = te_o.loc[~te_d0, "v_phys6"].values * np.exp(r_te_shrunk[~te_d0])
    outer_pred[ote] = te_pred

    fold_gammas.append(best_gamma)
    test_date = str(df.iloc[ote[0]]["date"])
    mixed_sds = []
    for (dt, fp), grp in te_o.groupby(["date", "flow_point"]):
        if len(grp) < 3 or grp["disturbance_id"].nunique() < 2: continue
        idx = grp.index.values
        e = (te_pred[idx] - te_o.loc[idx, "standard_volume_m3"].values) / te_o.loc[idx, "standard_volume_m3"].values * 100
        mixed_sds.append(e.std(ddof=1))
    max_mix = max(mixed_sds) if mixed_sds else 0
    print(f"  折{of+1}/{len(set(dates))} 日期={test_date} γ={best_gamma:.2f} max_mixed_SD={max_mix:.4f}%", flush=True)

final_err = (outer_pred - std_vol) / std_vol * 100
m = ev(final_err)
e_p = (df["v_phys6"].values - std_vol) / std_vol * 100; m_p = ev(e_p)

print(f"\n{'='*55}")
print(f"               Pass    MAE       u_L       u_r       u_d")
print(f"Phys6          {m_p['pass']}/30    {m_p['MAE']:.4f}%    {m_p['u_L']:.4f}%    {m_p['u_r']:.4f}%    {m_p['u_d']:.4f}%")
print(f"H1(之前)       9/30    0.1087%   0.0330%   0.1179%   0.2457%")
print(f"H1+组内收缩    {m['pass']}/30    {m['MAE']:.4f}%    {m['u_L']:.4f}%    {m['u_r']:.4f}%    {m['u_d']:.4f}%")
print(f"选用γ: {dict(zip(*np.unique(fold_gammas,return_counts=True)))}")

pd.DataFrame([{"model":"phys6",**m_p},{"model":"h1_shrink",**m}]).to_csv(
    OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json","w",encoding="utf-8") as f: json.dump(m,f,ensure_ascii=False,indent=2)
print(f"输出: {OUT_DIR}")
