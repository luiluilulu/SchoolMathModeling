"""
最终模型：D0(OWICS+LOFPO) + 扰流(Phys6+固定ET+组内中位数收缩)。
ET超参固定(leaf=5/depth=6/mf=0.5/nest=200)，仅外层LODO选γ。
简化自嵌套LODO——消融实验已证明ET超参在本数据上不敏感。
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/final_model"
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
ET_PARAMS = {"min_samples_leaf": 5, "max_depth": 6, "max_features": 0.5, "n_estimators": 200}
GAMMAS = [0.0, 0.25, 0.5, 0.75]

df = pd.read_csv(DATA, encoding="utf-8-sig")
dur = df["duration_s"].astype(float).values; ch = df[CHORD_COLS].astype(float).values
df["v_owics"] = AREA * dur * (ch @ W_OWICS)
df["v_phys6"] = AREA * dur * (ch @ W_PHYS6)
df["base_rate_m3h"] = df["v_phys6"] / dur * 3600
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values

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

def apply_shrink(r_hat, df_sub, gamma):
    result = r_hat.copy()
    for (dt, fp), grp in df_sub.groupby(["date", "flow_point"]):
        if len(grp) < 2: continue
        idx = grp.index.values
        med = np.median(r_hat[idx])
        result[idx] = (1 - gamma) * r_hat[idx] + gamma * med
    return result

def inner_score(err, df_sub):
    gp=0; gs=0.0; gm=0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)

# ==== 外层LODO（固定ET，仅选γ）====
outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))
fold_gammas = []

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote].reset_index(drop=True)
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 折内D0 LOFPO
    d0_tr = tr_o[tr_o["disturbance_id"] == "D0"]

    # ET固定参数拟合+内层LODO获取OOF预测
    ip_oof = np.zeros(len(tr_o))
    for itr, iva in inner.split(tr_o, groups=idates):
        tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
        Xtr_f = tri[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        Xva_f = vai[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        et = ExtraTreesRegressor(**ET_PARAMS, random_state=2026, n_jobs=-1)
        et.fit(Xtr_f, np.log(tri["standard_volume_m3"].values / tri["v_phys6"].values))
        ip_oof[iva] = et.predict(Xva_f)

    # γ选择：内层OOF上仅选γ
    best_gamma = 0.0; best_gs = (np.inf, np.inf, -np.inf, np.inf)
    for gamma in GAMMAS:
        ip_shrunk = apply_shrink(ip_oof, tr_o, gamma)
        vol = tr_o["v_phys6"].values * np.exp(ip_shrunk)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
        m_inner = ev(err, tr_o)
        sc = (m_inner["u_r"] / 0.040, m_inner["u_d"] / 0.115 if m_inner["u_d"] > 0 else 0,
              -m_inner["pass"], m_inner["MAE"])
        if sc < best_gs: best_gs = sc; best_gamma = gamma

    # 最终拟合+预测
    Xall = tr_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    Xte = te_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    et = ExtraTreesRegressor(**ET_PARAMS, random_state=2026 + of, n_jobs=-1)
    et.fit(Xall, np.log(tr_o["standard_volume_m3"].values / tr_o["v_phys6"].values))
    r_te = et.predict(Xte)
    r_te_shrunk = apply_shrink(r_te, te_o, best_gamma)

    te_pred = np.zeros(len(te_o))
    te_d0 = te_o["disturbance_id"] == "D0"
    if te_d0.sum() > 0:
        if len(d0_tr) >= 4:
            for fp_te in sorted(te_o.loc[te_d0, "flow_point"].unique()):
                fp_mask = (te_o["flow_point"] == fp_te) & te_d0
                z_val = (fp_te - 50) / 30
                z_tr = (d0_tr["flow_point"].values - 50) / 30
                y_tr = np.log(d0_tr["standard_volume_m3"].values / d0_tr["v_owics"].values)
                a, b = np.polyfit(z_tr, y_tr, 1)
                te_pred[fp_mask] = te_o.loc[fp_mask, "v_owics"].values * np.exp(a * z_val + b)
        else:
            te_pred[te_d0] = te_o.loc[te_d0, "v_owics"].values
    te_pred[~te_d0] = te_o.loc[~te_d0, "v_phys6"].values * np.exp(r_te_shrunk[~te_d0])
    outer_pred[ote] = te_pred

    fold_gammas.append(best_gamma)
    test_date = str(df.iloc[ote[0]]["date"])
    print(f"  折{of+1}/{len(set(dates))} 日期={test_date} γ={best_gamma:.2f}", flush=True)

final_err = (outer_pred - std_vol) / std_vol * 100
m = ev(final_err)
e_p = (df["v_phys6"].values - std_vol) / std_vol * 100; m_p = ev(e_p)

print(f"\n最终模型 (固定ET + LODO选γ)")
print(f"  Phys6: pass={m_p['pass']}/30 u_L={m_p['u_L']:.4f}% u_r={m_p['u_r']:.4f}% u_d={m_p['u_d']:.4f}% MAE={m_p['MAE']:.4f}%")
print(f"  本模型: pass={m['pass']}/30 u_L={m['u_L']:.4f}% u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}% MAE={m['MAE']:.4f}%")
print(f"  γ分布: {dict(zip(*np.unique(fold_gammas, return_counts=True)))}")

pd.DataFrame([{"model":"phys6",**m_p},{"model":"final",**m}]).to_csv(
    OUT_DIR / "final_result.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json","w",encoding="utf-8") as f: json.dump(m, f, ensure_ascii=False, indent=2)
print(f"输出: {OUT_DIR}")
