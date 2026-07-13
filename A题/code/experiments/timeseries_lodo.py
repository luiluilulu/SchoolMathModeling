"""
逐秒时序特征嵌套LODO对比：原13维 | 逐秒25维 | 13+25=38维。
ExtraTrees, η=1.0, 统一嵌套留一日期验证。
输出: output/results/timeseries_lodo/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
TS_DIR = HERE / "../output/features/timeseries"
OUT_DIR = HERE / "../output/results/timeseries_lodo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

WINDOW_FEATS_13 = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
    "dyn_first_0p1_s", "dyn_tail_0p1_s", "dyn_start_over_plateau",
    "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "base_rate_m3h",
]

# 加载逐秒紧凑特征
ts_compact = pd.read_csv(TS_DIR / "timeseries_features_compact.csv", encoding="utf-8-sig")
TS_FEATS_25 = [c for c in ts_compact.columns if c != "window_id"]

# 合并数据
df = pd.read_csv(DATA, encoding="utf-8-sig")
df = df.merge(ts_compact, on="window_id", how="left", validate="one_to_one")

# 检查缺失
ts_missing = [c for c in TS_FEATS_25 if df[c].isna().any()]
if ts_missing:
    print(f"警告: {len(ts_missing)}个逐秒特征含缺失值，将用中位数填充")

duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()

FEATURE_SETS = {
    "window_13": ("原13维窗口特征", WINDOW_FEATS_13),
    "timeseries_25": ("25维逐秒紧凑特征", TS_FEATS_25),
    "combined_38": ("13+25=38维", WINDOW_FEATS_13 + TS_FEATS_25),
}

ET_PG = [
    {"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
    for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]
]


def ev(err):
    work = df.copy(); work["ep"] = err
    gd = []
    for _, grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3: continue
        ee = grp["ep"].values
        gd.append({"m": ee.mean(), "s": ee.std(ddof=1)})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0
    d0 = work[work["disturbance_id"] == "D0"]
    d0m = d0.groupby("flow_point")["ep"].mean()
    ul = math.sqrt((d0m ** 2).sum() / max(len(d0m) - 1, 1))
    ur = gdf["s"].max() if not gdf.empty else 0.0
    use = work[work["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")].groupby("flow_point")["ep"].mean()
    d1 = use[use["condition_note"].eq("disturbed_test")].groupby(
        ["disturbance_id", "flow_point"])["ep"].agg(["mean", "std"]).reset_index()
    d1["bm"] = d1["flow_point"].map(bm); d1 = d1.dropna(subset=["bm"])
    d1["drift"] = (d1["bm"] - d1["mean"]).abs()
    ud = math.sqrt((d1["drift"].max() / math.sqrt(3)) ** 2 + d1["std"].fillna(0).max() ** 2)
    return {"MAE": float(np.abs(err).mean()), "pass": gp, "total": len(gdf),
            "u_L": float(ul), "u_r": float(ur), "u_d": float(ud)}


def inner_score(err, df_sub):
    gp = 0; gs = 0.0; gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


results = {}

for key, (label, feats) in FEATURE_SETS.items():
    print(f"\n{'='*55}")
    print(f"{label} ({len(feats)}维)")
    print(f"{'='*55}", flush=True)

    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote]
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        best_score = (-1, -np.inf, -np.inf)
        best_p = None

        for p in ET_PG:
            ip = np.zeros(len(tr_o))
            for itr, iva in inner.split(tr_o, groups=idates):
                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                Xtr_f = tri[feats].astype(float)
                Xva_f = vai[feats].astype(float)
                med = Xtr_f.median()
                Xtr_f = Xtr_f.fillna(med).values
                Xva_f = Xva_f.fillna(med).values
                m = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
                m.fit(Xtr_f, tri["target"].values)
                ip[iva] = m.predict(Xva_f)
            vol = tr_o["base_vol"].values * np.exp(ip)
            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
            sc = inner_score(err, tr_o)
            if sc > best_score:
                best_score = sc
                best_p = p

        Xall = tr_o[feats].astype(float)
        med = Xall.median()
        Xall_f = Xall.fillna(med).values
        Xte_f = te_o[feats].astype(float).fillna(med).values
        fm = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
        fm.fit(Xall_f, tr_o["target"].values)
        outer_pred[ote] = fm.predict(Xte_f)

        print(f"  折{of+1}/{n_dates} 日期={te_o['date'].iloc[0]} "
              f"leaf={best_p['min_samples_leaf']} d={best_p['max_depth']} "
              f"mf={best_p['max_features']} inner={best_score[0]}", flush=True)

    final_err = (df["base_vol"].values * np.exp(outer_pred)
                 - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
    r = ev(final_err)
    r["feature_set"] = label
    r["n_features"] = len(feats)
    results[key] = r
    print(f"  => pass={r['pass']}/{r['total']} MAE={r['MAE']:.4f}% "
          f"u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%", flush=True)

# 汇总
e_phys6 = (df["base_vol"].values - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
r_phys6 = ev(e_phys6)
r_phys6["feature_set"] = "Phys6基线"

print(f"\n{'='*75}")
print(f"逐秒时序特征 嵌套LODO 结果")
print(f"{'='*75}")
print(f"{'特征集':20s} {'维':>3s} {'Pass':>7s} {'MAE':>8s} {'u_L':>8s} {'u_r':>8s} {'u_d':>8s}")
print("-" * 75)
for r in [r_phys6] + [results[k] for k in ["window_13", "timeseries_25", "combined_38"]]:
    print(f"{r['feature_set']:20s} {r.get('n_features',0):3d} "
          f"{r['pass']:2d}/{r['total']}  {r['MAE']:.4f}% {r['u_L']:.4f}% {r['u_r']:.4f}% {r['u_d']:.4f}%")

# 与历史最优比较
print(f"\n  历史最优 u_r: ET(28d) = 0.1144%")
best_ur = min(r["u_r"] for k, r in results.items())
print(f"  本次最优 u_r: {best_ur:.4f}%")

rows = [r_phys6] + [results[k] for k in ["window_13", "timeseries_25", "combined_38"]]
pd.DataFrame(rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({k: results[k] for k in results}, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
