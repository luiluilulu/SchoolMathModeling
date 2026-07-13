"""
GP不确定性门控ExtraTrees：ET预测残差，GP评估可信度。
η_i = τ²/(τ²+σ²_GP,i)  或  分段门控。
τ在每折内层验证中选择。嵌套LODO。
输出: output/results/gp_gated_et/
"""
import pandas as pd, numpy as np, math, json, warnings
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/gp_gated_et"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

FEATS_ET = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
    "dyn_first_0p1_s", "dyn_tail_0p1_s", "dyn_start_over_plateau",
    "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "base_rate_m3h",
]
FEATS_GP = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
    "dyn_plateau_cv", "dyn_start_over_plateau", "base_rate_m3h",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()

dist_counts = df.groupby(["date", "flow_point"])["disturbance_id"].nunique()

# ET 参数网格（缩小以提速，保留已验证有效的组合）
ET_PG = [
    {"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
    for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]
]

# GP：固定Matern核，不优化（仅用于不确定性估计）
GP_KERNEL = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
    length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5
) + WhiteKernel(1e-4, (1e-6, 1e-1))

TAU_GRID = [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01, 1e6]
GATE_MODES = ["adaptive", "fixed_0.5", "fixed_1.0"]


def ev(err):
    work = df.copy(); work["ep"] = err
    gd = []
    for (dt, fp), grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3: continue
        ee = grp["ep"].values; dc = dist_counts.get((dt, fp), 1)
        gd.append({"m": ee.mean(), "s": ee.std(ddof=1), "dc": dc})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0
    sm = gdf["dc"] == 1; mx = gdf["dc"] > 1
    sp = int(((gdf.loc[sm,"m"].abs() <= 0.2) & (gdf.loc[sm,"s"] <= 0.040)).sum()) if sm.any() else 0
    mp = int(((gdf.loc[mx,"m"].abs() <= 0.2) & (gdf.loc[mx,"s"] <= 0.040)).sum()) if mx.any() else 0
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
            "pass_single": sp, "n_single": int(sm.sum()), "pass_mixed": mp, "n_mixed": int(mx.sum()),
            "u_L": float(ul), "u_r": float(ur), "u_d": float(ud)}


def inner_score(err, df_sub):
    gp = 0; gs = 0.0; gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


# ==== 嵌套 LODO ====
outer = LeaveOneGroupOut()
outer_r_hat = np.zeros(len(df))   # ET残差预测
outer_sigma = np.zeros(len(df))   # GP不确定性
outer_tau = np.zeros(len(df))     # 每折选的τ
fold_info = []

print(f"GP门控ET 嵌套LODO ({n_dates}折)", flush=True)

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 步骤1: ET 内层 LODO → 选最优PG
    best_et_score = (-1, -np.inf, -np.inf)
    best_et_p = None
    for p in ET_PG:
        ip_et = np.zeros(len(tr_o))
        for itr, iva in inner.split(tr_o, groups=idates):
            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
            Xtr_f = tri[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
            Xva_f = vai[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
            et = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
            et.fit(Xtr_f, tri["target"].values)
            ip_et[iva] = et.predict(Xva_f)
        vol = tr_o["base_vol"].values * np.exp(ip_et)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
        sc = inner_score(err, tr_o)
        if sc > best_et_score:
            best_et_score = sc
            best_et_p = p

    # 步骤2: GP 内层 LODO → 获取σ（固定Matern核）
    ip_gp_sigma = np.zeros(len(tr_o))
    for itr, iva in inner.split(tr_o, groups=idates):
        tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
        Xtr_f = tri[FEATS_GP].astype(float).fillna(tri[FEATS_GP].median()).values
        Xva_f = vai[FEATS_GP].astype(float).fillna(tri[FEATS_GP].median()).values
        scaler = StandardScaler()
        gp = GaussianProcessRegressor(
            kernel=GP_KERNEL, alpha=1e-6, n_restarts_optimizer=2,
            normalize_y=True, random_state=2026,
        )
        gp.fit(scaler.fit_transform(Xtr_f), tri["target"].values)
        _, sigma_va = gp.predict(scaler.transform(Xva_f), return_std=True)
        ip_gp_sigma[iva] = sigma_va

    # 步骤3: τ 选择（在ET内层预测 + GP内层σ上）
    # 先用最优ET PG重新跑一次内层LODO获取ET r̂
    ip_et_best = np.zeros(len(tr_o))
    for itr, iva in inner.split(tr_o, groups=idates):
        tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
        Xtr_f = tri[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        Xva_f = vai[FEATS_ET].astype(float).fillna(tri[FEATS_ET].median()).values
        et = ExtraTreesRegressor(**best_et_p, random_state=2026, n_jobs=-1)
        et.fit(Xtr_f, tri["target"].values)
        ip_et_best[iva] = et.predict(Xva_f)

    best_tau_score = (-1, -np.inf, -np.inf)
    best_tau = 1e6
    for tau in TAU_GRID:
        eta_i = tau**2 / (tau**2 + ip_gp_sigma**2 + 1e-20)
        vol = tr_o["base_vol"].values * np.exp(ip_et_best * eta_i)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
        sc = inner_score(err, tr_o)
        if sc > best_tau_score:
            best_tau_score = sc
            best_tau = tau

    # 步骤4: 最终拟合
    # ET
    Xtr_et = tr_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    Xte_et = te_o[FEATS_ET].astype(float).fillna(tr_o[FEATS_ET].median()).values
    et_final = ExtraTreesRegressor(**best_et_p, random_state=2026 + of, n_jobs=-1)
    et_final.fit(Xtr_et, tr_o["target"].values)

    # GP
    Xtr_gp = tr_o[FEATS_GP].astype(float).fillna(tr_o[FEATS_GP].median()).values
    Xte_gp = te_o[FEATS_GP].astype(float).fillna(tr_o[FEATS_GP].median()).values
    scaler_gp = StandardScaler()
    gp_final = GaussianProcessRegressor(
        kernel=GP_KERNEL, alpha=1e-6, n_restarts_optimizer=3,
        normalize_y=True, random_state=2026 + of,
    )
    gp_final.fit(scaler_gp.fit_transform(Xtr_gp), tr_o["target"].values)

    # 预测
    r_te = et_final.predict(Xte_et)
    _, sigma_te = gp_final.predict(scaler_gp.transform(Xte_gp), return_std=True)
    outer_r_hat[ote] = r_te
    outer_sigma[ote] = sigma_te
    outer_tau[ote] = best_tau

    test_date = str(te_o["date"].iloc[0])
    fold_info.append({
        "fold": of + 1, "test_date": test_date,
        "et_leaf": best_et_p["min_samples_leaf"],
        "et_depth": str(best_et_p["max_depth"]),
        "et_mf": best_et_p["max_features"],
        "tau": best_tau, "inner_et_pass": best_et_score[0],
        "sigma_mean": float(np.mean(sigma_te)),
    })
    print(f"  折{of+1}/{n_dates} 日期={test_date} ET=({best_et_p['min_samples_leaf']},"
          f"{best_et_p['max_depth']},{best_et_p['max_features']}) "
          f"τ={best_tau:.5f} inner_pass={best_et_score[0]} "
          f"σ̄={np.mean(sigma_te):.4f}", flush=True)

# ==== 三模式评价 ====
print(f"\n{'='*65}")
print(f"GP门控ET 结果")
print(f"{'='*65}")

results = {}
for mode in GATE_MODES:
    if mode == "adaptive":
        eta = outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20)
        label = "adaptive τ"
    elif mode == "fixed_0.5":
        eta = np.full(len(df), 0.5)
        label = "fixed η=0.5"
    else:
        eta = np.ones(len(df))
        label = "fixed η=1.0"

    vol = df["base_vol"].values * np.exp(outer_r_hat * eta)
    err = (vol - std_vol) / std_vol * 100.0
    r = ev(err)
    r["mode"] = label
    results[label] = r
    print(f"\n  {label}: pass={r['pass']}/{r['total']} "
          f"(single={r['pass_single']}/{r['n_single']} mixed={r['pass_mixed']}/{r['n_mixed']}) "
          f"MAE={r['MAE']:.4f}% u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%")
    if mode == "adaptive":
        print(f"    η范围: [{eta.min():.3f}, {eta.max():.3f}] η中位: {np.median(eta):.3f}")

# 不确定性与错误
e_adaptive = (df["base_vol"].values * np.exp(outer_r_hat * (
    outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20))) - std_vol) / std_vol * 100.0
high = outer_sigma > np.percentile(outer_sigma, 75)
low = outer_sigma <= np.percentile(outer_sigma, 25)
print(f"\n  高不确定(>P75): MAE={np.abs(e_adaptive[high]).mean():.4f}% n={high.sum()}")
print(f"  低不确定(<P25): MAE={np.abs(e_adaptive[low]).mean():.4f}% n={low.sum()}")
print(f"  区分度: {np.abs(e_adaptive[high]).mean()/np.abs(e_adaptive[low]).mean():.1f}:1")

# 汇总
e_phys6 = (df["base_vol"].values - std_vol) / std_vol * 100.0
r_phys6 = ev(e_phys6)
print(f"\n{'='*65}")
print(f"{'版本':20s} {'通过':7s} {'单扰':6s} {'混合':6s} {'MAE':8s} {'u_L':8s} {'u_r':8s} {'u_d':8s}")
print("-" * 65)
for label, r in [("Phys6", r_phys6)] + [(k, v) for k, v in results.items()]:
    print(f"{label:20s} {r['pass']:2d}/{r['total']}  "
          f"{r['pass_single']:2d}/{r['n_single']:2d}  "
          f"{r['pass_mixed']:2d}/{r['n_mixed']:2d}  "
          f"{r['MAE']:.4f}% {r['u_L']:.4f}% {r['u_r']:.4f}% {r['u_d']:.4f}%")

# 保存
all_rows = [r_phys6] + [results[v] for v in results]
pd.DataFrame(all_rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(fold_info).to_csv(OUT_DIR / "fold_details.csv", index=False, encoding="utf-8-sig")

pred_df = df[["window_id", "date", "flow_point", "disturbance_id",
               "standard_volume_m3", "base_vol"]].copy()
pred_df["r_hat_et"] = outer_r_hat
pred_df["gp_sigma"] = outer_sigma
pred_df["eta_adaptive"] = outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20)
pred_df["pred_volume_m3"] = df["base_vol"].values * np.exp(
    outer_r_hat * pred_df["eta_adaptive"].values)
pred_df["error_pct"] = e_adaptive
pred_df.to_csv(OUT_DIR / "window_predictions.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({k: results[k] for k in results}, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
