"""
高斯过程残差补偿 + 不确定性自适应收缩。
r = log(V_std/V_phys6), GP预测 r~N(μ,σ²)。
三版本对比: plain(η=1), fixed_η, adaptive_η(τ²/(τ²+σ²))。
嵌套LODO验证, τ在内层选择。
输出: output/results/gpr/
"""
import pandas as pd, numpy as np, math, json, warnings
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, Matern, WhiteKernel,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/gpr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

FEATS = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
    "dyn_plateau_cv", "dyn_start_over_plateau",
    "base_rate_m3h",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()

# 预计算评价分组标记
dist_counts = df.groupby(["date", "flow_point"])["disturbance_id"].nunique()


def ev(err):
    """官方指标 + 单/混合扰流分层。"""
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
    ns = int(sm.sum()); nm = int(mx.sum())
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
            "pass_single": sp, "n_single": ns, "pass_mixed": mp, "n_mixed": nm,
            "u_L": float(ul), "u_r": float(ur), "u_d": float(ud)}


def inner_score(err, df_sub):
    gp = 0; gs = 0.0; gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


# 核函数定义
KERNELS = {
    "RBF": ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
           + WhiteKernel(1e-4, (1e-6, 1e-1)),
    "Matern": ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0,
              length_scale_bounds=(1e-2, 1e2), nu=1.5)
              + WhiteKernel(1e-4, (1e-6, 1e-1)),
}

# τ 候选：基于内层σ分布自适应，fallback用固定网格
TAU_CANDIDATES = [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01, 1e6]  # 1e6≈η=1

# 固定η候选
ETA_FIXED = [0.25, 0.5, 0.75, 1.0]


# ==== 嵌套 LODO ====
outer = LeaveOneGroupOut()
outer_mu = np.zeros(len(df))       # GP均值预测
outer_sigma = np.zeros(len(df))    # GP预测标准差
outer_tau = np.zeros(len(df))      # 每窗口使用的τ
fold_info = []

print(f"GPR 嵌套LODO ({n_dates}折, {len(FEATS)}特征, {len(KERNELS)}核)", flush=True)

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    best_score = (-1, -np.inf, -np.inf)
    best_kernel_name = None
    best_tau = None

    for kn, kernel in KERNELS.items():
        # 内层 LODO：每核跑一次
        ip_mu = np.zeros(len(tr_o))
        ip_sigma = np.zeros(len(tr_o))
        for itr, iva in inner.split(tr_o, groups=idates):
            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
            Xtr_f = tri[FEATS].astype(float).fillna(tri[FEATS].median()).values
            Xva_f = vai[FEATS].astype(float).fillna(tri[FEATS].median()).values
            ytr = tri["target"].values

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_f)
            Xva_s = scaler.transform(Xva_f)

            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, n_restarts_optimizer=3,
                normalize_y=True, random_state=2026,
            )
            gp.fit(Xtr_s, ytr)
            ip_mu[iva], ip_sigma[iva] = gp.predict(Xva_s, return_std=True)

        # τ 搜索（后处理，不重训GP）
        sigma_vals = ip_sigma[ip_sigma > 0]
        if len(sigma_vals) > 0:
            tau_adaptive = list(set(
                [float(np.percentile(sigma_vals, p)) for p in [25, 50, 75]]
                + TAU_CANDIDATES
            ))
        else:
            tau_adaptive = TAU_CANDIDATES

        for tau in sorted(tau_adaptive):
            eta_i = tau**2 / (tau**2 + ip_sigma**2 + 1e-20)
            vol = tr_o["base_vol"].values * np.exp(ip_mu * eta_i)
            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
            sc = inner_score(err, tr_o)
            if sc > best_score:
                best_score = sc
                best_kernel_name = kn
                best_tau = tau

    # 最优核+τ，在完整外层训练集拟合
    Xtr_f = tr_o[FEATS].astype(float).fillna(tr_o[FEATS].median()).values
    Xte_f = te_o[FEATS].astype(float).fillna(tr_o[FEATS].median()).values
    ytr = tr_o["target"].values

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_f)
    Xte_s = scaler.transform(Xte_f)

    gp = GaussianProcessRegressor(
        kernel=KERNELS[best_kernel_name], alpha=1e-6,
        n_restarts_optimizer=3, normalize_y=True, random_state=2026 + of,
    )
    gp.fit(Xtr_s, ytr)
    mu_te, sigma_te = gp.predict(Xte_s, return_std=True)

    outer_mu[ote] = mu_te
    outer_sigma[ote] = sigma_te
    outer_tau[ote] = best_tau

    test_date = str(te_o["date"].iloc[0])
    fold_info.append({
        "fold": of + 1, "test_date": test_date,
        "kernel": best_kernel_name, "tau": best_tau,
        "inner_pass": best_score[0],
        "sigma_mean": float(np.mean(sigma_te)),
        "sigma_median": float(np.median(sigma_te)),
        "sigma_max": float(np.max(sigma_te)),
    })
    print(f"  折{of+1}/{n_dates} 日期={test_date} kernel={best_kernel_name} "
          f"τ={best_tau:.5f} inner_pass={best_score[0]} "
          f"σ̄={np.mean(sigma_te):.4f} σ_max={np.max(sigma_te):.4f}", flush=True)

# ==== 三版本评价 ====
results = {}
print(f"\n{'='*65}")
print(f"GPR 三版本对比 (嵌套LODO)")
print(f"{'='*65}")

for version, eta_mode in [
    ("plain (η=1)", "plain"),
    ("fixed η (best)", "fixed"),
    ("adaptive τ", "adaptive"),
]:
    if eta_mode == "plain":
        eta = np.ones(len(df))
    elif eta_mode == "fixed":
        # 全局最优固定η（在外层预测上搜索，仅做对照，不代表严格CV）
        best_fixed_score = (-1,)
        best_fixed_eta = 1.0
        for e in ETA_FIXED:
            vol = df["base_vol"].values * np.exp(outer_mu * e)
            err = (vol - std_vol) / std_vol * 100.0
            sc = inner_score(err, df)
            if sc > best_fixed_score:
                best_fixed_score = sc
                best_fixed_eta = e
        eta = np.full(len(df), best_fixed_eta)
    else:  # adaptive
        eta = outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20)

    final_vol = df["base_vol"].values * np.exp(outer_mu * eta)
    final_err = (final_vol - std_vol) / std_vol * 100.0
    r = ev(final_err)
    r["version"] = version
    results[version] = r

    print(f"\n  {version}:")
    print(f"    pass={r['pass']}/{r['total']} (single={r['pass_single']}/{r['n_single']} "
          f"mixed={r['pass_mixed']}/{r['n_mixed']})")
    print(f"    MAE={r['MAE']:.4f}% u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%")
    if eta_mode == "fixed":
        print(f"    best_global_η={best_fixed_eta:.2f}")
    if eta_mode == "adaptive":
        print(f"    η范围: [{eta.min():.3f}, {eta.max():.3f}] η中位: {np.median(eta):.3f}")

# Phys6 基线
e_phys6 = (df["base_vol"].values - std_vol) / std_vol * 100.0
r_phys6 = ev(e_phys6)
r_phys6["version"] = "phys6"

# 不确定性与错误的关系
high_unc = outer_sigma > np.percentile(outer_sigma, 75)
low_unc = outer_sigma <= np.percentile(outer_sigma, 25)
e_adaptive = (df["base_vol"].values * np.exp(outer_mu * (
    outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20))) - std_vol) / std_vol * 100.0
print(f"\n  高不确定性窗口(>P75) MAE: {np.abs(e_adaptive[high_unc]).mean():.4f}% (n={high_unc.sum()})")
print(f"  低不确定性窗口(<P25) MAE: {np.abs(e_adaptive[low_unc]).mean():.4f}% (n={low_unc.sum()})")
print(f"  高不确定性窗口主要日期: {df.loc[high_unc, 'date'].value_counts().to_dict()}")

# 汇总
print(f"\n{'='*65}")
print(f"{'版本':20s} {'通过':7s} {'单扰':6s} {'混合':6s} {'MAE':8s} {'u_L':8s} {'u_r':8s} {'u_d':8s}")
print("-" * 65)
for ver in ["phys6"] + list(results.keys()):
    r = r_phys6 if ver == "phys6" else results[ver]
    print(f"{ver:20s} {r['pass']:2d}/{r['total']}  "
          f"{r['pass_single']:2d}/{r['n_single']:2d}  "
          f"{r['pass_mixed']:2d}/{r['n_mixed']:2d}  "
          f"{r['MAE']:.4f}% {r['u_L']:.4f}% {r['u_r']:.4f}% {r['u_d']:.4f}%")

# 保存
all_rows = [r_phys6] + [results[v] for v in results]
pd.DataFrame(all_rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(fold_info).to_csv(OUT_DIR / "fold_details.csv", index=False, encoding="utf-8-sig")

pred_df = df[["window_id", "date", "flow_point", "disturbance_id",
               "standard_volume_m3", "base_vol"]].copy()
pred_df["gp_mu"] = outer_mu
pred_df["gp_sigma"] = outer_sigma
eta_adaptive = outer_tau**2 / (outer_tau**2 + outer_sigma**2 + 1e-20)
pred_df["eta_adaptive"] = eta_adaptive
pred_df["pred_volume_m3"] = df["base_vol"].values * np.exp(outer_mu * eta_adaptive)
pred_df["error_pct"] = e_adaptive
pred_df.to_csv(OUT_DIR / "window_predictions.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({v: results[v] for v in results}, f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUT_DIR}")
