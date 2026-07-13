"""
五目标综合优化：内层多目标选参 + 流量趋势后标定 + 组内收缩 + 扰流组稳健加权。
嵌套LODO验证，ET(28d)基线。
输出: output/results/optimize_five/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/optimize_five"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
FEATS_28 = CHORD_COLS + AB_COLS + [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner", "profile_inner_skew",
    "profile_ab_abs", "profile_swirl", "dyn_first_0p1_s", "dyn_tail_0p1_s",
    "dyn_start_over_plateau", "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "zero_rate_med", "zero_rate_mad", "zero_age_s", "base_rate_m3h", "duration_s", "flow_point",
]

ET_PG = [
    {"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
    for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()


def compute_full_metrics(err, df_ref):
    """在子集df_ref上计算完整官方指标。err对齐df_ref索引。"""
    work = df_ref.copy()
    work["ep"] = err
    gd = []
    for _, grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3: continue
        ee = grp["ep"].values
        gd.append({"m": ee.mean(), "s": ee.std(ddof=1)})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0

    # u_L
    d0 = work[work["disturbance_id"] == "D0"]
    if len(d0) >= 2:
        d0m = d0.groupby("flow_point")["ep"].mean()
        ul = math.sqrt((d0m ** 2).sum() / max(len(d0m) - 1, 1))
    else:
        ul = np.nan

    # u_r
    ur = gdf["s"].max() if not gdf.empty else 0.0

    # u_d
    use = work[work["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")]
    dt = use[use["condition_note"].eq("disturbed_test")]
    if len(bm) >= 1 and len(dt) >= 1:
        bm_mean = bm.groupby("flow_point")["ep"].mean()
        d1 = dt.groupby(["disturbance_id", "flow_point"])["ep"].agg(["mean", "std"]).reset_index()
        d1["bm"] = d1["flow_point"].map(bm_mean)
        d1 = d1.dropna(subset=["bm"])
        d1["drift"] = (d1["bm"] - d1["mean"]).abs()
        udc = d1["drift"].max() / math.sqrt(3)
        udr = d1["std"].fillna(0).max()
        ud = math.sqrt(udc ** 2 + udr ** 2)
    else:
        ud = np.nan

    mae = float(np.abs(err).mean()) if len(err) > 0 else np.nan
    return {"pass": gp, "total": len(gdf), "u_L": ul, "u_r": ur, "u_d": ud, "MAE": mae}


def five_target_score(metrics):
    """五目标综合评分。越小越好。缺失指标不计入。"""
    terms = []
    if np.isfinite(metrics.get("u_L", np.nan)):
        terms.append((metrics["u_L"] / 0.036) ** 2)
    if np.isfinite(metrics.get("u_r", np.nan)):
        terms.append((metrics["u_r"] / 0.040) ** 2)
    if np.isfinite(metrics.get("u_d", np.nan)):
        terms.append((metrics["u_d"] / 0.115) ** 2)
    return np.mean(terms) if terms else np.inf


# ==== Phase 1: 多目标内层选参 ====
print("Phase 1: 五目标内层选参", flush=True)
outer = LeaveOneGroupOut()
outer_pred_phase1 = np.zeros(len(df))

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 收集所有候选的完整指标
    candidates = []
    for p in ET_PG:
        ip = np.zeros(len(tr_o))
        for itr, iva in inner.split(tr_o, groups=idates):
            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
            Xtr = tri[FEATS_28].astype(float); Xva = vai[FEATS_28].astype(float)
            med = Xtr.median()
            et = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
            et.fit(Xtr.fillna(med).values, tri["target"].values)
            ip[iva] = et.predict(Xva.fillna(med).values)
        vol = tr_o["base_vol"].values * np.exp(ip)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
        m = compute_full_metrics(err, tr_o)
        m["pg"] = p
        candidates.append(m)

    # 五目标选择：允许牺牲1个通过组
    best_pass = max(c["pass"] for c in candidates)
    best_p = None
    best_loss = np.inf
    for c in candidates:
        if c["pass"] < best_pass - 1:
            continue
        loss = five_target_score(c)
        tiebreaker = (-c["pass"], c["MAE"] if np.isfinite(c["MAE"]) else np.inf)
        if loss < best_loss or (abs(loss - best_loss) < 1e-10 and tiebreaker < best_tiebreaker):
            best_loss = loss
            best_tiebreaker = tiebreaker
            best_p = c["pg"]
            best_m = c

    # 拟合 + 预测
    Xall = tr_o[FEATS_28].astype(float); med = Xall.median()
    Xte = te_o[FEATS_28].astype(float).fillna(med).values
    et = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
    et.fit(Xall.fillna(med).values, tr_o["target"].values)
    outer_pred_phase1[ote] = et.predict(Xte)

    print(f"  折{of+1}/{n_dates} 日期={te_o['date'].iloc[0]} "
          f"leaf={best_p['min_samples_leaf']} d={best_p['max_depth']} "
          f"mf={best_p['max_features']} pass={best_m['pass']} "
          f"u_L={best_m.get('u_L',np.nan):.4f} u_r={best_m.get('u_r',0):.4f} loss={best_loss:.2f}",
          flush=True)

r_hat_et = outer_pred_phase1
e1 = (df["base_vol"].values * np.exp(r_hat_et) - std_vol) / std_vol * 100.0
m1 = compute_full_metrics(e1, df)
print(f"\n  Phase1结果: pass={m1['pass']}/{m1['total']} MAE={m1['MAE']:.4f}% "
      f"u_L={m1['u_L']:.4f}% u_r={m1['u_r']:.4f}% u_d={m1['u_d']:.4f}% loss={five_target_score(m1):.2f}")

# ==== Phase 2: 流量趋势后标定 ====
print("\nPhase 2: 流量趋势后标定", flush=True)
outer_pred_phase2 = np.zeros(len(df))

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]

    # 用Phase1的ET预测获取外层训练集的r_hat
    r_hat_tr = r_hat_et[otr]
    resid_tr = tr_o["target"].values - r_hat_tr  # ET未能解释的残差

    # 简单线性标定: resid ~ a + b * flow_point_normalized
    fp = tr_o["flow_point"].astype(float).values
    fp_mean, fp_std = fp.mean(), fp.std()
    fp_norm = (fp - fp_mean) / max(fp_std, 1e-6)

    # Ridge拟合流量趋势
    X_cal = np.column_stack([np.ones(len(tr_o)), fp_norm])
    cal = Ridge(alpha=1.0)
    cal.fit(X_cal, resid_tr)

    # 预测
    fp_te = te_o["flow_point"].astype(float).values
    fp_te_norm = (fp_te - fp_mean) / max(fp_std, 1e-6)
    X_te_cal = np.column_stack([np.ones(len(te_o)), fp_te_norm])
    correction = cal.predict(X_te_cal)

    outer_pred_phase2[ote] = r_hat_et[ote] + correction
    print(f"  折{of+1}/{n_dates} 日期={te_o['date'].iloc[0]} "
          f"cal_a={cal.coef_[0] if len(cal.coef_)>1 else 0:.4e} cal_b={cal.intercept_:.4e}",
          flush=True)

e2 = (df["base_vol"].values * np.exp(outer_pred_phase2) - std_vol) / std_vol * 100.0
m2 = compute_full_metrics(e2, df)
print(f"\n  Phase2结果: pass={m2['pass']}/{m2['total']} MAE={m2['MAE']:.4f}% "
      f"u_L={m2['u_L']:.4f}% u_r={m2['u_r']:.4f}% u_d={m2['u_d']:.4f}% loss={five_target_score(m2):.2f}")

# ==== Phase 3: 组内收缩（外层测试集上应用，γ在内层选择）====
print("\nPhase 3: 组内收缩", flush=True)
GAMMA_GRID = [0.0, 0.25, 0.5, 0.75]
outer_pred_phase3 = np.zeros(len(df))

def apply_group_shrinkage(r_hat, df_sub, gamma):
    """对df_sub中的r_hat按(date,flow_point)组做中位数收缩。
    r_hat: numpy数组, 位置索引对齐df_sub.reset_index(drop=True)。
    df_sub: 需reset_index后的DataFrame。
    """
    result = r_hat.copy()
    for (dt, fp), grp in df_sub.groupby(["date", "flow_point"]):
        if len(grp) < 2:
            continue
        idx = grp.index.values  # reset_index后为0..n-1连续索引
        g_vals = r_hat[idx]
        med = np.median(g_vals)
        result[idx] = (1 - gamma) * g_vals + gamma * med
    return result

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote].reset_index(drop=True)
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 内层选γ
    r_tr_phase2 = outer_pred_phase2[otr]  # 位置索引，对齐tr_o
    best_gamma_score = (-1, np.inf, np.inf)
    best_gamma = 0.0
    for gamma in GAMMA_GRID:
        r_shrunk = apply_group_shrinkage(r_tr_phase2, tr_o, gamma)
        vol = tr_o["base_vol"].values * np.exp(r_shrunk)
        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
        m_inner = compute_full_metrics(err, tr_o)
        sc = (m_inner["pass"], five_target_score(m_inner), m_inner.get("MAE", np.inf))
        if sc[0] > best_gamma_score[0] or (sc[0] == best_gamma_score[0] and sc[1] < best_gamma_score[1]):
            best_gamma_score = sc
            best_gamma = gamma

    # 对外层测试集应用最优γ
    r_te_phase2 = outer_pred_phase2[ote]
    outer_pred_phase3[ote] = apply_group_shrinkage(r_te_phase2, te_o, best_gamma)
    print(f"  折{of+1}/{n_dates} γ={best_gamma:.2f}", flush=True)

e3 = (df["base_vol"].values * np.exp(outer_pred_phase3) - std_vol) / std_vol * 100.0
m3 = compute_full_metrics(e3, df)
print(f"  Phase3结果: pass={m3['pass']}/{m3['total']} MAE={m3['MAE']:.4f}% "
      f"u_L={m3['u_L']:.4f}% u_r={m3['u_r']:.4f}% u_d={m3['u_d']:.4f}% loss={five_target_score(m3):.2f}")

# ==== 汇总 ====
e_phys6 = (df["base_vol"].values - std_vol) / std_vol * 100.0
m_phys6 = compute_full_metrics(e_phys6, df)

print(f"\n{'='*75}")
print(f"五目标优化 汇总")
print(f"{'='*75}")
print(f"{'版本':20s} {'Pass':>7s} {'MAE':>8s} {'u_L':>8s} {'u_r':>8s} {'u_d':>8s} {'loss':>8s}")
print("-" * 75)
for label, m in [("Phys6", m_phys6), ("Phase1 多目标选参", m1),
                  ("Phase2 +流量标定", m2), ("Phase3 +组内收缩", m3)]:
    print(f"{label:20s} {m['pass']:2d}/{m['total']}  {m['MAE']:.4f}% {m['u_L']:.4f}% "
          f"{m['u_r']:.4f}% {m['u_d']:.4f}% {five_target_score(m):.2f}")

rows = [{"version": "phys6", **m_phys6},
        {"version": "phase1_multiobj", **m1},
        {"version": "phase2_calib", **m2},
        {"version": "phase3_shrink", **m3}]
pd.DataFrame(rows).to_csv(OUT_DIR / "optimization_results.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
