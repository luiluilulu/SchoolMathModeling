"""
特征消融实验：逐组添加特征，ExtraTrees 嵌套 LODO。
输出: output/results/ablation/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from utils.metrics import inner_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
PROFILE = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
]
DYNAMIC = [
    "dyn_first_0p1_s", "dyn_tail_0p1_s", "dyn_start_over_plateau",
    "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
]
ZERO = ["zero_rate_med", "zero_rate_mad", "zero_age_s"]
AUX = ["base_rate_m3h", "duration_s", "flow_point"]

# 特征组合定义
FEATURE_SETS = {
    "profile+rate":        PROFILE + ["base_rate_m3h"],
    "profile+AB+rate":     PROFILE + AB_COLS + ["base_rate_m3h"],
    "profile+dyn+rate":    PROFILE + DYNAMIC + ["base_rate_m3h"],
    "profile+zero+rate":   PROFILE + ZERO + ["base_rate_m3h"],
    "profile+AB+dyn+rate": PROFILE + AB_COLS + DYNAMIC + ["base_rate_m3h"],
    "all_28":              CHORD_COLS + AB_COLS + PROFILE + DYNAMIC + ZERO + AUX,
    "all_no_chord":        AB_COLS + PROFILE + DYNAMIC + ZERO + AUX,
    "all_no_flow_point":   CHORD_COLS + AB_COLS + PROFILE + DYNAMIC + ZERO + ["base_rate_m3h", "duration_s"],
    "profile_only":        PROFILE,
}

PG = [
    {"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
    for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200, 300]
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()


def ev(err):
    """完整官方指标，含单扰流/混合扰流分层统计。"""
    work = df.copy()
    work["ep"] = err

    # 预计算每组扰动类型数
    dist_counts = df.groupby(["date", "flow_point"])["disturbance_id"].nunique()

    gd = []
    for (dt, fp), grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3:
            continue
        ee = grp["ep"].values
        dc = dist_counts.get((dt, fp), 1)
        gd.append({"m": ee.mean(), "s": ee.std(ddof=1), "dist_count": dc})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0

    single_mask = gdf["dist_count"] == 1
    mixed_mask = gdf["dist_count"] > 1
    single_pass = int(((gdf.loc[single_mask, "m"].abs() <= 0.2) &
                        (gdf.loc[single_mask, "s"] <= 0.040)).sum()) if not gdf.empty else 0
    mixed_pass = int(((gdf.loc[mixed_mask, "m"].abs() <= 0.2) &
                       (gdf.loc[mixed_mask, "s"] <= 0.040)).sum()) if not gdf.empty else 0
    n_single = int(single_mask.sum())
    n_mixed = int(mixed_mask.sum())

    d0 = work[work["disturbance_id"] == "D0"]
    d0m = d0.groupby("flow_point")["ep"].mean()
    ul = math.sqrt((d0m ** 2).sum() / max(len(d0m) - 1, 1))
    ur = gdf["s"].max() if not gdf.empty else 0.0
    use = work[work["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")].groupby("flow_point")["ep"].mean()
    d1 = use[use["condition_note"].eq("disturbed_test")].groupby(
        ["disturbance_id", "flow_point"]
    )["ep"].agg(["mean", "std"]).reset_index()
    d1["bm"] = d1["flow_point"].map(bm)
    d1 = d1.dropna(subset=["bm"])
    d1["drift"] = (d1["bm"] - d1["mean"]).abs()
    udc = d1["drift"].max() / math.sqrt(3)
    udr = d1["std"].fillna(0).max()
    ud = math.sqrt(udc ** 2 + udr ** 2)
    return {
        "MAE": float(np.abs(err).mean()), "pass": gp, "total": len(gdf),
        "pass_single": single_pass, "n_single": n_single,
        "pass_mixed": mixed_pass, "n_mixed": n_mixed,
        "u_L": float(ul), "u_r": float(ur), "u_d": float(ud),
        "max_SD": float(ur),
    }





def run_ablation(name, feat_cols):
    """对给定特征集运行嵌套 LODO，返回预测误差数组。"""
    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote]
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        best_score = (-1, -np.inf, -np.inf)
        best_p = None

        for p in PG:
            ip = np.zeros(len(tr_o))
            for itr, iva in inner.split(tr_o, groups=idates):
                tri = tr_o.iloc[itr]
                vai = tr_o.iloc[iva]
                Xtr = tri[feat_cols].astype(float)
                Xva = vai[feat_cols].astype(float)
                med = Xtr.median()
                m = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
                m.fit(Xtr.fillna(med).values, tri["target"].values)
                ip[iva] = m.predict(Xva.fillna(med).values)
            vol = tr_o["base_vol"].values * np.exp(ip)
            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
            sc = inner_score(err, tr_o)
            if sc > best_score:
                best_score = sc
                best_p = p

        Xall = tr_o[feat_cols].astype(float)
        med = Xall.median()
        fm = ExtraTreesRegressor(**best_p, random_state=2026 + of, n_jobs=-1)
        fm.fit(Xall.fillna(med).values, tr_o["target"].values)
        Xte = te_o[feat_cols].astype(float).fillna(med).values
        outer_pred[ote] = fm.predict(Xte)

    final_vol = df["base_vol"].values * np.exp(outer_pred)
    final_err = (final_vol - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
    return final_err


# ==== 主循环 ====
print(f"特征消融实验 ({len(FEATURE_SETS)} 组特征, {len(PG)} 组超参, {n_dates} 折)", flush=True)
results = []

for name, feats in FEATURE_SETS.items():
    print(f"\n--- {name} ({len(feats)}维) ---", flush=True)
    err = run_ablation(name, feats)
    r = ev(err)
    r["features"] = name
    r["n_features"] = len(feats)
    results.append(r)
    print(f"  pass={r['pass']}/{r['total']} (single={r['pass_single']}/{r['n_single']} "
          f"mixed={r['pass_mixed']}/{r['n_mixed']}) "
          f"MAE={r['MAE']:.4f}% u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%",
          flush=True)

# ==== 汇总 ====
print(f"\n{'='*80}")
header = f"{'特征集':25s} {'维':>3s} {'通过':>7s} {'单扰':>6s} {'混合':>6s} {'MAE':>8s} {'u_L':>8s} {'u_r':>8s} {'u_d':>8s}"
print(header)
print("-" * 80)
for r in sorted(results, key=lambda x: x["pass"], reverse=True):
    print(f"{r['features']:25s} {r['n_features']:3d} "
          f"{r['pass']:2d}/{r['total']}  "
          f"{r['pass_single']:2d}/{r['n_single']:2d}  "
          f"{r['pass_mixed']:2d}/{r['n_mixed']:2d}  "
          f"{r['MAE']:.4f}% {r['u_L']:.4f}% {r['u_r']:.4f}% {r['u_d']:.4f}%")

pd.DataFrame(results).to_csv(OUT_DIR / "ablation_results.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
