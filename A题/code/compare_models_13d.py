"""
四模型嵌套LODO对比：Ridge | RBF-SVR | GBRT | ExtraTrees。
固定特征集：profile(6) + dynamic(6) + base_rate(1) = 13维。
固定 η=1.0，统一嵌套留一日期验证。
输出: output/results/model_compare_13d/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/model_compare_13d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

FEATS_13 = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_ab_abs", "profile_swirl",
    "dyn_first_0p1_s", "dyn_tail_0p1_s", "dyn_start_over_plateau",
    "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "base_rate_m3h",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()


def ev(err):
    """完整官方指标。"""
    work = df.copy()
    work["ep"] = err
    gd = []
    for _, grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3:
            continue
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
        "u_L": float(ul), "u_r": float(ur), "u_d": float(ud),
    }


def inner_score(err, df_sub):
    """内层评分：(组通过数, -max_SD, -max_|mean|)。"""
    gp = 0
    gs = 0.0
    gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3:
            continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040:
            gp += 1
        gs = max(gs, ee.std(ddof=1))
        gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


# ---- 参数网格 ----
# Ridge
RIDGE_ALPHA = [0.01, 0.1, 1.0, 10.0, 100.0]

# RBF-SVR
SVR_C = [0.1, 1.0, 10.0]
SVR_GAMMA = ["scale", 0.01, 0.1]
# epsilon: 每个折内根据训练target自适应

# GBRT
GBRT_LEAF = [5, 10]
GBRT_DEPTH = [2, 3]
GBRT_LR = [0.03, 0.05]
GBRT_NEST = [100, 200]

# ExtraTrees
ET_LEAF = [3, 5]
ET_DEPTH = [4, 6, None]
ET_MF = [0.3, 0.5]
ET_NEST = [200, 300]


def run_nested_lodo(model_name):
    """返回 (outer_pred, final_err)。"""
    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote]
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        best_score = (-1, -np.inf, -np.inf)
        best_config = None

        # --- Ridge ---
        if model_name == "ridge":
            for alpha in RIDGE_ALPHA:
                ip = np.zeros(len(tr_o))
                for itr, iva in inner.split(tr_o, groups=idates):
                    tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                    Xtr_f = tri[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                    Xva_f = vai[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                    scaler = StandardScaler()
                    m = Ridge(alpha=alpha)
                    m.fit(scaler.fit_transform(Xtr_f), tri["target"].values)
                    ip[iva] = m.predict(scaler.transform(Xva_f))
                vol = tr_o["base_vol"].values * np.exp(ip)
                err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                sc = inner_score(err, tr_o)
                if sc > best_score:
                    best_score = sc
                    best_config = {"alpha": alpha}

        # --- RBF-SVR ---
        elif model_name == "svr":
            t_abs = tr_o["target"].abs()
            eps_candidates = sorted(set([
                float(np.percentile(t_abs, 10)),
                float(np.percentile(t_abs, 25)),
                float(np.percentile(t_abs, 50)),
            ]))
            for C in SVR_C:
                for gamma in SVR_GAMMA:
                    for eps in eps_candidates:
                        ip = np.zeros(len(tr_o))
                        for itr, iva in inner.split(tr_o, groups=idates):
                            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                            Xtr_f = tri[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                            Xva_f = vai[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                            scaler = StandardScaler()
                            m = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps, cache_size=512)
                            m.fit(scaler.fit_transform(Xtr_f), tri["target"].values)
                            ip[iva] = m.predict(scaler.transform(Xva_f))
                        vol = tr_o["base_vol"].values * np.exp(ip)
                        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                        sc = inner_score(err, tr_o)
                        if sc > best_score:
                            best_score = sc
                            best_config = {"C": C, "gamma": gamma, "epsilon": eps}

        # --- GBRT ---
        elif model_name == "gbrt":
            for lr in GBRT_LR:
                for leaf in GBRT_LEAF:
                    for depth in GBRT_DEPTH:
                        for nest in GBRT_NEST:
                            ip = np.zeros(len(tr_o))
                            for itr, iva in inner.split(tr_o, groups=idates):
                                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                                Xtr_f = tri[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                                Xva_f = vai[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                                m = GradientBoostingRegressor(
                                    n_estimators=nest, learning_rate=lr,
                                    max_depth=depth, min_samples_leaf=leaf,
                                    loss="huber", random_state=2026,
                                )
                                m.fit(Xtr_f, tri["target"].values)
                                ip[iva] = m.predict(Xva_f)
                            vol = tr_o["base_vol"].values * np.exp(ip)
                            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                            sc = inner_score(err, tr_o)
                            if sc > best_score:
                                best_score = sc
                                best_config = {"lr": lr, "leaf": leaf, "depth": depth, "nest": nest}

        # --- ExtraTrees ---
        elif model_name == "et":
            for leaf in ET_LEAF:
                for depth in ET_DEPTH:
                    for mf in ET_MF:
                        for nest in ET_NEST:
                            ip = np.zeros(len(tr_o))
                            for itr, iva in inner.split(tr_o, groups=idates):
                                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                                Xtr_f = tri[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                                Xva_f = vai[FEATS_13].astype(float).fillna(tri[FEATS_13].median()).values
                                m = ExtraTreesRegressor(
                                    n_estimators=nest, min_samples_leaf=leaf,
                                    max_depth=depth, max_features=mf,
                                    random_state=2026, n_jobs=-1,
                                )
                                m.fit(Xtr_f, tri["target"].values)
                                ip[iva] = m.predict(Xva_f)
                            vol = tr_o["base_vol"].values * np.exp(ip)
                            err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                            sc = inner_score(err, tr_o)
                            if sc > best_score:
                                best_score = sc
                                best_config = {"leaf": leaf, "depth": depth, "mf": mf, "nest": nest}

        # 最优参数拟合外层训练集
        Xall_f = tr_o[FEATS_13].astype(float).fillna(tr_o[FEATS_13].median()).values
        Xte_f = te_o[FEATS_13].astype(float).fillna(tr_o[FEATS_13].median()).values

        if model_name in ("ridge", "svr"):
            scaler = StandardScaler()
            Xall_s = scaler.fit_transform(Xall_f)
            Xte_s = scaler.transform(Xte_f)
            if model_name == "ridge":
                fm = Ridge(alpha=best_config["alpha"])
            else:
                fm = SVR(kernel="rbf", C=best_config["C"], gamma=best_config["gamma"],
                         epsilon=best_config["epsilon"], cache_size=512)
            fm.fit(Xall_s, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_s)
        elif model_name == "gbrt":
            fm = GradientBoostingRegressor(
                n_estimators=best_config["nest"], learning_rate=best_config["lr"],
                max_depth=best_config["depth"], min_samples_leaf=best_config["leaf"],
                loss="huber", random_state=2026 + of,
            )
            fm.fit(Xall_f, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_f)
        elif model_name == "et":
            fm = ExtraTreesRegressor(
                n_estimators=best_config["nest"], min_samples_leaf=best_config["leaf"],
                max_depth=best_config["depth"], max_features=best_config["mf"],
                random_state=2026 + of, n_jobs=-1,
            )
            fm.fit(Xall_f, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_f)

        test_date = str(te_o["date"].iloc[0])
        print(f"  {model_name} 折{of+1}/{n_dates} 日期={test_date} "
              f"cfg={best_config} inner_pass={best_score[0]}", flush=True)

    final_vol = df["base_vol"].values * np.exp(outer_pred)
    final_err = (final_vol - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
    return outer_pred, final_err


# ==== 运行 ====
results = {}

for name in ["ridge", "svr", "gbrt", "et"]:
    print(f"\n{'='*50}")
    print(f"模型: {name} (13维特征)")
    print(f"{'='*50}", flush=True)
    pred, err = run_nested_lodo(name)
    r = ev(err)
    r["model"] = name
    results[name] = {"metrics": r, "pred": pred, "err": err}
    print(f"  => pass={r['pass']}/{r['total']} MAE={r['MAE']:.4f}% "
          f"u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%", flush=True)

# Phys6基线
v_phys6 = df["base_vol"].values
e_phys6 = (v_phys6 - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
r_phys6 = ev(e_phys6)
r_phys6["model"] = "phys6"
results["phys6"] = {"metrics": r_phys6, "err": e_phys6}


# ==== 汇总 ====
print(f"\n{'='*60}")
print(f"13维特征集四模型对比 (嵌套LODO, η=1.0)")
print(f"{'='*60}")
print(f"{'模型':12s} {'通过':7s} {'MAE':8s} {'u_L':8s} {'u_r':8s} {'u_d':8s}")
print("-" * 60)
for name in ["phys6", "ridge", "svr", "gbrt", "et"]:
    r = results[name]["metrics"]
    print(f"{name:12s} {r['pass']:2d}/{r['total']}  {r['MAE']:.4f}% {r['u_L']:.4f}% {r['u_r']:.4f}% {r['u_d']:.4f}%")

# 保存
summary_rows = [results[name]["metrics"] for name in ["phys6", "ridge", "svr", "gbrt", "et"]]
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "model_comparison_13d.csv", index=False, encoding="utf-8-sig")

for name in ["ridge", "svr", "gbrt", "et"]:
    pred_df = df[["window_id", "date", "flow_point", "disturbance_id",
                   "standard_volume_m3", "base_vol"]].copy()
    pred_df["pred_volume_m3"] = df["base_vol"].values * np.exp(results[name]["pred"])
    pred_df["error_pct"] = results[name]["err"]
    pred_df.to_csv(OUT_DIR / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({name: results[name]["metrics"] for name in results},
              f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUT_DIR}")
