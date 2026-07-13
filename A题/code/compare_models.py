"""
嵌套LODO模型对比：Phys6 | Ridge | RBF-SVR | ExtraTrees。
统一框架：V_base = AT·Σw·chord, r = log(V_std/V_base), V_hat = V_base·exp(r_hat)。
仅替换残差学习器，全部使用相同嵌套留一日期验证。
输出: output/results/model_compare/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/model_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
AB_COLS = [f"ab{i}" for i in range(5)]
FEATS = CHORD_COLS + AB_COLS + [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner", "profile_inner_skew",
    "profile_ab_abs", "profile_swirl", "dyn_first_0p1_s", "dyn_tail_0p1_s",
    "dyn_start_over_plateau", "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "zero_rate_med", "zero_rate_mad", "zero_age_s", "base_rate_m3h", "duration_s", "flow_point",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()

# ---- 评价函数 ----
def ev(err):
    """完整官方指标。err: (n,) relative error in pct。"""
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


# ---- 模型定义 ----
def make_svr(C, gamma, epsilon):
    return SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon, cache_size=512)

def make_ridge(alpha):
    return Ridge(alpha=alpha)

def make_et(leaf, depth, mf, nest):
    return ExtraTreesRegressor(
        n_estimators=nest, min_samples_leaf=leaf,
        max_depth=depth, max_features=mf,
        random_state=2026, n_jobs=-1,
    )


# ---- 参数网格 ----
# SVR: epsilon根据残差尺度自适应（在折内设置）
SVR_C = [0.1, 1.0, 10.0, 100.0]
SVR_GAMMA = ["scale", 0.01, 0.1, 1.0]
# epsilon在折内根据训练target设定

RIDGE_ALPHA = [0.01, 0.1, 1.0, 10.0, 100.0]

ET_PG = [
    {"leaf": l, "depth": d, "mf": m, "nest": n}
    for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200, 300]
]


# ---- 嵌套 LODO 主循环 ----
def run_nested_lodo(model_name):
    """返回 (outer_pred, outer_err_pct, fold_info)。"""
    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))
    fold_info = []

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote]
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        # Phys6 基线不需要内层选择
        if model_name == "phys6":
            outer_pred[ote] = np.zeros(len(te_o))  # r_hat=0 → V=V_base
            test_date = str(te_o["date"].iloc[0])
            fold_info.append({"fold": of + 1, "test_date": test_date})
            print(f"  Phys6 折{of+1}/{n_dates} 日期={test_date}", flush=True)
            continue

        best_score = (-1, -np.inf, -np.inf)
        best_config = None

        # --- SVR ---
        if model_name == "svr":
            # epsilon候选：基于训练target的百分位数
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
                            tri = tr_o.iloc[itr]
                            vai = tr_o.iloc[iva]
                            Xtr = tri[FEATS].astype(float)
                            Xva = vai[FEATS].astype(float)
                            med = Xtr.median()
                            Xtr_f = Xtr.fillna(med).values
                            Xva_f = Xva.fillna(med).values
                            scaler = StandardScaler()
                            Xtr_s = scaler.fit_transform(Xtr_f)
                            Xva_s = scaler.transform(Xva_f)
                            m = make_svr(C, gamma, eps)
                            m.fit(Xtr_s, tri["target"].values)
                            ip[iva] = m.predict(Xva_s)
                        # 评分
                        vol = tr_o["base_vol"].values * np.exp(ip)
                        err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                        sc = inner_score(err, tr_o)
                        if sc > best_score:
                            best_score = sc
                            best_config = {"C": C, "gamma": gamma, "epsilon": eps}

        # --- Ridge ---
        elif model_name == "ridge":
            for alpha in RIDGE_ALPHA:
                ip = np.zeros(len(tr_o))
                for itr, iva in inner.split(tr_o, groups=idates):
                    tri = tr_o.iloc[itr]
                    vai = tr_o.iloc[iva]
                    Xtr = tri[FEATS].astype(float)
                    Xva = vai[FEATS].astype(float)
                    med = Xtr.median()
                    Xtr_f = Xtr.fillna(med).values
                    Xva_f = Xva.fillna(med).values
                    scaler = StandardScaler()
                    Xtr_s = scaler.fit_transform(Xtr_f)
                    Xva_s = scaler.transform(Xva_f)
                    m = make_ridge(alpha)
                    m.fit(Xtr_s, tri["target"].values)
                    ip[iva] = m.predict(Xva_s)
                vol = tr_o["base_vol"].values * np.exp(ip)
                err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                sc = inner_score(err, tr_o)
                if sc > best_score:
                    best_score = sc
                    best_config = {"alpha": alpha}

        # --- ExtraTrees ---
        elif model_name == "et":
            for p in ET_PG:
                ip = np.zeros(len(tr_o))
                for itr, iva in inner.split(tr_o, groups=idates):
                    tri = tr_o.iloc[itr]
                    vai = tr_o.iloc[iva]
                    Xtr = tri[FEATS].astype(float)
                    Xva = vai[FEATS].astype(float)
                    med = Xtr.median()
                    m = make_et(p["leaf"], p["depth"], p["mf"], p["nest"])
                    m.fit(Xtr.fillna(med).values, tri["target"].values)
                    ip[iva] = m.predict(Xva.fillna(med).values)
                vol = tr_o["base_vol"].values * np.exp(ip)
                err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
                sc = inner_score(err, tr_o)
                if sc > best_score:
                    best_score = sc
                    best_config = p

        # 最优配置在完整外层训练集拟合
        Xall = tr_o[FEATS].astype(float)
        med = Xall.median()
        Xall_f = Xall.fillna(med).values
        Xte_f = te_o[FEATS].astype(float).fillna(med).values

        if model_name == "svr":
            scaler = StandardScaler()
            Xall_s = scaler.fit_transform(Xall_f)
            Xte_s = scaler.transform(Xte_f)
            fm = make_svr(best_config["C"], best_config["gamma"], best_config["epsilon"])
            fm.fit(Xall_s, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_s)
        elif model_name == "ridge":
            scaler = StandardScaler()
            Xall_s = scaler.fit_transform(Xall_f)
            Xte_s = scaler.transform(Xte_f)
            fm = make_ridge(best_config["alpha"])
            fm.fit(Xall_s, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_s)
        elif model_name == "et":
            fm = make_et(best_config["leaf"], best_config["depth"], best_config["mf"], best_config["nest"])
            fm.fit(Xall_f, tr_o["target"].values)
            outer_pred[ote] = fm.predict(Xte_f)

        test_date = str(te_o["date"].iloc[0])
        fold_info.append({"fold": of + 1, "test_date": test_date, **best_config,
                          "inner_pass": best_score[0]})
        print(f"  {model_name} 折{of+1}/{n_dates} 日期={test_date} "
              f"cfg={best_config} inner_pass={best_score[0]}", flush=True)

    # 计算误差
    final_vol = df["base_vol"].values * np.exp(outer_pred)
    final_err = (final_vol - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0
    return outer_pred, final_err, fold_info


# ==== 运行全部模型 ====
results = {}

for name in ["phys6", "ridge", "svr", "et"]:
    print(f"\n{'='*50}")
    print(f"模型: {name}")
    print(f"{'='*50}", flush=True)
    pred, err, folds = run_nested_lodo(name)
    r = ev(err)
    r["model"] = name
    results[name] = {"metrics": r, "folds": folds, "pred": pred, "err": err}
    print(f"  => pass={r['pass']}/{r['total']} MAE={r['MAE']:.4f}% "
          f"u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%", flush=True)


# ==== 汇总对比 ====
print(f"\n{'='*50}")
print("模型对比汇总")
print(f"{'='*50}")
summary_rows = []
for name in ["phys6", "ridge", "svr", "et"]:
    r = results[name]["metrics"]
    summary_rows.append(r)
    print(f"  {name:6s}: pass={r['pass']:2d}/{r['total']} MAE={r['MAE']:.4f}% "
          f"u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}%")

pd.DataFrame(summary_rows).to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

# 保存各模型逐窗口预测
for name in ["phys6", "ridge", "svr", "et"]:
    pred_df = df[[
        "window_id", "date", "flow_point", "disturbance_id",
        "standard_volume_m3", "base_vol",
    ]].copy()
    pred_df["pred_volume_m3"] = df["base_vol"].values * np.exp(results[name]["pred"])
    pred_df["error_pct"] = results[name]["err"]
    pred_df.to_csv(OUT_DIR / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(results[name]["folds"])
    fold_df.to_csv(OUT_DIR / f"folds_{name}.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({name: results[name]["metrics"] for name in results},
              f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUT_DIR}")
