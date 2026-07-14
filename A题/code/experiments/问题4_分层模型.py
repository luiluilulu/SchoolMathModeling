"""
问题4分层模型：D0(OWICS+流量校准) + 扰流(Phys6+A/B软门控平滑补偿) + 可选ET残差。
M0: Phys6+ET基线  M1: D0校准+ET  M2: D0校准+A/B补偿  M3: M2+收缩ET。
嵌套LODO验证，优化目标：最大超标倍数优先。
输出: output/results/problem4_layered/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/problem4_layered"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W_OWICS = np.array([0.221205, 0.112176, 0.333238, 0.112176, 0.221205])
W_PHYS6 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]
FEATS_28 = CHORD_COLS + [f"ab{i}" for i in range(5)] + [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner", "profile_inner_skew",
    "profile_ab_abs", "profile_swirl", "dyn_first_0p1_s", "dyn_tail_0p1_s",
    "dyn_start_over_plateau", "dyn_end_over_plateau", "dyn_plateau_cv", "dyn_active_eq_s",
    "zero_rate_med", "zero_rate_mad", "zero_age_s", "base_rate_m3h", "duration_s", "flow_point",
]
# 问题3聚类结果
CLUSTER_MAP = {"D1": "A", "D2": "A", "D5": "A", "D7": "A",
               "D3": "B", "D4": "B", "D6": "B", "D8": "B"}

# 问题3在线分类特征
FEATS_CLASSIFY = [
    "norm_chord0", "norm_chord1", "norm_chord2", "norm_chord3", "norm_chord4",
    "ab0", "ab1", "ab2", "ab3", "ab4", "profile_swirl", "profile_ab_abs",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
chord_mat = df[CHORD_COLS].astype(float).values
df["base_vol_owics"] = AREA * duration * (chord_mat @ W_OWICS)
df["base_vol_phys6"] = AREA * duration * (chord_mat @ W_PHYS6)
df["base_rate_m3h"] = df["base_vol_phys6"] / duration * 3600.0
# 归一化chord（用于分类）
chord_sum = chord_mat.sum(axis=1)
for j in range(5):
    df[f"norm_chord{j}"] = chord_mat[:, j] / np.maximum(chord_sum, 1e-12)
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()
dist_counts = df.groupby(["date", "flow_point"])["disturbance_id"].nunique()


def ev(err, df_sub=None):
    """完整官方指标。df_sub为子集DataFrame，默认为全局df。"""
    work = (df if df_sub is None else df_sub).copy()
    work["ep"] = err
    gd = []
    for _, grp in work.groupby(["date", "flow_point"]):
        if len(grp) < 3: continue
        ee = grp["ep"].values
        gd.append({"m": ee.mean(), "s": ee.std(ddof=1)})
    gdf = pd.DataFrame(gd)
    gp = int(((gdf["m"].abs() <= 0.2) & (gdf["s"] <= 0.040)).sum()) if not gdf.empty else 0
    d0 = work[work["disturbance_id"] == "D0"]
    d0m = d0.groupby("flow_point")["ep"].mean() if len(d0) > 0 else pd.Series(dtype=float)
    ul = math.sqrt((d0m ** 2).sum() / max(len(d0m) - 1, 1)) if len(d0m) > 1 else np.nan
    ur = gdf["s"].max() if not gdf.empty else 0.0
    use = work[work["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")]
    dt = use[use["condition_note"].eq("disturbed_test")]
    if len(bm) > 0 and len(dt) > 0:
        bm_mean = bm.groupby("flow_point")["ep"].mean()
        d1 = dt.groupby(["disturbance_id", "flow_point"])["ep"].agg(["mean", "std"]).reset_index()
        d1["bm"] = d1["flow_point"].map(bm_mean); d1 = d1.dropna(subset=["bm"])
        d1["drift"] = (d1["bm"] - d1["mean"]).abs()
        udc = d1["drift"].max() / math.sqrt(3); udr = d1["std"].fillna(0).max()
        ud = math.sqrt(udc ** 2 + udr ** 2)
    else:
        ud = np.nan
    mae = float(np.abs(err).mean())
    return {"pass": gp, "total": len(gdf), "u_L": float(ul) if np.isfinite(ul) else 0.0,
            "u_r": float(ur), "u_d": float(ud) if np.isfinite(ud) else 0.0, "MAE": mae}


def five_target_score(m):
    """越小越好。max超标倍数优先，再归一化平方和。"""
    rl = m["u_L"] / 0.036; rr = m["u_r"] / 0.040; rd = m["u_d"] / 0.115
    return (max(rl, rr, rd), rl**2 + rr**2 + rd**2, -m["pass"], m["MAE"])


def inner_score(err, df_sub):
    gp = 0; gs = 0.0; gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3: continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040: gp += 1
        gs = max(gs, ee.std(ddof=1)); gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


def fit_d0_calibration(df_train):
    """D0流量校准：用LOFO选择k0或(a,b)。返回最优校准函数。"""
    d0_data = df_train[df_train["disturbance_id"] == "D0"]
    if len(d0_data) < 4:
        return {"method": "none"}
    fps = d0_data["flow_point"].unique()
    if len(fps) < 3:
        return {"method": "k0"}  # 不够做LOFO，用k0

    # LOFO评估两种方案
    best_method = "k0"
    best_ul = np.inf
    for method in ["k0", "linear"]:
        preds = []
        truths = []
        for fp in fps:
            train_fp = d0_data[d0_data["flow_point"] != fp]
            test_fp = d0_data[d0_data["flow_point"] == fp]
            target = np.log(test_fp["standard_volume_m3"].values
                           / test_fp["base_vol_owics"].values)
            if method == "k0":
                k0 = np.exp(np.mean(np.log(
                    train_fp["standard_volume_m3"].values
                    / train_fp["base_vol_owics"].values)))
                pred = np.full(len(test_fp), np.log(k0))
            else:
                z = (train_fp["flow_point"].values - 50) / 30
                y = np.log(train_fp["standard_volume_m3"].values
                           / train_fp["base_vol_owics"].values)
                a, b = np.polyfit(z, y, 1)
                z_test = (test_fp["flow_point"].values - 50) / 30
                pred = a * z_test + b
            preds.append(pred)
            truths.append(target)
        all_pred = np.concatenate(preds); all_true = np.concatenate(truths)
        err = (np.exp(all_pred) - np.exp(all_true)) / np.exp(all_true) * 100
        # u_L from LOFO predictions on D0
        work = d0_data.copy()
        work["ep"] = np.concatenate([
            (np.exp(p) - np.exp(t)) / np.exp(t) * 100
            for p, t in zip(preds, truths)
        ])
        fp_means = work.groupby("flow_point")["ep"].mean()
        ul = math.sqrt((fp_means ** 2).sum() / max(len(fp_means) - 1, 1))
        if ul < best_ul:
            best_ul = ul
            best_method = method

    if best_method == "k0":
        k0 = np.exp(np.mean(np.log(
            d0_data["standard_volume_m3"].values / d0_data["base_vol_owics"].values)))
        return {"method": "k0", "k0": k0}
    else:
        z = (d0_data["flow_point"].values - 50) / 30
        y = np.log(d0_data["standard_volume_m3"].values / d0_data["base_vol_owics"].values)
        a, b = np.polyfit(z, y, 1)
        return {"method": "linear", "a": a, "b": b}


def apply_d0_calibration(df_sub, cal):
    """对D0窗口应用校准。"""
    v_owics = df_sub["base_vol_owics"].values
    if cal["method"] == "none":
        return v_owics
    elif cal["method"] == "k0":
        return v_owics * cal["k0"]
    else:
        z = (df_sub["flow_point"].values - 50) / 30
        return v_owics * np.exp(cal["a"] * z + cal["b"])


def fit_classifier_per_fold(train_df):
    """在训练集上拟合问题3的在线A/B分类器。返回(soft_proba_fn, cluster_swap)。"""
    d0_tr = train_df[train_df["disturbance_id"] == "D0"]
    if len(d0_tr) == 0:
        return None, False

    # 构建12维特征
    X = train_df[FEATS_CLASSIFY].astype(float).values
    d0_mask = train_df["disturbance_id"] == "D0"
    mu = X[d0_mask].mean(axis=0)
    sig = X[d0_mask].std(axis=0) + 1e-12
    X_scaled = (X - mu) / sig

    from sklearn.decomposition import PCA
    pca = PCA(0.90).fit(X_scaled)
    X_pc = pca.transform(X_scaled)

    # A/B中心
    centers = {}
    for label in ["A", "B"]:
        mask_label = train_df["disturbance_id"].map(CLUSTER_MAP) == label
        if mask_label.sum() > 0:
            centers[label] = X_pc[mask_label.values].mean(axis=0)

    # 标签对齐
    tb_a = train_df[train_df["disturbance_id"].map(CLUSTER_MAP) == "A"]["profile_top_bottom"].mean()
    tb_b = train_df[train_df["disturbance_id"].map(CLUSTER_MAP) == "B"]["profile_top_bottom"].mean()
    swap = tb_b > tb_a

    def proba_fn(df_sub):
        X_sub = df_sub[FEATS_CLASSIFY].astype(float).values
        X_sub_s = (X_sub - mu) / sig
        X_sub_pc = pca.transform(X_sub_s)
        d_a = np.sqrt(((X_sub_pc - centers["A"]) ** 2).sum(axis=1))
        d_b = np.sqrt(((X_sub_pc - centers["B"]) ** 2).sum(axis=1))
        if swap:
            d_a, d_b = d_b, d_a
        ea = np.exp(-d_a**2 / 2); eb = np.exp(-d_b**2 / 2)
        p_a = ea / (ea + eb + 1e-12)
        return p_a, 1 - p_a
    return proba_fn, swap


def run_model(name, config):
    """嵌套LODO运行分层模型。config控制各组件开关。"""
    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))
    fold_info = []

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote].reset_index(drop=True)

        # D0校准
        d0_cal = fit_d0_calibration(tr_o) if config.get("d0_cal", False) else {"method": "none"}

        # A/B分类器
        proba_fn = None
        if config.get("ab_gate", False):
            proba_fn, _ = fit_classifier_per_fold(tr_o)

        # A/B流量平滑补偿参数（内层LOFO选择）
        ab_params = {"g0_a": 0, "g0_b": 0, "gA_a": 0, "gA_b": 0, "gB_a": 0, "gB_b": 0}
        if config.get("ab_comp", False) and proba_fn is not None:
            # 扰流训练集：Phys6基线残差
            dist_tr = tr_o[tr_o["disturbance_id"] != "D0"]
            if len(dist_tr) >= 6:
                z = (dist_tr["flow_point"].values - 50) / 30
                y = np.log(dist_tr["standard_volume_m3"].values
                          / dist_tr["base_vol_phys6"].values)
                p_a, p_b = proba_fn(dist_tr)
                # g0 + pA*gA + pB*gB，其中gB = -gA*gB_ratio避免共线
                # 简化：分别拟合A和B
                mask_a = dist_tr["disturbance_id"].map(CLUSTER_MAP) == "A"
                mask_b = dist_tr["disturbance_id"].map(CLUSTER_MAP) == "B"
                if mask_a.sum() >= 3:
                    a_a, b_a = np.polyfit(z[mask_a], y[mask_a], 1)
                    ab_params["gA_a"] = a_a; ab_params["gA_b"] = b_a
                if mask_b.sum() >= 3:
                    a_b, b_b = np.polyfit(z[mask_b], y[mask_b], 1)
                    ab_params["gB_a"] = a_b; ab_params["gB_b"] = b_b
                # 公共趋势来自全部扰流
                a0, b0 = np.polyfit(z, y, 1)
                ab_params["g0_a"] = a0; ab_params["g0_b"] = b0

        # 对训练集生成预测以评估内层指标
        tr_pred = np.zeros(len(tr_o))
        # D0
        d0_mask_tr = tr_o["disturbance_id"] == "D0"
        if d0_mask_tr.sum() > 0:
            tr_pred[d0_mask_tr] = apply_d0_calibration(tr_o[d0_mask_tr], d0_cal)
        # 扰流
        dist_mask_tr = tr_o["disturbance_id"] != "D0"
        if dist_mask_tr.sum() > 0:
            v_phys6 = tr_o.loc[dist_mask_tr, "base_vol_phys6"].values
            z_dist = (tr_o.loc[dist_mask_tr, "flow_point"].values - 50) / 30
            if config.get("ab_comp", False) and proba_fn is not None:
                pa, pb = proba_fn(tr_o[dist_mask_tr])
                ga = ab_params["gA_a"] * z_dist + ab_params["gA_b"]
                gb = ab_params["gB_a"] * z_dist + ab_params["gB_b"]
                g0 = ab_params["g0_a"] * z_dist + ab_params["g0_b"]
                correction = g0 + pa * ga + pb * gb
                tr_pred[dist_mask_tr] = v_phys6 * np.exp(correction)
            else:
                tr_pred[dist_mask_tr] = v_phys6

        # ET残差（可选）
        if config.get("et_residual", False):
            et_best_p = None; et_best_score = (np.inf, np.inf, -np.inf, np.inf)
            ET_PG = [{"min_samples_leaf": l, "max_depth": d, "max_features": m, "n_estimators": n}
                     for l in [3, 5] for d in [4, 6, None] for m in [0.3, 0.5] for n in [200]]
            idates = tr_o["date"].astype(str).values
            inner = LeaveOneGroupOut()
            r_current = np.log(tr_o["standard_volume_m3"].values / tr_pred.clip(1e-10))
            # ET选参用已验证有效的inner_score
            et_best_is = (-1, -np.inf, -np.inf)
            for p in ET_PG:
                ip = np.zeros(len(tr_o))
                for itr, iva in inner.split(tr_o, groups=idates):
                    tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                    Xtr = tri[FEATS_28].astype(float).fillna(tri[FEATS_28].median()).values
                    Xva = vai[FEATS_28].astype(float).fillna(tri[FEATS_28].median()).values
                    et = ExtraTreesRegressor(**p, random_state=2026, n_jobs=-1)
                    et.fit(Xtr, r_current[itr])
                    ip[iva] = et.predict(Xva)
                vol = tr_pred * np.exp(ip)
                err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
                iscore = inner_score(err, tr_o)
                if iscore > et_best_is:
                    et_best_is = iscore
                    et_best_p = p
            # η选择用五目标评分
            ip_best = np.zeros(len(tr_o))
            for itr, iva in inner.split(tr_o, groups=idates):
                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                Xtr = tri[FEATS_28].astype(float).fillna(tri[FEATS_28].median()).values
                Xva = vai[FEATS_28].astype(float).fillna(tri[FEATS_28].median()).values
                et = ExtraTreesRegressor(**et_best_p, random_state=2026, n_jobs=-1)
                et.fit(Xtr, r_current[itr])
                ip_best[iva] = et.predict(Xva)
            for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
                vol = tr_pred * np.exp(ip_best * eta)
                err = (vol - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100
                m = ev(err, tr_o)
                sc = five_target_score(m)
                if sc < et_best_score:
                    et_best_score = sc
                    et_best_eta = eta
            # 最终拟合
            Xall = tr_o[FEATS_28].astype(float).fillna(tr_o[FEATS_28].median()).values
            et_final = ExtraTreesRegressor(**et_best_p, random_state=2026 + of, n_jobs=-1)
            et_final.fit(Xall, r_current)
            Xte = te_o[FEATS_28].astype(float).fillna(tr_o[FEATS_28].astype(float).median()).values
            r_te = et_final.predict(Xte)
        else:
            et_best_eta = 0.0
            r_te = np.zeros(len(te_o))

        # 对外层测试集预测
        te_pred = np.zeros(len(te_o))
        d0_mask_te = te_o["disturbance_id"] == "D0"
        dist_mask_te = te_o["disturbance_id"] != "D0"
        if d0_mask_te.sum() > 0:
            te_pred[d0_mask_te] = apply_d0_calibration(te_o[d0_mask_te], d0_cal)
        if dist_mask_te.sum() > 0:
            v_phys6_te = te_o.loc[dist_mask_te, "base_vol_phys6"].values
            z_te = (te_o.loc[dist_mask_te, "flow_point"].values - 50) / 30
            if config.get("ab_comp", False) and proba_fn is not None:
                pa_te, pb_te = proba_fn(te_o[dist_mask_te])
                ga = ab_params["gA_a"] * z_te + ab_params["gA_b"]
                gb = ab_params["gB_a"] * z_te + ab_params["gB_b"]
                g0 = ab_params["g0_a"] * z_te + ab_params["g0_b"]
                te_pred[dist_mask_te] = v_phys6_te * np.exp(g0 + pa_te * ga + pb_te * gb)
            else:
                te_pred[dist_mask_te] = v_phys6_te
        outer_pred[ote] = te_pred * np.exp(r_te * et_best_eta)

        test_date = str(df.iloc[ote[0]]["date"])
        print(f"  {name} 折{of+1}/{n_dates} 日期={test_date} "
              f"d0_cal={d0_cal.get('method','none')} "
              f"eta={et_best_eta if config.get('et_residual') else 'N/A'}",
              flush=True)

    final_err = (outer_pred - std_vol) / std_vol * 100
    return ev(final_err), outer_pred


# ==== 运行四模型 ====
configs = {
    "M0_baseline": {"d0_cal": False, "ab_comp": False, "ab_gate": False, "et_residual": True},
    "M1_d0cal":    {"d0_cal": True,  "ab_comp": False, "ab_gate": False, "et_residual": True},
    "M2_abcomp":   {"d0_cal": True,  "ab_comp": True,  "ab_gate": True,  "et_residual": False},
    "M3_full":     {"d0_cal": True,  "ab_comp": True,  "ab_gate": True,  "et_residual": True},
}

all_results = {}
for name, cfg in configs.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}", flush=True)
    m, pred = run_model(name, cfg)
    all_results[name] = m
    sc = five_target_score(m)
    print(f"  => pass={m['pass']}/{m['total']} MAE={m['MAE']:.4f}% "
          f"u_L={m['u_L']:.4f}% u_r={m['u_r']:.4f}% u_d={m['u_d']:.4f}% "
          f"max超标={sc[0]:.1f}x sum²={sc[1]:.1f}", flush=True)

# Phys6 + OWICS 基线
e_p = (df["base_vol_phys6"].values - std_vol) / std_vol * 100
e_o = (df["base_vol_owics"].values - std_vol) / std_vol * 100
m_phys6 = ev(e_p); m_owics = ev(e_o)

print(f"\n{'='*65}")
print(f"分层模型对比")
print(f"{'='*65}")
print(f"{'模型':15s} {'Pass':>7s} {'MAE':>8s} {'u_L':>8s} {'u_r':>8s} {'u_d':>8s} {'max超标':>7s}")
print("-" * 65)
for label, m in [("Phys6", m_phys6), ("OWICS", m_owics)] + list(all_results.items()):
    sc = five_target_score(m)
    print(f"{label:15s} {m['pass']:2d}/{m['total']}  {m['MAE']:.4f}% {m['u_L']:.4f}% "
          f"{m['u_r']:.4f}% {m['u_d']:.4f}% {sc[0]:.1f}x")

rows = [{"model": "phys6", **m_phys6}, {"model": "owics", **m_owics}]
for name, m in all_results.items():
    rows.append({"model": name, **m})
pd.DataFrame(rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({k: all_results[k] for k in all_results}, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
