"""
动态声道权重 v2：对称+反对称模态 + 可选全局尺度松弛。
Δw_0 = s_0+a_0, Δw_4 = s_0-a_0  (外侧上下)
Δw_1 = s_1+a_1, Δw_3 = s_1-a_1  (内侧上下)
Δw_2 = -2s_0-2s_1              (中心, 由ΣΔw=0约束)
s_* 由对称特征驱动, a_* 由反对称特征驱动。
岭回归 + 嵌套LODO。三版本对比: S / SA / SA-Scale。
输出: output/results/dynamic_weights_v2/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/dynamic_weights_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W0 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

# 特征分组
Z_SYM = [   # 驱动对称模态 s_0, s_1
    "profile_center_all", "profile_edge_inner", "profile_ab_abs",
    "dyn_plateau_cv", "base_rate_m3h",
]
Z_ASYM = [  # 驱动反对称模态 a_0, a_1
    "profile_top_bottom", "profile_inner_skew", "profile_swirl",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
chord_mat = df[CHORD_COLS].astype(float).values
df["base_vol"] = AREA * duration * (chord_mat @ W0)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()


def build_design(df_sub, version):
    """构造ridge设计矩阵。

    体积修正 = A*T * [s_0*(c_0+c_4-2c_2) + s_1*(c_1+c_3-2c_2)
                      + a_0*(c_0-c_4) + a_1*(c_1-c_3)]
    其中 s_*, a_* 为特征线性组合。
    version: 'S'=仅对称, 'SA'=对称+反对称, 'SA-S'=SA+全局尺度
    """
    T = df_sub["duration_s"].astype(float).values
    C = df_sub[CHORD_COLS].astype(float).values
    V_phys6 = AREA * T * (C @ W0)

    # 声道组合
    c_s0 = C[:, 0] + C[:, 4] - 2 * C[:, 2]   # 外侧相对中心 (对称)
    c_s1 = C[:, 1] + C[:, 3] - 2 * C[:, 2]   # 内侧相对中心 (对称)
    c_a0 = C[:, 0] - C[:, 4]                  # 外侧上下差 (反对称)
    c_a1 = C[:, 1] - C[:, 3]                  # 内侧上下差 (反对称)

    # 对称特征 → s_0, s_1
    Zs = df_sub[Z_SYM].astype(float).values
    X_sym = np.zeros((len(df_sub), 2 * len(Z_SYM)))
    for k in range(len(Z_SYM)):
        X_sym[:, 2*k]   = AREA * T * c_s0 * Zs[:, k]   # → α_k (s_0系数)
        X_sym[:, 2*k+1] = AREA * T * c_s1 * Zs[:, k]   # → β_k (s_1系数)

    if version == "S":
        X = X_sym
    else:
        # 反对称特征 → a_0, a_1
        Za = df_sub[Z_ASYM].astype(float).values
        X_asym = np.zeros((len(df_sub), 2 * len(Z_ASYM)))
        for k in range(len(Z_ASYM)):
            X_asym[:, 2*k]   = AREA * T * c_a0 * Za[:, k]   # → γ_k (a_0系数)
            X_asym[:, 2*k+1] = AREA * T * c_a1 * Za[:, k]   # → δ_k (a_1系数)
        X = np.column_stack([X_sym, X_asym])

    if version == "SA-S":
        # 添加全局尺度特征 (允许∑w_j偏离0.993)
        X_scale = V_phys6.reshape(-1, 1)  # 等价于整体比例修正
        X = np.column_stack([X, X_scale])

    y = df_sub["standard_volume_m3"].astype(float).values - V_phys6
    return X, y, V_phys6


def predict_vol(df_sub, scaler, model):
    X, _, V6 = build_design(df_sub, "SA-S")  # 用最大版本构造(多余列会被scaler/model忽略)
    # 统一: 根据实际X列数截取
    n_feat = len(scaler.mean_)
    X = X[:, :n_feat]
    X_s = scaler.transform(X)
    return V6 + model.predict(X_s)


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
    udc = d1["drift"].max() / math.sqrt(3); udr = d1["std"].fillna(0).max()
    ud = math.sqrt(udc ** 2 + udr ** 2)
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


# ==== 嵌套 LODO（三版本）====
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
VERSIONS = ["S", "SA", "SA-S"]
all_results = {}

for ver in VERSIONS:
    print(f"\n{'='*60}")
    print(f"版本: DynW-{ver}")
    if ver == "S":   print("  对称模态 only (s_0, s_1)")
    if ver == "SA":  print("  对称 + 反对称模态 (s_0, s_1, a_0, a_1)")
    if ver == "SA-S": print("  对称 + 反对称 + 全局尺度松弛")
    print(f"{'='*60}", flush=True)

    outer = LeaveOneGroupOut()
    outer_pred = np.zeros(len(df))
    corr_pcts = []

    for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
        tr_o = df.iloc[otr].reset_index(drop=True)
        te_o = df.iloc[ote]
        idates = tr_o["date"].astype(str).values
        inner = LeaveOneGroupOut()

        X_tr, y_tr, _ = build_design(tr_o, ver)
        n_feat = X_tr.shape[1]
        best_score = (-1, -np.inf, -np.inf)
        best_alpha = None

        for alpha in ALPHAS:
            ip = np.zeros(len(tr_o))
            for itr, iva in inner.split(tr_o, groups=idates):
                tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
                X_tri, y_tri, v6_tri = build_design(tri, ver)
                X_vai, _, v6_vai = build_design(vai, ver)

                scaler = StandardScaler()
                X_tri_s = scaler.fit_transform(X_tri)
                m = Ridge(alpha=alpha)
                m.fit(X_tri_s, y_tri)
                ip[iva] = v6_vai + m.predict(scaler.transform(X_vai))

            err = (ip - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
            sc = inner_score(err, tr_o)
            if sc > best_score:
                best_score = sc
                best_alpha = alpha

        # 最优alpha拟合完整外层训练集
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        fm = Ridge(alpha=best_alpha)
        fm.fit(X_tr_s, y_tr)

        # 预测
        X_te, _, v6_te = build_design(te_o, ver)
        correction = fm.predict(scaler.transform(X_te))
        outer_pred[ote] = v6_te + correction

        rel_corr = np.abs(correction) / (np.abs(v6_te) + 1e-10) * 100
        corr_pcts.append(float(np.mean(rel_corr)))

        test_date = str(te_o["date"].iloc[0])
        print(f"  折{of+1}/{n_dates} 日期={test_date} α={best_alpha:.1f} "
              f"inner_pass={best_score[0]} n_feat={n_feat} "
              f"|corr|={np.mean(rel_corr):.3f}% ||β||={np.linalg.norm(fm.coef_):.3e}",
              flush=True)

    final_err = (outer_pred - std_vol) / std_vol * 100.0
    r = ev(final_err)
    r["version"] = ver
    r["mean_corr_pct"] = float(np.mean(corr_pcts))
    r["corr_sd_pct"] = float(np.std(corr_pcts))
    all_results[ver] = r

    print(f"  => pass={r['pass']}/{r['total']} MAE={r['MAE']:.4f}% "
          f"u_L={r['u_L']:.4f}% u_r={r['u_r']:.4f}% u_d={r['u_d']:.4f}% "
          f"mean|corr|={np.mean(corr_pcts):.3f}%", flush=True)

# ==== 汇总 ====
e_phys6 = (df["base_vol"].values - std_vol) / std_vol * 100.0
r_phys6 = ev(e_phys6)
r_phys6["version"] = "phys6"

print(f"\n{'='*70}")
print(f"动态声道权重 v2 汇总 (嵌套LODO)")
print(f"{'='*70}")
print(f"{'版本':12s} {'通过':7s} {'MAE':8s} {'u_L':8s} {'u_r':8s} {'u_d':8s} {'|corr|':8s}")
print("-" * 70)
for ver in ["phys6", "S", "SA", "SA-S"]:
    r = r_phys6 if ver == "phys6" else all_results[ver]
    corr_str = f"{r.get('mean_corr_pct', 0):.3f}%" if 'mean_corr_pct' in r else "-"
    print(f"{'DynW-'+ver if ver!='phys6' else 'Phys6':12s} "
          f"{r['pass']:2d}/{r['total']}  {r['MAE']:.4f}% {r['u_L']:.4f}% "
          f"{r['u_r']:.4f}% {r['u_d']:.4f}% {corr_str:>8s}")

# 保存
rows = [r_phys6] + [all_results[v] for v in VERSIONS]
pd.DataFrame(rows).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump({v: all_results[v] for v in VERSIONS}, f, ensure_ascii=False, indent=2)
print(f"\n输出: {OUT_DIR}")
