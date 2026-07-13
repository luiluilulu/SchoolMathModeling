"""
动态声道权重模型：根据剖面特征在线调整五声道积分权重。
w_j(z) = w_j^(0) + Δw_j(z)
约束: ΣΔw_j=0, w_0=w_4, w_1=w_3 (对称)
化为2K个ridge系数，嵌套LODO验证。
输出: output/results/dynamic_weights/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/dynamic_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREA = 0.13138219017128852
W0 = np.array([0.209874, 0.153223, 0.266827, 0.153223, 0.209874])
CHORD_COLS = [f"chord{i}" for i in range(5)]

# 剖面特征（驱动权重调整）
Z_FEATS = [
    "profile_top_bottom", "profile_center_all", "profile_edge_inner",
    "profile_inner_skew", "profile_swirl", "profile_ab_abs",
    "dyn_plateau_cv", "base_rate_m3h",
]

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
chord_mat = df[CHORD_COLS].astype(float).values
df["base_vol"] = AREA * duration * (chord_mat @ W0)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
std_vol = df["standard_volume_m3"].astype(float).values
dates = df["date"].astype(str).values
n_dates = df["date"].astype(str).nunique()


def build_design_matrix(df_sub):
    """构造动态权重ridge设计矩阵。

    推导：
    V_pred = A*T*Σ_j (w_j^(0)+Δw_j)*c_j
    约束 ΣΔw_j=0 → β_2k = -2(β_0k+β_1k)
    对称 w_0=w_4, w_1=w_3 → 只需 β_0k, β_1k

    特征:
    X_{i,2k}   = A*T_i * z_ik * (c_0i+c_4i-2c_2i)  → β_0k
    X_{i,2k+1} = A*T_i * z_ik * (c_1i+c_3i-2c_2i)  → β_1k
    目标: y_i = V_std_i - V_phys6_i
    """
    T = df_sub["duration_s"].astype(float).values
    C = df_sub[CHORD_COLS].astype(float).values
    Z = df_sub[Z_FEATS].astype(float).values
    V_phys6 = AREA * T * (C @ W0)

    # 声道组合（利用对称性）
    c_edge = C[:, 0] + C[:, 4]   # chord_0 + chord_4
    c_inner = C[:, 1] + C[:, 3]  # chord_1 + chord_3
    c_center = C[:, 2]           # chord_2

    # 二阶差分（edge/inner相对center）
    d_edge = c_edge - 2 * c_center   # (c0+c4) - 2*c2
    d_inner = c_inner - 2 * c_center # (c1+c3) - 2*c2

    K = Z.shape[1]
    X = np.zeros((len(df_sub), 2 * K))
    for k in range(K):
        X[:, 2*k]   = AREA * T * Z[:, k] * d_edge
        X[:, 2*k+1] = AREA * T * Z[:, k] * d_inner

    y = df_sub["standard_volume_m3"].astype(float).values - V_phys6
    return X, y, V_phys6


def predict_volume(df_sub, scaler, model):
    """用训练好的ridge模型预测体积。"""
    X, _, V_phys6 = build_design_matrix(df_sub)
    X_s = scaler.transform(X)
    correction = model.predict(X_s)
    return V_phys6 + correction


def ev(err):
    """官方指标。"""
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
    """内层评分。"""
    gp = 0; gs = 0.0; gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3:
            continue
        ee = err[g.index]
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040:
            gp += 1
        gs = max(gs, ee.std(ddof=1))
        gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


# ==== 嵌套 LODO ====
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))
fold_info = []

print(f"动态声道权重 嵌套LODO ({n_dates}折, {len(Z_FEATS)}特征→{2*len(Z_FEATS)}维ridge)", flush=True)

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    best_score = (-1, -np.inf, -np.inf)
    best_alpha = None

    X_tr, y_tr, _ = build_design_matrix(tr_o)

    for alpha in ALPHAS:
        ip = np.zeros(len(tr_o))
        for itr, iva in inner.split(tr_o, groups=idates):
            tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
            X_tri, y_tri, v6_tri = build_design_matrix(tri)
            X_vai, _, v6_vai = build_design_matrix(vai)

            scaler = StandardScaler()
            X_tri_s = scaler.fit_transform(X_tri)
            X_vai_s = scaler.transform(X_vai)

            m = Ridge(alpha=alpha)
            m.fit(X_tri_s, y_tri)
            ip[iva] = v6_vai + m.predict(X_vai_s)

        err = (ip - tr_o["standard_volume_m3"].values) / tr_o["standard_volume_m3"].values * 100.0
        sc = inner_score(err, tr_o)
        if sc > best_score:
            best_score = sc
            best_alpha = alpha

    # 最优alpha在完整外层训练集拟合
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    fm = Ridge(alpha=best_alpha)
    fm.fit(X_tr_s, y_tr)

    # 预测外层
    X_te, _, v6_te = build_design_matrix(te_o)
    X_te_s = scaler.transform(X_te)
    outer_pred[ote] = v6_te + fm.predict(X_te_s)

    # 诊断：修正量相对幅度
    correction = fm.predict(X_te_s)
    rel_corr = np.abs(correction) / (v6_te + 1e-10) * 100
    weight_change_pct = np.abs(correction) / (v6_te + 1e-10) * 100  # 等价于权重相对变化

    test_date = str(te_o["date"].iloc[0])
    fold_info.append({
        "fold": of + 1, "test_date": test_date,
        "alpha": best_alpha, "inner_pass": best_score[0],
        "mean_abs_corr_pct": float(np.mean(weight_change_pct)),
        "max_abs_corr_pct": float(np.max(weight_change_pct)),
    })
    print(f"  折{of+1}/{n_dates} 日期={test_date} α={best_alpha:.1f} "
          f"inner_pass={best_score[0]} "
          f"|corr|={np.mean(weight_change_pct):.3f}% max={np.max(weight_change_pct):.3f}%",
          flush=True)

# 最终评价
final_err = (outer_pred - std_vol) / std_vol * 100.0
r = ev(final_err)
r["model"] = "dynamic_weights"

# Phys6基线对照
e_phys6 = (df["base_vol"].values - std_vol) / std_vol * 100.0
r_phys6 = ev(e_phys6)
r_phys6["model"] = "phys6"

print(f"\n{'='*60}")
print(f"动态声道权重 vs Phys6 基线")
print(f"{'='*60}")
for label, res in [("Phys6", r_phys6), ("DynW", r)]:
    print(f"  {label:6s}: pass={res['pass']:2d}/{res['total']} MAE={res['MAE']:.4f}% "
          f"u_L={res['u_L']:.4f}% u_r={res['u_r']:.4f}% u_d={res['u_d']:.4f}%")

# 保存
pd.DataFrame([r_phys6, r]).to_csv(OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(fold_info).to_csv(OUT_DIR / "fold_details.csv", index=False, encoding="utf-8-sig")

pred_df = df[["window_id", "date", "flow_point", "disturbance_id",
               "standard_volume_m3", "base_vol"]].copy()
pred_df["pred_volume_m3"] = outer_pred
pred_df["error_pct"] = final_err
pred_df.to_csv(OUT_DIR / "window_predictions.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUT_DIR}")
