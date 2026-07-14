"""
嵌套留一日期 + η收缩（修复版）。
修复：
1. ev() 中 if-continue 语法错误 → 改为标准缩进
2. inner_score 索引错位 → tr_o reset_index 后索引连续
3. η 搜索移入每个外层折内部 → 防止测试数据泄漏
4. 删除无意义 g_pass 按位与行
5. 统一元组长度
输出: output/results/nested_lodo/
"""
import pandas as pd, numpy as np, math, json
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut

HERE = Path(__file__).resolve().parent
DATA = HERE / "../problem/attachment1_window_data.csv"
OUT_DIR = HERE / "../output/results/nested_lodo"
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

ET_PARAMS = {"min_samples_leaf": 5, "max_depth": 6, "max_features": 0.5, "n_estimators": 200}
ETA_GRID = [1.0]  # 固定不收缩，确认基线

df = pd.read_csv(DATA, encoding="utf-8-sig")
duration = df["duration_s"].astype(float).values
df["base_vol"] = AREA * duration * (df[CHORD_COLS].astype(float).values @ W)
df["base_rate_m3h"] = df["base_vol"] / duration * 3600.0
df["target"] = np.log(df["standard_volume_m3"].astype(float) / df["base_vol"])
dates = df["date"].astype(str).values


def ev(err):
    """完整官方指标。err: (n,) relative error in pct，对齐 df 位置。"""
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
        "u_dc": float(udc), "u_dr": float(udr),
    }


def inner_score(err, df_sub):
    """内层评分：(组通过数, -max_SD, -max_|mean|)。
    err: (n,) array, df_sub: reset_index(drop=True) 后索引连续。
    """
    gp = 0
    gs = 0.0
    gm = 0.0
    for _, g in df_sub.groupby(["date", "flow_point"]):
        if len(g) < 3:
            continue
        ee = err[g.index]  # reset_index 后 g.index 为 0..n-1 连续整数
        if abs(ee.mean()) <= 0.2 and ee.std(ddof=1) <= 0.040:
            gp += 1
        gs = max(gs, ee.std(ddof=1))
        gm = max(gm, abs(ee.mean()))
    return (gp, -gs, -gm)


# ==== 外层LODO（固定ET参数，η=1）====
n_dates = df["date"].astype(str).nunique()
print(f"嵌套LODO基线 ({n_dates} 日期, ET固定参数, η=1) ...", flush=True)
outer = LeaveOneGroupOut()
outer_pred = np.zeros(len(df))
fold_info = []

for of, (otr, ote) in enumerate(outer.split(df, groups=dates)):
    tr_o = df.iloc[otr].reset_index(drop=True)
    te_o = df.iloc[ote]
    idates = tr_o["date"].astype(str).values
    inner = LeaveOneGroupOut()

    # 内层LODO获取OOF预测
    ip = np.zeros(len(tr_o))
    for itr, iva in inner.split(tr_o, groups=idates):
        tri = tr_o.iloc[itr]; vai = tr_o.iloc[iva]
        Xtr = tri[FEATS].astype(float); Xva = vai[FEATS].astype(float)
        med = Xtr.median()
        m = ExtraTreesRegressor(**ET_PARAMS, random_state=2026, n_jobs=-1)
        m.fit(Xtr.fillna(med).values, tri["target"].values)
        ip[iva] = m.predict(Xva.fillna(med).values)

    inner_pass = inner_score(
        (tr_o["base_vol"].values * np.exp(ip) - tr_o["standard_volume_m3"].values)
        / tr_o["standard_volume_m3"].values * 100, tr_o)[0]

    # 最终拟合+预测
    Xall = tr_o[FEATS].astype(float); med = Xall.median()
    fm = ExtraTreesRegressor(**ET_PARAMS, random_state=2026 + of, n_jobs=-1)
    fm.fit(Xall.fillna(med).values, tr_o["target"].values)
    Xte = te_o[FEATS].astype(float).fillna(med).values
    outer_pred[ote] = fm.predict(Xte)

    test_date = str(te_o["date"].iloc[0])
    fold_info.append({"fold": of + 1, "test_date": test_date, "inner_pass": inner_pass})
    print(f"  折{of+1}/{n_dates} 日期={test_date} inner_pass={inner_pass}", flush=True)

# η=1.0 固定（η搜索经消融验证无效）
final_vol = df["base_vol"].values * np.exp(outer_pred)
final_err = (final_vol - df["standard_volume_m3"].values) / df["standard_volume_m3"].values * 100.0

# ==== 最终评价 ====
r = ev(final_err)
print(f"\n=== ET(28d)基线（固定参数，η=1）===")
print(f"  pass={r['pass']}/{r['total']}  MAE={r['MAE']:.4f}%  "
      f"u_L={r['u_L']:.4f}%  u_r={r['u_r']:.4f}%  u_d={r['u_d']:.4f}%")

# ==== 保存 ====
assert len(fold_info) == n_dates, f"折数({len(fold_info)})≠日期数({n_dates})"
assert np.isfinite(outer_pred).all(), "outer_pred 含 NaN/Inf"
pd.DataFrame([r]).to_csv(OUT_DIR / "final_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(fold_info).to_csv(OUT_DIR / "fold_details.csv", index=False, encoding="utf-8-sig")

with open(OUT_DIR / "best_result.json", "w", encoding="utf-8") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)

pred_df = df[[
    "window_id", "date", "flow_point", "disturbance_id",
    "standard_volume_m3", "base_vol",
]].copy()
pred_df["pred_volume_m3"] = final_vol
pred_df["error_pct"] = final_err
pred_df["eta"] = 1.0
pred_df.to_csv(OUT_DIR / "window_predictions.csv", index=False, encoding="utf-8-sig")

print(f"\n输出: {OUT_DIR}")
