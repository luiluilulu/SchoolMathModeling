"""官方评价指标：组通过、u_L、u_r、u_d、MAE。"""

import math
import numpy as np
import pandas as pd


def evaluate(err, df):
    """完整官方五指标。

    Args:
        err: (n,) ndarray, 相对误差百分比，对齐 df 行序
        df: 全局 DataFrame，须含 date, flow_point, disturbance_id,
            condition_note, standard_volume_m3 列

    Returns:
        dict: pass, total, u_L, u_r, u_d, MAE
    """
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

    # u_L: D0 per-flow-point linearity
    d0 = work[work["disturbance_id"] == "D0"]
    d0m = d0.groupby("flow_point")["ep"].mean() if len(d0) > 0 else pd.Series(dtype=float)
    ul = math.sqrt((d0m ** 2).sum() / max(len(d0m) - 1, 1)) if len(d0m) > 1 else 0.0

    # u_r: max group SD
    ur = gdf["s"].max() if not gdf.empty else 0.0

    # u_d: disturbance composite
    use = work[work["flow_point"].between(40, 100)]
    bm = use[use["condition_note"].eq("no_disturbance_reference")]
    dt = use[use["condition_note"].eq("disturbed_test")]
    if len(bm) > 0 and len(dt) > 0:
        bmm = bm.groupby("flow_point")["ep"].mean()
        d1 = dt.groupby(["disturbance_id", "flow_point"])["ep"].agg(
            ["mean", "std"]
        ).reset_index()
        d1["bm"] = d1["flow_point"].map(bmm)
        d1 = d1.dropna(subset=["bm"])
        d1["drift"] = (d1["bm"] - d1["mean"]).abs()
        ud = math.sqrt(
            (d1["drift"].max() / math.sqrt(3)) ** 2 + d1["std"].fillna(0).max() ** 2
        )
    else:
        ud = 0.0

    mae = float(np.abs(err).mean())
    return {
        "pass": gp, "total": len(gdf),
        "u_L": float(ul), "u_r": float(ur), "u_d": float(ud),
        "MAE": mae,
    }


def inner_score(err, df_sub):
    """内层评分（用于超参选择）：(组通过数, -max_SD, -max_|mean|)。

    df_sub 须 reset_index(drop=True) 后索引连续。
    """
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
