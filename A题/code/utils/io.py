"""数据读写。"""

import pandas as pd

from .paths import PROBLEM_DIR


def load_attachment1():
    """加载附件1窗口主数据，类型转换。"""
    df = pd.read_csv(PROBLEM_DIR / "attachment1_window_data.csv", encoding="utf-8-sig")
    float_cols = [
        "standard_volume_m3",
        "phys6_volume_m3", "owics_volume_m3",
        "lagrange_volume_m3", "equal_weight_volume_m3",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df
