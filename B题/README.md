# B题：高精度非圆曲线磨削的砂轮中心轨迹规划

## 目录结构

```
B题/
├── problem/                    # 原始赛题材料（只读）
│   ├── B题_高精度非圆曲线磨削.pdf            # 赛题题面
│   ├── B题_高精度非圆曲线磨削.tex            # 题面 LaTeX 源文件
│   ├── attachment1_profile_points_raw.csv    # 原始轮廓离散点
│   ├── attachment2_profile_points_clean.csv  # 推荐建模轮廓点
│   ├── attachment3_machine_params.csv        # 砂轮、机床和精度参数
│   ├── attachment4_baseline_summary.csv      # 基础统计和样例评价
│   ├── attachment5_data_dictionary.csv       # 字段和附件说明
│   ├── attachment6_problem_diagram.png       # 问题几何示意图
│   └── sample_submission_discrete_offset_2000.csv  # 样例提交
│
├── code/                        # 建模 & 求解代码
│   ├── evaluate_submission.py                # 官方评价脚本
│   ├── eda/                   # 数据探索与可视化
│   ├── model/                 # 模型实现
│   └── utils/                 # 工具函数
│
├── output/                      # 中间产出
│   ├── figures/               # 图表
│   └── results/               # 结果 CSV
│
├── paper/                       # 论文工程
│   ├── main.tex               # 主文件
│   ├── sections/              # 各章节 tex
│   │   ├── 01_problem_analysis.tex
│   │   ├── 02_model.tex
│   │   ├── 03_solution.tex
│   │   ├── 04_results.tex
│   │   └── 05_conclusion.tex
│   ├── figures/               # 论文插图
│   └── ref.bib                # 参考文献
│
└── README.md
```

## 提交格式

```csv
tool_id,x_center_mm,y_center_mm
```

## 评价脚本

```powershell
python code/evaluate_submission.py problem/sample_submission_discrete_offset_2000.csv --output-dir output/results
```
