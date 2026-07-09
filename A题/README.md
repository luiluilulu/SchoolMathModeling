# A题：扰流条件下的超声波流量计

## 目录结构

```
A题/
├── problem/                    # 原始赛题材料（只读）
│   ├── A题_扰流条件下的超声波流量计.pdf       # 赛题题面
│   ├── A题_扰流条件下的超声波流量计.tex       # 题面 LaTeX 源文件
│   ├── attachment1_window_data.csv           # 窗口级主数据
│   ├── attachment2_condition_schedule.csv    # 测试日期与扰流状态
│   ├── attachment3_data_dictionary.csv       # 字段说明
│   ├── attachment4_baseline_summary.csv      # 基础方法误差统计
│   ├── attachment5_inspection_targets.csv    # 评价指标说明
│   ├── attachment6_window_raw_samples.csv    # 窗口级原始声道观测（～9MB）
│   ├── attachment7_meter_geometry.csv        # 仪表几何与声道参数
│   └── sample_submission_phys6.csv           # 样例提交文件
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
window_id,model_volume_m3
```

## 评价脚本

```powershell
python code/evaluate_submission.py problem/sample_submission_phys6.csv --output-dir output/results
```
