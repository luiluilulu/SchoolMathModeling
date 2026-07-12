# A题：扰流条件下的超声波流量计

## 目录结构

```
A题/
├── analysis/                    # 建模手产出
│   ├── 问题1_题目分析报告.md
│   ├── 问题2_题目分析报告.md
│   └── 术语表格.md
│
├── code/                        # 编程手产出
│   ├── evaluate_submission.py   # 官方评价脚本
│   ├── 问题1_误差对比.py
│   ├── 问题2_无扰流模型.py
│   └── utils/                   # 路径、数据加载
│
├── output/                      # 图表 + 结果
│   ├── figures/
│   └── results/
│
├── paper/                       # 写作手产出
│   ├── main.tex
│   ├── sections/
│   ├── ref.bib
│   └── 写作模板.md
│
├── problem/                     # 原始赛题材料（只读）
│   ├── attachment1~7
│   └── sample_submission_phys6.csv
│
└── README.md
```

## 运行

```bash
cd A题/code

# 问题1：四方法误差对比
python 问题1_误差对比.py

# 问题2：Gauss-Jacobi 零参数模型
python 问题2_无扰流模型.py

# 官方评价
python evaluate_submission.py ../output/results/problem2_results.csv --output-dir ../output/results/eval
```

## 提交格式

```csv
window_id,model_volume_m3
```
