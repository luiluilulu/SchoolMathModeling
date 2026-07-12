# 数学建模校内赛 — A题：扰流条件下的超声波流量计

## 团队成员

| 角色 | 产出目录 |
|------|------|
| 建模手 | `A题/analysis/` |
| 编程手 | `A题/code/` |
| 写作手 | `A题/paper/` |

## 目录结构

```
├── analysis/    # 建模手 — 分析报告、公式推导
│   ├── 问题1_题目分析报告.md
│   ├── 问题2_题目分析报告.md
│   ├── 问题3_扰流剖面识别与补偿.md
│   ├── 问题4_最终达标模型.md
│   └── 术语表格.md
├── code/        # 编程手 — 求解脚本 + utils
│   ├── 问题1_误差对比.py
│   ├── 问题2_无扰流模型.py
│   ├── evaluate_submission.py
│   └── utils/
├── output/      # 图表 + 结果 CSV
│   ├── figures/
│   └── results/
├── paper/       # 写作手 — LaTeX 论文
│   ├── main.tex
│   ├── sections/
│   └── ref.bib
├── problem/     # 原始赛题材料（只读）
│   ├── attachment1_window_data.csv
│   ├── attachment2_condition_schedule.csv
│   ├── attachment3_data_dictionary.csv
│   ├── attachment4_baseline_summary.csv
│   ├── attachment5_inspection_targets.csv
│   ├── attachment6_window_raw_samples.csv
│   ├── attachment7_meter_geometry.csv
│   └── sample_submission_phys6.csv
├── refs/            # 论文格式规范
├── CLAUDE.md        # AI 风格指南
├── CODE_STYLE.md    # 代码与写作规范
└── requirements.txt
```

## 协作约定

直接在 `master` 上开发——三人各改各的目录，互不冲突。push 前 `git pull --rebase`。

提交信息格式：`model:` / `solver:` / `paper:` / `fix:` / `chore:` + 中文简述。

## 运行

```bash
pip install -r requirements.txt
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
