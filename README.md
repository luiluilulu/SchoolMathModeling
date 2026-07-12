# 数学建模校内赛 — A题：扰流条件下的超声波流量计

## 团队成员

| 角色 | 姓名 | 产出目录 |
|------|------|------|
| 建模手 | — | `A题/analysis/` |
| 编程手 | — | `A题/code/` |
| 写作手 | — | `A题/paper/` |

## 目录结构

```
├── A题/
│   ├── analysis/    # 建模手 — 分析报告、公式推导
│   ├── code/        # 编程手 — 求解脚本 + utils
│   ├── output/      # 图表 + 结果 CSV
│   │   ├── figures/
│   │   └── results/
│   ├── paper/       # 写作手 — LaTeX 论文
│   ├── problem/     # 原始赛题材料（只读）
│   └── README.md
├── refs/            # 论文格式规范
├── CLAUDE.md        # AI 风格指南
├── CODE_STYLE.md    # 代码与写作规范
├── requirements.txt
└── README.md
```

## 协作约定

直接在 `master` 上开发——三人各改各的目录（analysis / code / paper），互不冲突。push 前 `git pull --rebase`。

## 快速开始

```bash
pip install -r requirements.txt
cd A题/code

# 问题1：误差对比
python 问题1_误差对比.py

# 问题2：无扰流模型
python 问题2_无扰流模型.py
```
