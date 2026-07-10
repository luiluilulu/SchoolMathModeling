# 数学建模校内赛

## 团队成员 & 分工

| 角色 | 姓名 | 职责 | 分支 |
|------|------|------|------|
| 建模手 | — | 问题分析、数学建模、公式推导 | `model` |
| 编程手 | — | 数据处理、模型求解、算法实现 | `solver` |
| 写作手 | — | 论文撰写、图表美化、排版 | `paper` |

## 目录结构

```
├── A题/                        # A题：扰流条件下的超声波流量计
│   ├── analysis/               #   建模手产出（分析报告、公式推导）
│   ├── problem/                #   原始赛题材料（只读，不改）
│   ├── code/                   #   编程手产出（求解脚本 + utils）
│   ├── output/                 #   图表 & 结果 CSV
│   │   ├── figures/
│   │   └── results/
│   ├── paper/                  #   论文手产出（LaTeX 工程）
│   └── README.md
│
├── B题/                        # B题：高精度非圆曲线磨削
│   └── ...（同上结构）
│
├── refs/                       # 共享参考文献 / 资料
├── .gitignore
└── README.md
```

## 协作约定

### 分支策略

```bash
# 每人从 main 拉自己的分支
git checkout -b model    # 建模手
git checkout -b solver   # 编程手
git checkout -b paper    # 写作手
```

- **不要直接往 main 上 push**，在自己的分支上开发
- 阶段性成果通过 PR / merge 合入 main
- **每次 merge 前确保代码可运行、论文可编译**

### 工作流程

```
1. 三人一起讨论选题（A/B），读题、查资料
2. 选定后只保留对应题目目录重点推进
3. 建模手 → 模型设计文档 → 编程手接数据实现
4. 编程手 → 结果 csv + 图表 → 写作手接论文
5. 写作手 → 初稿 → 三人轮流审阅 → 终稿
```

### 提交要求

- 建模论文（PDF）
- 结果 CSV 文件
- 可复现支撑代码（完整源码 + 运行说明 + 环境依赖）

## 快速开始

```bash
# 克隆后安装依赖
pip install numpy scipy pandas matplotlib

# 运行评价脚本（以 A 题为例）
cd A题/code
python ../problem/evaluate_submission.py ../problem/sample_submission_phys6.csv --output-dir ../output/results
```
