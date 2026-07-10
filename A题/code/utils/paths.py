"""项目路径常量与辅助函数。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # A题/
PROBLEM_DIR = ROOT / "problem"
CODE_DIR = ROOT / "code"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    """创建 output 子目录。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
