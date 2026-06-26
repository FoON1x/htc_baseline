"""
HYDRA 实验结果汇总脚本。

扫描 results/ 目录下的所有实验，生成：
1. 每个变体 x 每个种子的详细指标表
2. 每个变体的 mean ± std 汇总
3. 与 HYDRA 论文结果和当前实验结果的对比
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_FILE = PROJECT_DIR / "docs" / "experiment_results_summary.md"


# HYDRA 论文结果 (RoBERTa)
HYDRA_PAPER = {
    'local': {'micro_f1': 86.90, 'macro_f1': 81.18},
    'local_global': {'micro_f1': 86.91, 'macro_f1': 81.22},
    'local_nested': {'micro_f1': 86.90, 'macro_f1': 81.14},
}

# 当前实验结果 (SciBERT)
CURRENT_EXPERIMENT = {
    'parent_micro_f1': 93.31,
    'parent_macro_f1': 93.55,
    'child_micro_f1': 84.99,
    'child_macro_f1': 84.44,
    'hierarchical_consistency': 98.09,
}


def find_experiments(results_dir):
    """扫描所有实验目录，按架构和种子分组。"""
    experiments = defaultdict(list)

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith('hydra_'):
            continue

        test_file = run_dir / 'test_metrics.json'
        config_file = run_dir / 'config.json'

        if not test_file.exists():
            continue

        with open(test_file) as f:
            metrics = json.load(f)
        with open(config_file) as f:
            config = json.load(f)

        arch = config.get('architecture', 'unknown')
        seed = config.get('seed', '?')
        run_name = run_dir.name

        experiments[arch].append({
            'seed': seed,
            'run_name': run_name,
            'metrics': metrics,
            'best_epoch': metrics.get('best_epoch', '?'),
            'training_time': metrics.get('training_time_seconds', 0),
        })

    return experiments


def compute_summary(runs, key):
    """计算某个指标在多个种子上的 mean ± std。"""
    values = [r['metrics'].get(key, None) for r in runs]
    values = [v for v in values if v is not None]
    if not values:
        return "N/A", "N/A"
    return np.mean(values), np.std(values)


def fmt(mean, std):
    """格式化 mean ± std。"""
    if isinstance(mean, str):
        return mean
    return f"{mean*100:.2f} ± {std*100:.2f}"


def fmt_val(v):
    """格式化单个值。"""
    if isinstance(v, float):
        return f"{v*100:.2f}"
    return str(v)


def generate_report(experiments):
    """生成 Markdown 格式的汇总报告。"""
    lines = []
    lines.append("# HYDRA 复现实验结果汇总\n")
    lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- 总览对比表 ----
    lines.append("## 1. 总览对比\n")
    lines.append("| 方法 | Child Micro-F1 | Child Macro-F1 | Parent Micro-F1 | Parent Macro-F1 | Hier Cons |")
    lines.append("|------|---------------|---------------|----------------|----------------|-----------|")

    # 当前实验
    lines.append(
        f"| 当前方法 (SciBERT) | {CURRENT_EXPERIMENT['child_micro_f1']:.2f} "
        f"| {CURRENT_EXPERIMENT['child_macro_f1']:.2f} "
        f"| {CURRENT_EXPERIMENT['parent_micro_f1']:.2f} "
        f"| {CURRENT_EXPERIMENT['parent_macro_f1']:.2f} "
        f"| {CURRENT_EXPERIMENT['hierarchical_consistency']:.2f}% |"
    )

    # HYDRA 论文
    for arch_name, paper_results in HYDRA_PAPER.items():
        display_name = f"HYDRA-{arch_name} (论文, RoBERTa)"
        lines.append(
            f"| {display_name} | {paper_results['micro_f1']:.2f} "
            f"| {paper_results['macro_f1']:.2f} | - | - | - |"
        )

    # HYDRA 复现
    arch_display = {'local': 'Local', 'local_global': 'Local+Global', 'local_nested': 'Local+Nested'}
    for arch in ['local', 'local_global', 'local_nested']:
        if arch in experiments:
            runs = experiments[arch]
            m_cf1, s_cf1 = compute_summary(runs, 'child_micro_f1_argmax')
            m_cmf1, s_cmf1 = compute_summary(runs, 'child_macro_f1_argmax')
            m_pf1, s_pf1 = compute_summary(runs, 'parent_micro_f1_argmax')
            m_pmf1, s_pmf1 = compute_summary(runs, 'parent_macro_f1_argmax')
            m_hc, s_hc = compute_summary(runs, 'hierarchical_consistency')
            lines.append(
                f"| HYDRA-{arch_display.get(arch, arch)} (复现, SciBERT, {len(runs)}seeds) "
                f"| {fmt(m_cf1, s_cf1)} | {fmt(m_cmf1, s_cmf1)} "
                f"| {fmt(m_pf1, s_pf1)} | {fmt(m_pmf1, s_pmf1)} "
                f"| {fmt(m_hc, s_hc)} |"
            )

    lines.append("")

    # ---- 各变体详细结果 ----
    lines.append("## 2. 各变体详细结果\n")

    key_metrics = [
        ('parent_acc_argmax', 'Parent Acc'),
        ('child_acc_argmax', 'Child Acc'),
        ('parent_micro_f1_argmax', 'Parent Micro-F1'),
        ('parent_macro_f1_argmax', 'Parent Macro-F1'),
        ('child_micro_f1_argmax', 'Child Micro-F1'),
        ('child_macro_f1_argmax', 'Child Macro-F1'),
        ('child_micro_precision_argmax', 'Child Micro-Precision'),
        ('child_micro_recall_argmax', 'Child Micro-Recall'),
        ('overall_micro_f1', 'Overall Micro-F1'),
        ('overall_macro_f1', 'Overall Macro-F1'),
        ('hierarchical_consistency', 'Hier Consistency'),
    ]

    for arch in ['local', 'local_global', 'local_nested']:
        if arch not in experiments:
            continue

        runs = experiments[arch]
        lines.append(f"### HYDRA-{arch_display.get(arch, arch)}\n")

        # Per-seed table
        header = "| Seed | " + " | ".join(m[1] for m in key_metrics) + " | Best Epoch | Time |"
        sep = "|------|" + "|".join(["------" for _ in key_metrics]) + "|------|------|"
        lines.append(header)
        lines.append(sep)

        for run in sorted(runs, key=lambda x: x['seed']):
            vals = []
            for key, _ in key_metrics:
                v = run['metrics'].get(key, None)
                vals.append(fmt_val(v) if v is not None else 'N/A')
            epoch = run['best_epoch']
            time_h = f"{run['training_time']/3600:.1f}h" if run['training_time'] else 'N/A'
            lines.append(f"| {run['seed']} | " + " | ".join(vals) + f" | {epoch} | {time_h} |")

        # Mean ± Std row
        summary_vals = []
        for key, _ in key_metrics:
            m, s = compute_summary(runs, key)
            summary_vals.append(fmt(m, s))
        lines.append(f"| **Mean±Std** | " + " | ".join(summary_vals) + " | - | - |")
        lines.append("")

    # ---- 实验配置说明 ----
    lines.append("## 3. 实验配置\n")
    lines.append("| 配置项 | 值 |")
    lines.append("|--------|------|")
    lines.append("| 编码器 | allenai/scibert_scivocab_uncased |")
    lines.append("| 层级 | 2级 (7 parent + 134 child) |")
    lines.append("| 数据划分 | HYDRA官方: seed(7) + split(0.2, rs=0) x2 |")
    lines.append("| Batch size | 32 |")
    lines.append("| Max length | 256 |")
    lines.append("| 学习率 | 3.5e-5 |")
    lines.append("| 早停耐心 | 5 |")
    lines.append("| 损失 | BCE per level + alpha*Global/Nested |")
    lines.append("")

    # ---- 注意事项 ----
    lines.append("## 4. 注意事项\n")
    lines.append("- 论文使用 RoBERTa-large，本实验使用 SciBERT（与当前实验一致），结果不直接可比")
    lines.append("- 论文的 Micro-F1/Macro-F1 报告口径可能包含 overall 指标，需确认")
    lines.append("- 当前实验使用 143 child labels (Data.xlsx)，HYDRA 使用 134 child labels (Y.txt)，数据划分可能不同")
    lines.append("- 如需严格公平对比，建议使用相同数据划分和相同标签口径重新跑当前方法")

    return "\n".join(lines)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR

    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        sys.exit(1)

    experiments = find_experiments(results_dir)

    if not experiments:
        print("未找到实验结果")
        sys.exit(1)

    total_runs = sum(len(v) for v in experiments.values())
    print(f"找到 {total_runs} 个实验结果:")
    for arch, runs in experiments.items():
        print(f"  {arch}: {len(runs)} runs")

    report = generate_report(experiments)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print(report[:2000])


if __name__ == '__main__':
    main()
