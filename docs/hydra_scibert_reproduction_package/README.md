# HYDRA SciBERT 复现实验交接包

本目录是 HYDRA (EMNLP 2025) 在 WOS46985 上的 SciBERT 基线复现实验交接包。它面向后续同事阅读、复核、继续实验或整理论文结果。

## 阅读顺序

1. `report.md`：完整实验报告，包含目标、协议、结果、对比结论和注意事项。
2. `reference/method_baseline_comparison.md`：我们的方法与 HYDRA 基线的大表对比。
3. `reference/strict_hydra_scibert_results.md`：严格 HYDRA+SciBERT 的最终结果表。
4. `reference/strict_hydra_scibert_audit.md`：复现实验与官方 HYDRA 协议的对齐审计。
5. `reference/strict_hydra_scibert_protocol.md`：实验设计、训练配置、评测指标和完成标准。
6. `data/strict_hydra_scibert_runs.csv`：15 次完整实验的逐 run 指标。
7. `data/strict_hydra_scibert_summary.json`：按架构聚合的 mean/std/min/max 统计。
8. `data/run_artifacts/`：每个 run 的轻量 JSON 产物，包含 config、best validation metrics、training history 和 test metrics。

## 目录结构

```text
hydra_scibert_reproduction_package/
├── README.md
├── report.md
├── data/
│   ├── strict_hydra_scibert_runs.csv
│   ├── strict_hydra_scibert_summary.json
│   ├── run_artifact_manifest.json
│   └── run_artifacts/
├── reference/
│   ├── method_baseline_comparison.md
│   ├── strict_hydra_scibert_results.md
│   ├── strict_hydra_scibert_protocol.md
│   └── strict_hydra_scibert_audit.md
└── scripts/
    ├── run_strict_hydra_scibert.sh
    ├── check_strict_hydra_status.sh
    ├── ensure_strict_hydra_complete.sh
    ├── finalize_strict_hydra_when_done.sh
    ├── audit_strict_hydra_results.py
    └── summarize_strict_results.py
```

## 包含与不包含

本包包含完整实验记录、逐 run 指标、配置、训练历史和复现脚本副本。为了便于传输，本包不包含原始数据、SciBERT 模型文件、训练日志和 checkpoint 权重。

如需重新运行实验，需要在仓库根目录准备：

- `data/wos_raw/WOS46985/`
- `pretrained_models/scibert/`
- Python 环境和依赖

## 当前结论

严格 HYDRA+SciBERT 复现实验已完成 3 个 HYDRA 变体 x 5 个种子，共 15 次完整训练和测试，并通过最终审计。当前方法在项目使用的 parent/child argmax 评价口径下优于严格复现的 HYDRA+SciBERT 基线；但 HYDRA official overall 指标和当前方法指标口径不同，论文表述时需要明确说明。
