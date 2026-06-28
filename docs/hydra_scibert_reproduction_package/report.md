# HYDRA SciBERT 基线复现实验报告

## 1. 实验目标

本实验目标是严格复现 HYDRA (EMNLP 2025) 在 WOS46985 上的三种层级文本分类基线，并将官方使用的 RoBERTa 编码器替换为 SciBERT，使其能与当前项目方法进行更公平的编码器层面对比。

复现实验关注两件事：

- 复现流程是否严格可信：数据划分、训练配置、种子、早停和评测口径尽可能对齐 HYDRA 官方实现。
- 结果记录是否完整：记录 HYDRA 论文式 official metrics，也记录当前项目使用的 parent/child argmax 指标、accuracy、hierarchy consistency 等扩展指标。

## 2. 实验状态

实验已完成。

| 项目 | 状态 |
|---|---|
| HYDRA 变体 | `local`, `local_global`, `local_nested` |
| 种子 | `42, 1, 2, 3, 4` |
| 完整训练/测试 | 15/15 |
| 最终审计 | 通过 |
| 最终结果文件 | 已生成 |
| 可交接轻量产物 | 已整理到本目录 |

最终审计命令：

```bash
python scripts/audit_strict_hydra_results.py results_strict
```

审计结果：`Audit passed.`

## 3. 数据与模型设置

| 配置项 | 严格复现实验设置 |
|---|---|
| 数据集 | WOS46985 |
| 样本数 | 46,985 |
| 标签结构 | 7 parent + 134 child labels |
| 数据文件 | `X.txt`, `YL1.txt`, `YL2.txt`, `Y.txt` |
| 划分策略 | HYDRA 官方策略 |
| 训练/验证/测试 | 30,070 / 7,518 / 9,397 |
| 编码器 | local SciBERT: `pretrained_models/scibert` |
| HYDRA 论文编码器 | RoBERTa-base，用于参考，不作为本实验编码器 |

官方划分策略为：

```text
np.random.seed(7) shuffle
train_test_split(test_size=0.2, random_state=0)
train_test_split(test_size=0.2, random_state=0)
```

## 4. 训练协议

| 配置项 | 值 |
|---|---:|
| Batch size | 32 |
| Max length | 512 |
| Padding | `max_length` |
| Learning rate | 3.5e-5 |
| Warmup steps | 500 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Threshold | 0.5 |
| Loss alpha | 1.0 |
| CUDA fp16 | CUDA 可用时启用 |

早停指标对齐 HYDRA 官方实现：

| 架构 | 早停选择指标 |
|---|---|
| `local` | `training_mode_overall_macro_f1` |
| `local_global` | `unified_macro_f1` |
| `local_nested` | `unified_macro_f1` |

## 5. 评测指标

本实验记录两类指标。

HYDRA official metrics：

- threshold-based multi-label metrics
- training mode local-head metrics
- constrained inference mode metrics
- unified head metrics, where applicable
- overall Micro-F1 / Macro-F1 作为论文式主指标

项目对比指标：

- parent argmax accuracy / Micro-F1 / Macro-F1
- child argmax accuracy / Micro-F1 / Macro-F1
- overall accuracy
- hierarchy consistency
- precision / recall
- best epoch、训练时间、参数量

重要说明：HYDRA official overall metrics 与当前方法报告的 parent/child argmax metrics 不是同一评价口径，不能直接硬比。当前可以直接支持的结论是：在项目当前使用的 parent/child argmax 口径下，我们的方法优于严格复现的 HYDRA+SciBERT 基线。

## 6. 严格 HYDRA+SciBERT 结果

| Architecture | Runs | Official Primary Micro-F1 | Official Primary Macro-F1 | Child Argmax Micro-F1 | Child Argmax Macro-F1 | Parent Argmax Micro-F1 | Parent Argmax Macro-F1 | Overall Acc | Hierarchy Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYDRA Local | 5 | 87.31 +/- 0.10 | 82.70 +/- 0.12 | 82.16 +/- 0.11 | 81.55 +/- 0.17 | 91.50 +/- 0.16 | 91.78 +/- 0.17 | 79.43 +/- 0.51 | 93.56 +/- 0.82 |
| HYDRA Local+Global | 5 | **87.57 +/- 0.08** | 82.79 +/- 0.09 | **82.42 +/- 0.12** | **81.67 +/- 0.13** | **91.69 +/- 0.11** | **91.99 +/- 0.15** | **79.93 +/- 0.37** | **94.03 +/- 0.58** |
| HYDRA Local+Nested | 5 | 87.30 +/- 0.14 | **82.73 +/- 0.09** | 82.27 +/- 0.14 | **81.67 +/- 0.16** | 91.53 +/- 0.17 | 91.84 +/- 0.19 | 79.66 +/- 0.18 | 93.88 +/- 0.50 |

在严格 HYDRA+SciBERT 内部，`local_global` 是整体最强基线。

## 7. 与当前方法对比

| Method | Encoder | Data / Split | Runs | Official Overall Micro-F1 | Official Overall Macro-F1 | Child Micro-F1 | Child Macro-F1 | Parent Micro-F1 | Parent Macro-F1 | Hierarchy Consistency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current method | SciBERT | Project WOS view, 143 child labels | 1 summary | N/A | N/A | **84.99** | **84.44** | **93.31** | **93.55** | **98.09** |
| HYDRA Local+Global, strict reproduction | SciBERT | WOS46985 official split, 134 child labels | 5 seeds | 87.57 +/- 0.08 | 82.79 +/- 0.09 | 82.42 +/- 0.12 | 81.67 +/- 0.13 | 91.69 +/- 0.11 | 91.99 +/- 0.15 | 94.03 +/- 0.58 |

按项目当前 parent/child argmax 口径，相比最强 HYDRA+SciBERT 基线，当前方法提升：

| 指标 | 提升 |
|---|---:|
| Child Micro-F1 | +2.57 |
| Child Macro-F1 | +2.77 |
| Parent Micro-F1 | +1.62 |
| Parent Macro-F1 | +1.56 |
| Hierarchy Consistency | +4.06 |

但需要注意，当前方法使用的是 143 child label 的项目视图，而严格 HYDRA 复现实验使用官方 WOS46985 的 134 child label 口径。若要写成最严格论文主结论，建议后续把当前方法也跑在同一个 134 child label split 上，并补算 HYDRA official overall metrics。

## 8. 实际耗时

15 次严格实验总训练时间约 15.03 小时，RTX 4090 单卡运行。

| Architecture | Runs | Total Time | Avg Time / Run |
|---|---:|---:|---:|
| HYDRA Local | 5 | 5.64h | 1.13h |
| HYDRA Local+Global | 5 | 4.68h | 0.94h |
| HYDRA Local+Nested | 5 | 4.71h | 0.94h |

## 9. 文件说明

本交接包关键文件：

| 文件 | 用途 |
|---|---|
| `README.md` | 目录入口和阅读顺序 |
| `report.md` | 完整实验报告 |
| `reference/method_baseline_comparison.md` | 方法和基线总对比大表 |
| `reference/strict_hydra_scibert_results.md` | 严格 HYDRA 复现结果 |
| `reference/strict_hydra_scibert_protocol.md` | 复现实验协议 |
| `reference/strict_hydra_scibert_audit.md` | 协议审计记录 |
| `data/strict_hydra_scibert_runs.csv` | 15 次完整实验逐 run 指标 |
| `data/strict_hydra_scibert_summary.json` | 聚合统计 |
| `data/run_artifacts/` | 每个 run 的轻量 JSON 产物 |
| `data/run_artifact_manifest.json` | run artifact 索引 |
| `scripts/` | 复现实验脚本副本 |

`data/run_artifacts/` 中包含一个早期中断目录 `hydra_local_seed42_20260627_152059`，只有 `config.json` 和 `best_val_metrics.json`，没有 `test_metrics.json`。该目录已被最终汇总显式忽略，不计入 15 次完整实验。

## 10. 如何复现

在仓库根目录准备数据和模型：

```text
data/wos_raw/WOS46985/
pretrained_models/scibert/
```

运行完整实验：

```bash
bash scripts/run_strict_hydra_scibert.sh
```

查看状态：

```bash
bash scripts/check_strict_hydra_status.sh
```

重新汇总：

```bash
python scripts/summarize_strict_results.py results_strict
```

审计结果：

```bash
python scripts/audit_strict_hydra_results.py results_strict
```

## 11. 后续建议

下一步若要形成最终论文级比较，建议优先做两件事：

1. 将当前方法也运行在 HYDRA 官方 WOS46985 134 child label split 上。
2. 给当前方法补充 HYDRA official threshold-based overall Micro-F1 / Macro-F1。

完成后即可把当前方法与 HYDRA+SciBERT 在同一数据、同一 encoder、同一 metric 下直接比较。
