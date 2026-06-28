# Method and Baseline Comparison

Generated from the current project summary and the strict HYDRA SciBERT reproduction completed on 2026-06-28.

## Main Comparison Table

| Method | Encoder | Data / Split | Runs | Official Overall Micro-F1 | Official Overall Macro-F1 | Child Micro-F1 | Child Macro-F1 | Parent Micro-F1 | Parent Macro-F1 | Overall Acc | Hierarchy Consistency | Comparison Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Current method | SciBERT | Project WOS view, 143 child labels | 1 summary | N/A | N/A | **84.99** | **84.44** | **93.31** | **93.55** | N/A | **98.09** | Project argmax parent/child metrics; strongest directly comparable row for current project evaluation. |
| HYDRA Local, strict reproduction | SciBERT | WOS46985 official HYDRA split, 134 child labels | 5 seeds | 87.31 +/- 0.10 | 82.70 +/- 0.12 | 82.16 +/- 0.11 | 81.55 +/- 0.17 | 91.50 +/- 0.16 | 91.78 +/- 0.17 | 79.43 +/- 0.51 | 93.56 +/- 0.82 | Official metrics follow HYDRA threshold protocol; child/parent metrics are added argmax diagnostics. |
| HYDRA Local+Global, strict reproduction | SciBERT | WOS46985 official HYDRA split, 134 child labels | 5 seeds | **87.57 +/- 0.08** | 82.79 +/- 0.09 | **82.42 +/- 0.12** | **81.67 +/- 0.13** | **91.69 +/- 0.11** | **91.99 +/- 0.15** | **79.93 +/- 0.37** | **94.03 +/- 0.58** | Best strict HYDRA+SciBERT baseline by child/parent argmax metrics. |
| HYDRA Local+Nested, strict reproduction | SciBERT | WOS46985 official HYDRA split, 134 child labels | 5 seeds | 87.30 +/- 0.14 | **82.73 +/- 0.09** | 82.27 +/- 0.14 | **81.67 +/- 0.16** | 91.53 +/- 0.17 | 91.84 +/- 0.19 | 79.66 +/- 0.18 | 93.88 +/- 0.50 | Similar to Local+Global, slightly lower micro-F1 and consistency. |
| HYDRA Local, paper | RoBERTa-base | WOS46985 paper setting | 5 seeds | 86.90 | 81.18 | N/A | N/A | N/A | N/A | N/A | N/A | Paper result; encoder differs from this project's SciBERT comparison. |
| HYDRA Global Head, paper | RoBERTa-base | WOS46985 paper setting | 5 seeds | 86.91 | 81.22 | N/A | N/A | N/A | N/A | N/A | N/A | Paper result corresponding to the global-head HYDRA variant. |
| HYDRA Nested Head, paper | RoBERTa-base | WOS46985 paper setting | 5 seeds | 86.83 | 81.08 | N/A | N/A | N/A | N/A | N/A | N/A | Paper result corresponding to the nested-head HYDRA variant. |

## Interpretation

The current method is ahead of the strict HYDRA+SciBERT baselines under the project-style parent/child argmax evaluation. Compared with the strongest strict baseline, HYDRA Local+Global, the current method improves child Micro-F1 by 2.57 points, child Macro-F1 by 2.77 points, parent Micro-F1 by 1.62 points, parent Macro-F1 by 1.56 points, and hierarchy consistency by 4.06 points.

The HYDRA official overall metrics are not directly comparable to the current method's reported parent/child argmax metrics. HYDRA's paper-style protocol is threshold-based multi-label evaluation over the whole hierarchy, while the current method summary reports separate single-label parent and child metrics. For a fully strict paper claim, the current method should also be evaluated on the WOS46985 134-child-label split with the same HYDRA official overall metrics.

## Source Files

- Current method summary: `docs/experiment_results_summary.md`
- Strict HYDRA SciBERT report: `docs/strict_hydra_scibert_results.md`
- Strict per-run metrics: `docs/strict_hydra_scibert_runs.csv`
- Reproduction audit: `docs/strict_hydra_scibert_audit.md`
