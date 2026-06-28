# Strict HYDRA SciBERT Reproduction Results

Generated: 2026-06-28 18:18:41

Results directory: `results_strict`

## Protocol Check

Status: **complete**. Found all 3 architectures x 5 seeds with strict protocol settings.


Non-strict or incomplete result directories ignored by the summary:

- `hydra_local_seed42_20260627_152059`: missing test_metrics.json; incomplete or interrupted run

## Primary Official-Style Metrics

Primary metrics follow the official HYDRA evaluation mode for each architecture. `local` uses local-head training-mode overall Micro/Macro-F1. `local_global` and `local_nested` use the unified head Micro/Macro-F1, matching the validation metric used for early stopping.

| Architecture | Runs | Primary Micro-F1 | Primary Macro-F1 | Paper WOS Micro-F1 | Paper WOS Macro-F1 |
|---|---:|---:|---:|---:|---:|
| HYDRA Local | 5 | 87.31 +/- 0.10 | 82.70 +/- 0.12 | 86.90 | 81.18 |
| HYDRA Local+Global | 5 | 87.57 +/- 0.08 | 82.79 +/- 0.09 | 86.91 | 81.22 |
| HYDRA Local+Nested | 5 | 87.30 +/- 0.14 | 82.73 +/- 0.09 | 86.83 | 81.08 |

## Per-Seed Details

Per-seed tables include the official HYDRA metric families plus project comparison metrics. `Argmax` metrics are single-label parent/child diagnostics for comparison with the current method; they are not replacements for the official threshold metrics.

### HYDRA Local

| Seed | Best Epoch | Ran Epochs | Best Val Metric | TrainMode Micro | TrainMode Macro | InferMode Micro | InferMode Macro | Unified Micro | Unified Macro | Overall Acc | Child Argmax Micro | Child Argmax Macro | Child Acc | Parent Acc | Hierarchy Consistency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 20 | 82.67 | 87.10 | 82.71 | 87.03 | 82.57 | N/A | N/A | 78.58 | 82.06 | 81.64 | 82.06 | 91.18 | 92.16 |
| 2 | 22 | 27 | 83.37 | 87.33 | 82.50 | 87.31 | 82.44 | N/A | N/A | 79.87 | 82.05 | 81.24 | 82.05 | 91.57 | 94.35 |
| 3 | 21 | 26 | 82.92 | 87.34 | 82.76 | 87.29 | 82.67 | N/A | N/A | 79.75 | 82.13 | 81.58 | 82.13 | 91.58 | 94.29 |
| 4 | 13 | 18 | 82.70 | 87.35 | 82.66 | 87.28 | 82.49 | N/A | N/A | 79.08 | 82.25 | 81.55 | 82.25 | 91.55 | 93.16 |
| 42 | 16 | 21 | 83.34 | 87.40 | 82.85 | 87.39 | 82.76 | N/A | N/A | 79.86 | 82.32 | 81.76 | 82.32 | 91.60 | 93.85 |
| **Mean+/-Std** | - | - | - | 87.31 +/- 0.10 | 82.70 +/- 0.12 | 87.26 +/- 0.12 | 82.59 +/- 0.12 | N/A | N/A | 79.43 +/- 0.51 | 82.16 +/- 0.11 | 81.55 +/- 0.17 | 82.16 +/- 0.11 | 91.50 +/- 0.16 | 93.56 +/- 0.82 |

### HYDRA Local+Global

| Seed | Best Epoch | Ran Epochs | Best Val Metric | TrainMode Micro | TrainMode Macro | InferMode Micro | InferMode Macro | Unified Micro | Unified Macro | Overall Acc | Child Argmax Micro | Child Argmax Macro | Child Acc | Parent Acc | Hierarchy Consistency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 19 | 82.92 | 87.38 | 82.74 | 87.33 | 82.60 | 87.46 | 82.66 | 79.88 | 82.31 | 81.50 | 82.31 | 91.51 | 94.05 |
| 2 | 15 | 20 | 83.29 | 87.54 | 82.89 | 87.48 | 82.76 | 87.52 | 82.81 | 80.11 | 82.37 | 81.73 | 82.37 | 91.74 | 94.48 |
| 3 | 14 | 19 | 83.06 | 87.60 | 82.77 | 87.53 | 82.59 | 87.67 | 82.87 | 80.17 | 82.62 | 81.69 | 82.62 | 91.69 | 94.05 |
| 4 | 16 | 21 | 83.10 | 87.52 | 82.86 | 87.53 | 82.87 | 87.57 | 82.90 | 80.24 | 82.49 | 81.87 | 82.49 | 91.68 | 94.63 |
| 42 | 8 | 13 | 82.71 | 87.50 | 82.59 | 87.49 | 82.57 | 87.65 | 82.72 | 79.23 | 82.32 | 81.55 | 82.32 | 91.85 | 92.96 |
| **Mean+/-Std** | - | - | - | 87.51 +/- 0.07 | 82.77 +/- 0.10 | 87.47 +/- 0.08 | 82.68 +/- 0.12 | 87.57 +/- 0.08 | 82.79 +/- 0.09 | 79.93 +/- 0.37 | 82.42 +/- 0.12 | 81.67 +/- 0.13 | 82.42 +/- 0.12 | 91.69 +/- 0.11 | 94.03 +/- 0.58 |

### HYDRA Local+Nested

| Seed | Best Epoch | Ran Epochs | Best Val Metric | TrainMode Micro | TrainMode Macro | InferMode Micro | InferMode Macro | Unified Micro | Unified Macro | Overall Acc | Child Argmax Micro | Child Argmax Macro | Child Acc | Parent Acc | Hierarchy Consistency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 17 | 83.18 | 87.18 | 82.74 | 87.13 | 82.62 | 87.13 | 82.72 | 79.48 | 82.19 | 81.54 | 82.19 | 91.34 | 93.90 |
| 2 | 16 | 21 | 82.72 | 87.45 | 82.91 | 87.38 | 82.76 | 87.43 | 82.81 | 79.75 | 82.27 | 81.69 | 82.27 | 91.66 | 94.38 |
| 3 | 14 | 19 | 82.67 | 87.27 | 82.76 | 87.26 | 82.69 | 87.25 | 82.61 | 79.76 | 82.08 | 81.44 | 82.08 | 91.53 | 94.49 |
| 4 | 14 | 19 | 82.92 | 87.29 | 82.80 | 87.19 | 82.53 | 87.19 | 82.65 | 79.42 | 82.32 | 81.79 | 82.32 | 91.36 | 93.22 |
| 42 | 12 | 17 | 83.06 | 87.67 | 83.22 | 87.64 | 83.13 | 87.50 | 82.83 | 79.88 | 82.51 | 81.90 | 82.51 | 91.77 | 93.42 |
| **Mean+/-Std** | - | - | - | 87.37 +/- 0.17 | 82.89 +/- 0.18 | 87.32 +/- 0.18 | 82.74 +/- 0.21 | 87.30 +/- 0.14 | 82.73 +/- 0.09 | 79.66 +/- 0.18 | 82.27 +/- 0.14 | 81.67 +/- 0.16 | 82.27 +/- 0.14 | 91.53 +/- 0.17 | 93.88 +/- 0.50 |

## Experimental Settings

- Encoder: SciBERT from `pretrained_models/scibert`
- Data: WOS46985 raw txt files, 7 parent + 134 child labels
- Split: official HYDRA WOS split procedure (`np.random.seed(7)` shuffle, two `train_test_split(..., random_state=0)` calls)
- Hyperparameters: batch size 32, max length 512, max_length padding, learning rate 3.5e-5, warmup 500, threshold 0.5, loss alpha 1.0
- Early stopping: official HYDRA code rule, patience 5; local uses training-mode overall macro-F1, global/nested use unified macro-F1
- Paper comparison values: WOS RoBERTa-base rows from HYDRA Table 3; `local_global` uses Global Head and `local_nested` uses Nested Head because those match the unified-head primary metrics