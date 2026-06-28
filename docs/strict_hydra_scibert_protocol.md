# Strict HYDRA SciBERT Reproduction Protocol

## Objective

Run HYDRA on WOS46985 with the official HYDRA experimental protocol while replacing the encoder with SciBERT for fair comparison with the current method. The result should be auditable, reproducible, and include both paper-style metrics and this project's comparison metrics.

Protocol settings were cross-checked against the HYDRA paper and official repository. The paper's reported WOS comparison numbers use RoBERTa-base; this reproduction intentionally uses SciBERT, so exact score matching is not required.

## Dataset

- Source files: `data/wos_raw/WOS46985/X.txt`, `YL1.txt`, `YL2.txt`, `Y.txt`
- Samples: 46,985
- Labels: 7 parent labels + 134 global child labels from `Y.txt`
- Split: official HYDRA WOS preprocessing strategy:
  - `np.random.seed(7)` shuffle
  - `train_test_split(test_size=0.2, random_state=0)`
  - second `train_test_split(test_size=0.2, random_state=0)`
  - expected split sizes: train 30,070 / val 7,518 / test 9,397

## Model Variants

Run all three HYDRA variants:

- `local`: local heads only
- `local_global`: local heads plus a unified global head
- `local_nested`: local heads plus a nested unified head over local logits

The implementation follows the official HYDRA architecture: shared encoder, optional embedding projection, two-layer MLP heads, BCE loss per local level, and optional unified BCE loss.

## Strict Training Settings

- Encoder: `pretrained_models/scibert`
- Batch size: 32
- Max length: 512
- Padding: `max_length`, matching the official HYDRA dataset class
- Learning rate: 3.5e-5
- Warmup steps: 500
- Max epochs: 50
- Early stopping patience: 5
- Threshold: 0.5
- Loss alpha: 1.0
- Seeds: `42, 1, 2, 3, 4`
- CUDA fp16: enabled automatically when CUDA is available

Early stopping follows the official HYDRA implementation:

- `local`: select by validation `training_mode_overall_macro_f1`
- `local_global` / `local_nested`: select by validation `unified_macro_f1`

## Recorded Metrics

Each run writes `config.json`, `best_val_metrics.json`, `training_history.json`, and `test_metrics.json`. Metrics include:

- Official threshold metrics: accuracy, micro precision, micro recall, micro-F1, macro precision, macro recall, macro-F1
- Official modes: `training_mode_*`, `inference_mode_*`, and `unified_*` where available
- Current-method comparison metrics: parent/child argmax accuracy, micro/macro precision, recall, and F1
- Overall threshold metrics from concatenated parent + child labels
- Hierarchical consistency
- Average true/predicted labels per sample
- Best epoch, selected validation metric, total training time, and parameter count

Primary comparison uses the metric family that matches each architecture:

- `local`: local training-mode overall Micro/Macro-F1
- `local_global`: unified Global Head Micro/Macro-F1
- `local_nested`: unified Nested Head Micro/Macro-F1

## Run Command

```bash
bash scripts/run_strict_hydra_scibert.sh
```

The script writes run artifacts to `results_strict/`, logs to `logs_strict/`, and summary artifacts to:

- `docs/strict_hydra_scibert_results.md`
- `docs/strict_hydra_scibert_summary.json`
- `docs/strict_hydra_scibert_runs.csv`

## Estimated Runtime

Based on the first completed strict run on the RTX 4090 in this environment, one epoch is about 181 seconds with `max_length=512` and fixed padding. The first run (`local`, seed 42) stopped at epoch 21 after 3,824 seconds of training and selected epoch 16. Expected runtime is about 15-20 hours if most runs stop around 20-25 epochs, and up to about 38 hours if all 15 runs reach 50 epochs.

## Current Status

- [x] Strict training and evaluation code implemented
- [x] Strict run script added
- [x] Strict summary script added
- [x] Smoke test passed on a tiny debug subset
- [x] Detached full run started in `screen` session `hydra_strict`
- [x] Completion guard started in `screen` session `hydra_strict_guard`
- [ ] Full 15-run strict experiment completed
- [ ] Strict result summary generated from `results_strict/`

## Long-Running Execution Notes

The full experiment is running in detached `screen` sessions, so closing SSH or shutting down the local computer does not stop training. The remote server/AutoDL instance must remain powered on.

Use these commands after reconnecting:

```bash
cd /root/autodl-tmp/htc_baseline
bash scripts/check_strict_hydra_status.sh
```

Important background sessions:

- `hydra_strict`: main 15-run experiment
- `hydra_strict_guard`: waits for the main run, reruns missing/incomplete strict runs, then regenerates summaries

When `scripts/check_strict_hydra_status.sh` reports `total: 15/15`, inspect `docs/strict_hydra_scibert_results.md` for the final report.

## Completion Criteria

The experiment is complete only when all of the following hold:

- `bash scripts/check_strict_hydra_status.sh` reports `total: 15/15`
- `python scripts/audit_strict_hydra_results.py results_strict` exits successfully
- `docs/strict_hydra_scibert_results.md` contains `Status: **complete**`
- `docs/strict_hydra_scibert_summary.json` and `docs/strict_hydra_scibert_runs.csv` exist and cover all 15 runs

The audit checks strict SciBERT configuration, seed coverage, required official/project metrics, best-model artifacts, and summary files.
