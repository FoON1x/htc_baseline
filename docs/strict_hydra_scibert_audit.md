# Strict HYDRA SciBERT Reproduction Audit

## Scope

This audit checks whether the strict SciBERT reproduction is procedurally aligned with the official HYDRA implementation. Score differences from the HYDRA paper are acceptable because the official paper scripts use `FacebookAI/roberta-base`, while this reproduction intentionally uses local SciBERT for fair comparison with the current project.

## Official Reference

- Repository: `https://github.com/FKarl/HYDRA`
- Checked commit: `d0d38164d3ffe02179f9b9a084c73a9a252b3f8c`
- Local comparison copy: `/tmp/HYDRA_official`

## Protocol Alignment

| Item | Official HYDRA | Strict SciBERT run | Status |
|---|---|---|---|
| Architectures | `local`, `local_global`, `local_nested` | same | aligned |
| Seeds | `42 1 2 3 4` | `42,1,2,3,4` | aligned |
| Encoder | `FacebookAI/roberta-base` | `pretrained_models/scibert` | intentional change |
| Batch size | `32` | `32` | aligned |
| Max length | `512` | `512` | aligned |
| Padding | `max_length` in official dataset class | `max_length` | aligned |
| Epoch cap | `50` | `50` | aligned |
| Learning rate | `3.5e-5` | `3.5e-5` | aligned |
| Warmup | `500` | `500` | aligned |
| Early stop patience | `5` | `5` | aligned |
| Threshold | `0.5` | `0.5` | aligned |
| Loss alpha | `1.0` | `1.0` | aligned |

## Dataset Alignment

The strict run uses `data/wos_raw/WOS46985/X.txt`, `YL1.txt`, `YL2.txt`, and `Y.txt`. Runtime logs confirm 46,985 samples, 7 parent labels, 134 child labels, and split sizes `train=30070`, `val=7518`, `test=9397`.

The split matches official preprocessing: `np.random.seed(7)` shuffle, then `train_test_split(test_size=0.2, random_state=0)` followed by a second split of the train portion with the same random state.

## Metric Alignment

The reproduction records the official threshold metrics for training-mode local heads, constrained inference-mode local heads, and unified heads where applicable. It also records project comparison metrics, including parent/child argmax accuracy and F1, overall threshold metrics, hierarchy consistency, training time, parameter count, best epoch, and selected validation metric.

Early stopping follows the official choice:

- `local`: validation `training_mode_overall_macro_f1`
- `local_global` and `local_nested`: validation `unified_macro_f1`

## Known Distinctions

- The encoder is SciBERT, not RoBERTa-base, by design.
- The strict run uses original WOS46985 global child labels from `Y.txt` with 134 child classes. This differs from project experiments that use a 143-child-label `Data.xlsx` view.
- Older exploratory results in `docs/experiment_results_summary.md` used `max_length=256` and `num_epochs=30`; those are not the strict reproduction results.

## Completion Gate

The strict experiment is credible only after all 15 runs finish and `scripts/audit_strict_hydra_results.py results_strict` exits successfully. The final report must contain `Status: **complete**` in `docs/strict_hydra_scibert_results.md`.
