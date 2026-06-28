"""
HYDRA training and evaluation script for WOS46985.

Features:
- Supports all three HYDRA variants (local, local_global, local_nested)
- SciBERT encoder (same as current experiment for fair comparison)
- MPS/CUDA/CPU auto-detection
- CUDA fp16 mixed precision for faster training on GPU
- Dynamic padding for efficient memory usage
- Comprehensive metrics (per-level + overall + hierarchical consistency)
- Local JSON logging (no wandb dependency)
- Multiple seeds for reproducibility
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wos_dataset import get_wos_datasets, HierarchyInfo, HTCDataCollator
from src.models.hydra import HYDRA, HYDRAGlobal, HYDRANested
from src.utils.metrics import evaluate_all

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        logger.info("Using MPS device")
        return torch.device('mps')
    else:
        logger.info("Using CPU device")
        return torch.device('cpu')


def local_and_global_loss(level_outputs, unified_output, labels,
                          alpha=1.0, architecture='local'):
    """Compute HYDRA loss: sum of per-level BCE + optional global/nested BCE."""
    total_loss = torch.tensor(0.0, device=labels[0].device)

    if level_outputs:
        level_loss = torch.tensor(0.0, device=labels[0].device)
        for level, (output, label) in enumerate(zip(level_outputs, labels)):
            bce = F.binary_cross_entropy_with_logits(output, label)
            level_loss = level_loss + bce
        total_loss = total_loss + level_loss

    if architecture in ('local_global', 'local_nested') and unified_output is not None:
        unified_labels = torch.cat(labels, dim=1)
        unified_loss = F.binary_cross_entropy_with_logits(unified_output, unified_labels)
        total_loss = total_loss + alpha * unified_loss

    return total_loss


def binary_metrics(y_true, y_pred, prefix):
    """Compute official HYDRA-style threshold metrics for multi-hot labels."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        f'{prefix}accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}precision': precision_score(
            y_true, y_pred, average='micro', zero_division=0),
        f'{prefix}recall': recall_score(
            y_true, y_pred, average='micro', zero_division=0),
        f'{prefix}micro_f1': f1_score(
            y_true, y_pred, average='micro', zero_division=0),
        f'{prefix}macro_f1': f1_score(
            y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}macro_precision': precision_score(
            y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}macro_recall': recall_score(
            y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}avg_true_labels': float(y_true.sum(axis=1).mean()),
        f'{prefix}avg_pred_labels': float(y_pred.sum(axis=1).mean()),
    }


def add_prefixed_metrics(target, source, prefix):
    for key, value in source.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            target[f'{prefix}{key}'] = float(value)


def official_selection_key(architecture):
    """Metric used by the official HYDRA implementation for early stopping."""
    if architecture in ('local_global', 'local_nested'):
        return 'unified_macro_f1'
    return 'training_mode_overall_macro_f1'


def resolve_selection_key(config):
    if config.selection_metric == 'official':
        return official_selection_key(config.architecture)
    return config.selection_metric


def resolve_padding_strategy(config):
    if config.padding_strategy == 'none':
        return False
    return config.padding_strategy


def train_one_epoch(model, train_loader, optimizer, scheduler, device, config,
                    scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = [label.to(device) for label in batch['labels']]

        optimizer.zero_grad()

        if scaler is not None:
            # CUDA fp16 mixed precision
            with torch.amp.autocast('cuda'):
                level_outputs, unified_output = model(input_ids, attention_mask)
                loss = local_and_global_loss(
                    level_outputs, unified_output, labels,
                    alpha=config.loss_alpha, architecture=config.architecture,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            level_outputs, unified_output = model(input_ids, attention_mask)
            loss = local_and_global_loss(
                level_outputs, unified_output, labels,
                alpha=config.loss_alpha, architecture=config.architecture,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, data_loader, device, hierarchy_info, threshold=0.5,
             scaler=None):
    model.eval()
    all_parent_probs, all_child_probs = [], []
    all_parent_true, all_child_true = [], []
    unified_probs, unified_true = [], []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        if scaler is not None and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                level_outputs, unified_output = model(input_ids, attention_mask)
        else:
            level_outputs, unified_output = model(input_ids, attention_mask)

        all_parent_probs.append(torch.sigmoid(level_outputs[0]).cpu().float().numpy())
        all_child_probs.append(torch.sigmoid(level_outputs[1]).cpu().float().numpy())
        all_parent_true.append(labels[0].numpy())
        all_child_true.append(labels[1].numpy())
        if unified_output is not None:
            unified_probs.append(torch.sigmoid(unified_output).cpu().float().numpy())
            unified_true.append(torch.cat(labels, dim=1).numpy())

    parent_probs = np.concatenate(all_parent_probs, axis=0)
    child_probs = np.concatenate(all_child_probs, axis=0)
    parent_true = np.concatenate(all_parent_true, axis=0)
    child_true = np.concatenate(all_child_true, axis=0)

    parent_preds = (parent_probs > threshold).astype(int)
    child_preds = (child_probs > threshold).astype(int)

    metrics = evaluate_all(
        parent_true=parent_true, child_true=child_true,
        parent_preds=parent_preds, child_preds=child_preds,
        parent_probs=parent_probs, child_probs=child_probs,
        parent_child_map=hierarchy_info.parent_child_map.get(1, {}),
        threshold=threshold,
    )

    # Official HYDRA naming: "training_mode" is the direct thresholded local-head
    # output. Keep flat aliases for easy summary-table generation.
    add_prefixed_metrics(metrics, binary_metrics(parent_true, parent_preds, ''),
                         'training_mode_level_0_')
    add_prefixed_metrics(metrics, binary_metrics(child_true, child_preds, ''),
                         'training_mode_level_1_')
    all_true = np.concatenate([parent_true, child_true], axis=1)
    all_train_preds = np.concatenate([parent_preds, child_preds], axis=1)
    add_prefixed_metrics(metrics, binary_metrics(all_true, all_train_preds, ''),
                         'training_mode_overall_')

    # Official implementation also reports greedy constrained inference. It is
    # not the paper's main mode, but recording it makes the reproduction auditable.
    inf_parent_preds = parent_preds.copy()
    inf_child_preds = np.zeros_like(child_preds)
    parent_child_map = hierarchy_info.parent_child_map.get(1, {})
    for i in range(child_preds.shape[0]):
        parent_ids = np.where(inf_parent_preds[i] == 1)[0]
        allowed_children = []
        for parent_id in parent_ids:
            allowed_children.extend(parent_child_map.get(parent_id, []))
        if allowed_children:
            inf_child_preds[i, allowed_children] = child_preds[i, allowed_children]

    add_prefixed_metrics(metrics, binary_metrics(parent_true, inf_parent_preds, ''),
                         'inference_mode_level_0_')
    add_prefixed_metrics(metrics, binary_metrics(child_true, inf_child_preds, ''),
                         'inference_mode_level_1_')
    all_inf_preds = np.concatenate([inf_parent_preds, inf_child_preds], axis=1)
    add_prefixed_metrics(metrics, binary_metrics(all_true, all_inf_preds, ''),
                         'inference_mode_overall_')

    if unified_probs:
        unified_probs_arr = np.concatenate(unified_probs, axis=0)
        unified_true_arr = np.concatenate(unified_true, axis=0)
        unified_preds = (unified_probs_arr > threshold).astype(int)
        add_prefixed_metrics(metrics, binary_metrics(unified_true_arr, unified_preds, ''),
                             'unified_')

    metrics['eval_threshold'] = float(threshold)
    return metrics


def run_experiment(config):
    device = get_device()
    config.selection_metric_resolved = resolve_selection_key(config)
    config.padding_strategy_resolved = resolve_padding_strategy(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"hydra_{config.architecture}_seed{config.seed}_{timestamp}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=2)

    logger.info(f"Run: {run_name}")
    logger.info(f"Architecture: {config.architecture} | Encoder: {config.model_name}")
    logger.info(
        f"Max length: {config.max_length} | Batch size: {config.batch_size} | "
        f"Padding: {config.padding_strategy_resolved}")
    logger.info(f"Selection metric: {config.selection_metric_resolved}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset, test_dataset, hierarchy_info = get_wos_datasets(
        data_dir=config.data_dir, tokenizer=tokenizer, max_length=config.max_length,
        padding=config.padding_strategy_resolved)

    if config.limit_train_samples:
        train_dataset = Subset(train_dataset, range(min(config.limit_train_samples, len(train_dataset))))
        logger.info(f"Debug limit: train samples={len(train_dataset)}")
    if config.limit_eval_samples:
        val_dataset = Subset(val_dataset, range(min(config.limit_eval_samples, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(config.limit_eval_samples, len(test_dataset))))
        logger.info(f"Debug limit: val/test samples={len(val_dataset)}/{len(test_dataset)}")

    num_levels = len(hierarchy_info.label_dims)
    collator = HTCDataCollator(tokenizer, num_levels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            num_workers=0, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             num_workers=0, collate_fn=collator)

    label_dims = [hierarchy_info.label_dims[i] for i in range(num_levels)]
    logger.info(f"Label dims: {label_dims}")

    model_cls = {
        'local': HYDRA,
        'local_global': HYDRAGlobal,
        'local_nested': HYDRANested,
    }[config.architecture]
    model = model_cls(
        label_dims=label_dims, hierarchy_info=hierarchy_info,
        encoder=config.model_name, pooling=config.pooling,
        project_embedding=config.project_embedding,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    model.to(device)

    # CUDA fp16
    scaler = None
    if device.type == 'cuda' and config.fp16:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Using CUDA fp16 mixed precision")

    # Weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped, lr=config.learning_rate)

    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps)

    best_val_metric = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info("Starting training...")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info(f"Total training steps: {num_training_steps}")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     device, config, scaler)

        val_metrics = evaluate(model, val_loader, device, hierarchy_info,
                               config.threshold, scaler)
        val_metrics['train_loss'] = train_loss
        val_metrics['epoch'] = epoch + 1
        val_metrics['epoch_time'] = time.time() - epoch_start
        history.append(val_metrics)

        current_metric = val_metrics.get(config.selection_metric_resolved)
        if current_metric is None:
            available = ', '.join(sorted(val_metrics.keys()))
            raise KeyError(
                f"Selection metric '{config.selection_metric_resolved}' not found. "
                f"Available metrics: {available}")
        val_metrics['selection_metric'] = config.selection_metric_resolved
        val_metrics['selection_metric_value'] = float(current_metric)

        logger.info(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Loss: {train_loss:.4f} | Time: {val_metrics['epoch_time']:.0f}s | "
            f"Selection {config.selection_metric_resolved}: {current_metric:.4f} | "
            f"Training Overall Micro-F1: {val_metrics['training_mode_overall_micro_f1']:.4f} | "
            f"Training Overall Macro-F1: {val_metrics['training_mode_overall_macro_f1']:.4f} | "
            f"Child Argmax Micro-F1: {val_metrics['child_micro_f1_argmax']:.4f}")

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            with open(output_dir / 'best_val_metrics.json', 'w') as f:
                json.dump(val_metrics, f, indent=2)
            logger.info(f"  -> New best ({config.selection_metric_resolved}: {current_metric:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement ({patience_counter}/{config.early_stopping_patience})")
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    logger.info(f"Training done in {training_time:.1f}s ({training_time/3600:.1f}h), best epoch: {best_epoch}")

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set with best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))
    test_metrics = evaluate(model, test_loader, device, hierarchy_info,
                            config.threshold, scaler)
    test_metrics['best_epoch'] = best_epoch
    test_metrics['best_val_metric'] = float(best_val_metric)
    test_metrics['selection_metric'] = config.selection_metric_resolved
    test_metrics['protocol'] = 'strict_hydra_scibert_official_metrics'
    test_metrics['training_time_seconds'] = training_time
    test_metrics['total_params'] = total_params

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return test_metrics, output_dir


def main():
    parser = argparse.ArgumentParser(description='HYDRA on WOS46985')

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='data/wos_raw/WOS46985')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--padding_strategy', type=str, default='none',
                        choices=['none', 'max_length'],
                        help="Tokenizer padding strategy. Use max_length for strict official HYDRA reproduction.")
    parser.add_argument('--limit_train_samples', type=int, default=0,
                        help='Debug only: limit train samples when > 0.')
    parser.add_argument('--limit_eval_samples', type=int, default=0,
                        help='Debug only: limit val/test samples when > 0.')

    # Model
    parser.add_argument('--model_name', type=str, default='pretrained_models/scibert')
    parser.add_argument('--architecture', type=str, default='local',
                        choices=['local', 'local_global', 'local_nested'])
    parser.add_argument('--pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'last'])
    parser.add_argument('--project_embedding', action='store_true', default=True)
    parser.add_argument('--no_project_embedding', action='store_false', dest='project_embedding')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=3.5e-5)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--loss_alpha', type=float, default=1.0)
    parser.add_argument('--selection_metric', type=str, default='official',
                        help=("Metric used for early stopping. Use 'official' to match "
                              "HYDRA: local=training_mode_overall_macro_f1, "
                              "local_global/local_nested=unified_macro_f1."))
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use fp16 mixed precision (CUDA only)')

    # Seeds
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, default='results')

    config = parser.parse_args()

    if config.seeds:
        seeds = [int(s) for s in config.seeds.split(',')]
    else:
        seeds = [config.seed]

    all_results = {}
    for seed in seeds:
        config.seed = seed
        test_metrics, output_dir = run_experiment(config)
        all_results[str(seed)] = {'metrics': test_metrics, 'output_dir': str(output_dir)}

    if len(seeds) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-SEED SUMMARY")
        logger.info("=" * 60)
        key_metrics = [
            'parent_acc_argmax', 'child_acc_argmax',
            'child_micro_f1_argmax', 'child_macro_f1_argmax',
            'parent_micro_f1_argmax', 'parent_macro_f1_argmax',
            'overall_micro_f1', 'overall_macro_f1',
            'hierarchical_consistency',
        ]
        summary = {}
        for key in key_metrics:
            values = [all_results[s]['metrics'].get(key, 0) for s in all_results]
            if values:
                mean_val, std_val = np.mean(values), np.std(values)
                summary[key] = {'mean': float(mean_val), 'std': float(std_val)}
                logger.info(f"  {key}: {mean_val:.4f} +/- {std_val:.4f}")

        with open(Path(config.output_dir) / f'multi_seed_summary_{config.architecture}.json', 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
