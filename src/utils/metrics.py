"""
Comprehensive evaluation metrics for hierarchical text classification.

Metrics include:
- Per-level: Accuracy, Micro-Precision/Recall/F1, Macro-Precision/Recall/F1
- Overall (concatenated): Micro-F1, Macro-F1, Accuracy
- Hierarchical consistency
- Supports both single-label and multi-label evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from typing import Dict, List, Any


def compute_level_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          level_name: str = "") -> Dict[str, float]:
    """Compute comprehensive metrics for a single hierarchy level.

    For WOS (single-label per level), accuracy equals micro-F1.
    """
    metrics = {}

    # Accuracy (exact match per sample)
    metrics[f'{level_name}accuracy'] = accuracy_score(y_true, y_pred)

    # Micro-averaged
    metrics[f'{level_name}micro_precision'] = precision_score(
        y_true, y_pred, average='micro', zero_division=0)
    metrics[f'{level_name}micro_recall'] = recall_score(
        y_true, y_pred, average='micro', zero_division=0)
    metrics[f'{level_name}micro_f1'] = f1_score(
        y_true, y_pred, average='micro', zero_division=0)

    # Macro-averaged
    metrics[f'{level_name}macro_precision'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{level_name}macro_recall'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{level_name}macro_f1'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0)

    return metrics


def compute_hierarchical_consistency(parent_preds: np.ndarray,
                                     child_preds: np.ndarray,
                                     parent_child_map: Dict[int, List[int]],
                                     num_parent_labels: int,
                                     num_child_labels: int) -> float:
    """Compute hierarchical consistency: fraction of samples where
    the predicted child label is a valid child of the predicted parent.

    Args:
        parent_preds: (N, num_parent) binary predictions
        child_preds: (N, num_child) binary predictions
        parent_child_map: {parent_idx: [child_indices]}
        num_parent_labels: number of parent labels
        num_child_labels: number of child labels
    """
    n_samples = parent_preds.shape[0]
    consistent = 0

    # Build child -> valid parent mapping
    child_to_parents = {}
    for p_idx, children in parent_child_map.items():
        for c_idx in children:
            if c_idx not in child_to_parents:
                child_to_parents[c_idx] = set()
            child_to_parents[c_idx].add(p_idx)

    for i in range(n_samples):
        pred_parent = np.where(parent_preds[i] == 1)[0]
        pred_child = np.where(child_preds[i] == 1)[0]

        if len(pred_parent) == 0 or len(pred_child) == 0:
            consistent += 0
            continue

        # Check if any predicted child is a valid child of any predicted parent
        valid = False
        for p in pred_parent:
            for c in pred_child:
                if c in parent_child_map.get(p, []):
                    valid = True
                    break
            if valid:
                break

        if valid:
            consistent += 1

    return consistent / n_samples


def compute_single_label_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray,
                                   level_name: str = "") -> Dict[str, float]:
    """For single-label classification, compute accuracy from softmax probabilities.

    This is more precise than threshold-based evaluation for WOS.
    """
    pred_labels = np.argmax(y_pred_probs, axis=1)
    true_labels = np.argmax(y_true, axis=1)

    metrics = {}
    metrics[f'{level_name}acc_argmax'] = accuracy_score(true_labels, pred_labels)

    # Also compute F1 using argmax predictions (one-hot)
    pred_onehot = np.zeros_like(y_true)
    for i, p in enumerate(pred_labels):
        pred_onehot[i, p] = 1

    metrics[f'{level_name}micro_f1_argmax'] = f1_score(
        y_true, pred_onehot, average='micro', zero_division=0)
    metrics[f'{level_name}micro_precision_argmax'] = precision_score(
        y_true, pred_onehot, average='micro', zero_division=0)
    metrics[f'{level_name}micro_recall_argmax'] = recall_score(
        y_true, pred_onehot, average='micro', zero_division=0)
    metrics[f'{level_name}macro_f1_argmax'] = f1_score(
        y_true, pred_onehot, average='macro', zero_division=0)
    metrics[f'{level_name}macro_precision_argmax'] = precision_score(
        y_true, pred_onehot, average='macro', zero_division=0)
    metrics[f'{level_name}macro_recall_argmax'] = recall_score(
        y_true, pred_onehot, average='macro', zero_division=0)

    return metrics


def evaluate_all(parent_true: np.ndarray, child_true: np.ndarray,
                 parent_preds: np.ndarray, child_preds: np.ndarray,
                 parent_probs: np.ndarray, child_probs: np.ndarray,
                 parent_child_map: Dict[int, List[int]],
                 threshold: float = 0.5) -> Dict[str, float]:
    """Compute all metrics for 2-level HTC on WOS46985.

    Args:
        parent_true: (N, P) ground truth parent labels
        child_true: (N, C) ground truth child labels
        parent_preds: (N, P) thresholded binary predictions for parent
        child_preds: (N, C) thresholded binary predictions for child
        parent_probs: (N, P) raw sigmoid probabilities for parent
        child_probs: (N, C) raw sigmoid probabilities for child
        parent_child_map: {parent_idx: [child_indices]}
        threshold: binary threshold
    """
    all_metrics = {}

    # Per-level metrics (threshold-based)
    all_metrics.update(compute_level_metrics(parent_true, parent_preds, "parent_"))
    all_metrics.update(compute_level_metrics(child_true, child_preds, "child_"))

    # Per-level metrics (argmax-based, more suitable for single-label)
    all_metrics.update(compute_single_label_accuracy(parent_true, parent_probs, "parent_"))
    all_metrics.update(compute_single_label_accuracy(child_true, child_probs, "child_"))

    # Overall (concatenated) metrics
    all_true = np.concatenate([parent_true, child_true], axis=1)
    all_preds = np.concatenate([parent_preds, child_preds], axis=1)
    all_metrics['overall_micro_f1'] = f1_score(all_true, all_preds, average='micro', zero_division=0)
    all_metrics['overall_macro_f1'] = f1_score(all_true, all_preds, average='macro', zero_division=0)
    all_metrics['overall_accuracy'] = accuracy_score(all_true, all_preds)
    all_metrics['overall_micro_precision'] = precision_score(all_true, all_preds, average='micro', zero_division=0)
    all_metrics['overall_micro_recall'] = recall_score(all_true, all_preds, average='micro', zero_division=0)
    all_metrics['overall_macro_precision'] = precision_score(all_true, all_preds, average='macro', zero_division=0)
    all_metrics['overall_macro_recall'] = recall_score(all_true, all_preds, average='macro', zero_division=0)

    # Hierarchical consistency
    num_parent = parent_true.shape[1]
    num_child = child_true.shape[1]
    all_metrics['hierarchical_consistency'] = compute_hierarchical_consistency(
        parent_preds, child_preds, parent_child_map, num_parent, num_child)

    return all_metrics
