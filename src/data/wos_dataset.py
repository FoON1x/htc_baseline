"""
WOS46985 dataset loading and preprocessing for HYDRA.

Uses the original WOS46985 data files (X.txt, YL1.txt, YL2.txt, Y.txt)
which have 7 parent labels and 134 child labels, matching the official
dataset description and HYDRA paper.

Adapted from the official HYDRA implementation:
https://github.com/FKarl/HYDRA
"""

import os
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class HierarchyInfo:
    """Hierarchy metadata for HTC."""
    label_dims: Dict[int, int]
    parent_child_map: Dict[int, Dict[int, List[int]]]
    label_names: Dict[int, Dict[int, str]]
    level_names: Dict[int, str]


def clean_str(string):
    """Clean text string (from HYDRA's official preprocess_wos.py)."""
    string = string.strip().strip('"')
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_wos_raw(data_dir: str) -> Tuple[List[str], List[int], List[int], List[int]]:
    """Load WOS46985 from original X.txt / YL1.txt / YL2.txt / Y.txt files.

    Returns:
        texts: list of cleaned text strings
        yl1_labels: parent label indices (0-6)
        yl2_labels: child label indices per parent (local)
        y_labels: global child label indices (0-133)
    """
    x_path = os.path.join(data_dir, 'X.txt')
    yl1_path = os.path.join(data_dir, 'YL1.txt')
    yl2_path = os.path.join(data_dir, 'YL2.txt')
    y_path = os.path.join(data_dir, 'Y.txt')

    texts, yl1_labels, yl2_labels, y_labels = [], [], [], []

    with open(x_path, 'r', encoding='utf-8') as fx, \
         open(yl1_path, 'r') as fl1, \
         open(yl2_path, 'r') as fl2, \
         open(y_path, 'r') as fy:
        for text, l1, l2, y in zip(fx, fl1, fl2, fy):
            texts.append(clean_str(text))
            yl1_labels.append(int(l1.strip()))
            yl2_labels.append(int(l2.strip()))
            y_labels.append(int(y.strip()))

    logger.info(f"Loaded {len(texts)} samples from {data_dir}")
    return texts, yl1_labels, yl2_labels, y_labels


def split_wos_hydra(texts, yl1_labels, y_labels):
    """Split WOS data using HYDRA's official strategy:
    np.random.seed(7) shuffle, then train_test_split(test_size=0.2, random_state=0) x2
    """
    n = len(texts)
    indices = np.arange(n)

    np.random.seed(7)
    np.random.shuffle(indices)

    texts = [texts[i] for i in indices]
    yl1_labels = [yl1_labels[i] for i in indices]
    y_labels = [y_labels[i] for i in indices]

    train_val_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=0)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2, random_state=0)

    def _select(idx_list):
        sel_texts = [texts[i] for i in idx_list]
        sel_yl1 = [yl1_labels[i] for i in idx_list]
        sel_y = [y_labels[i] for i in idx_list]
        return sel_texts, sel_yl1, sel_y

    return _select(train_idx), _select(val_idx), _select(test_idx)


def build_hierarchy(yl1_labels: List[int], y_labels: List[int]) -> HierarchyInfo:
    """Build hierarchy info from label lists.

    Uses Y.txt global child indices (0-133) as child labels.
    """
    num_parents = max(yl1_labels) + 1
    num_children = max(y_labels) + 1

    parent_child_map = {}
    seen_pairs = set()
    for p, c in zip(yl1_labels, y_labels):
        if (p, c) not in seen_pairs:
            seen_pairs.add((p, c))
            if p not in parent_child_map:
                parent_child_map[p] = []
            parent_child_map[p].append(c)

    for p in parent_child_map:
        parent_child_map[p].sort()

    yl1_names = {
        0: 'CS', 1: 'ECE', 2: 'Psychology',
        3: 'MAE', 4: 'Civil', 5: 'Medical', 6: 'biochemistry'
    }

    return HierarchyInfo(
        label_dims={0: num_parents, 1: num_children},
        parent_child_map={1: parent_child_map},
        label_names={
            0: yl1_names,
            1: {i: f'child_{i}' for i in range(num_children)},
        },
        level_names={0: 'parent', 1: 'child'},
    )


class WOSDataset(Dataset):
    """PyTorch Dataset for WOS hierarchical text classification.

    Single-label per level: each sample has one parent and one child label.
    Uses truncation-only tokenization (no padding) for dynamic padding.
    """

    def __init__(self, texts: List[str], yl1_labels: List[int],
                 y_labels: List[int], num_parents: int, num_children: int,
                 tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.texts = texts
        self.yl1_labels = yl1_labels
        self.y_labels = y_labels
        self.num_parents = num_parents
        self.num_children = num_children
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors=None,
        )

        parent_label = torch.zeros(self.num_parents, dtype=torch.float32)
        parent_label[self.yl1_labels[idx]] = 1.0

        child_label = torch.zeros(self.num_children, dtype=torch.float32)
        child_label[self.y_labels[idx]] = 1.0

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': [parent_label, child_label],
        }


class HTCDataCollator:
    """Custom collator that handles dynamic padding for HTC batches."""

    def __init__(self, tokenizer: PreTrainedTokenizer, num_levels: int = 2):
        self.tokenizer = tokenizer
        self.num_levels = num_levels

    def __call__(self, features):
        batch_labels = [[] for _ in range(self.num_levels)]
        text_features = []

        for feature in features:
            for level in range(self.num_levels):
                batch_labels[level].append(feature['labels'][level])
            text_features.append({
                'input_ids': feature['input_ids'],
                'attention_mask': feature['attention_mask'],
            })

        padded = self.tokenizer.pad(
            text_features,
            padding='longest',
            return_tensors='pt',
        )

        stacked_labels = [
            torch.stack(batch_labels[level]) for level in range(self.num_levels)
        ]

        return {
            'input_ids': padded['input_ids'],
            'attention_mask': padded['attention_mask'],
            'labels': stacked_labels,
        }


def get_wos_datasets(data_dir: str, tokenizer: PreTrainedTokenizer,
                     max_length: int = 256) -> Tuple[WOSDataset, WOSDataset, WOSDataset, HierarchyInfo]:
    """Load train, val, test datasets for WOS46985 from original txt files."""
    if os.path.exists(os.path.join(data_dir, 'X.txt')):
        raw_dir = data_dir
    elif os.path.exists(os.path.join(data_dir, 'WOS46985', 'X.txt')):
        raw_dir = os.path.join(data_dir, 'WOS46985')
    else:
        raise FileNotFoundError(
            f"Cannot find WOS46985 data files in {data_dir} or {data_dir}/WOS46985/")

    texts, yl1_labels, yl2_labels, y_labels = load_wos_raw(raw_dir)

    hierarchy_info = build_hierarchy(yl1_labels, y_labels)
    num_parents = hierarchy_info.label_dims[0]
    num_children = hierarchy_info.label_dims[1]

    logger.info(f"WOS46985: {len(texts)} samples, "
                f"{num_parents} parent labels, {num_children} child labels")

    (train_texts, train_yl1, train_y), \
    (val_texts, val_yl1, val_y), \
    (test_texts, test_yl1, test_y) = split_wos_hydra(texts, yl1_labels, y_labels)

    logger.info(f"Split: train={len(train_texts)}, val={len(val_texts)}, test={len(test_texts)}")

    train_dataset = WOSDataset(train_texts, train_yl1, train_y,
                               num_parents, num_children, tokenizer, max_length)
    val_dataset = WOSDataset(val_texts, val_yl1, val_y,
                             num_parents, num_children, tokenizer, max_length)
    test_dataset = WOSDataset(test_texts, test_yl1, test_y,
                              num_parents, num_children, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset, hierarchy_info
