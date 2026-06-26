"""
HYDRA model implementations for hierarchical text classification.

Three variants:
  - HYDRA: Local Heads Only
  - HYDRAGlobal: Local Heads + Global Head
  - HYDRANested: Local Heads + Nested Head

Adapted from official implementation: https://github.com/FKarl/HYDRA
Modified to use SciBERT and support MPS/CUDA.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from src.data.wos_dataset import HierarchyInfo

logger = logging.getLogger(__name__)


class HYDRA(nn.Module):
    """HYDRA with Local Heads Only.

    Each hierarchy level gets its own classification head.
    Shared encoder + embedding projection.
    """

    def __init__(self, label_dims: List[int], hierarchy_info: HierarchyInfo,
                 encoder: str = 'allenai/scibert_scivocab_uncased',
                 pooling: str = 'cls', project_embedding: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        self.num_levels = len(label_dims)
        d = self.encoder.config.hidden_size
        self.hierarchy_info = hierarchy_info
        self.pooling = pooling

        self.project_embedding = project_embedding
        self.embedding_size = d
        if project_embedding:
            self.embedding_projector = nn.Linear(d, d * self.num_levels)
            self.embedding_size = d * self.num_levels

        # Level-specific classifiers (2-layer MLP as in official code)
        self.level_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size * 2),
                nn.LayerNorm(self.embedding_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_size * 2, label_dims[level]),
            ) for level in range(self.num_levels)
        ])

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            return hidden_states.mean(dim=1)
        elif self.pooling == 'max':
            return hidden_states.max(dim=1).values
        elif self.pooling == 'last':
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.project_embedding:
            hidden = self.embedding_projector(outputs.last_hidden_state)
        else:
            hidden = outputs.last_hidden_state

        full_embedding = self._pool(hidden)

        level_predictions = []
        for level in range(self.num_levels):
            level_logits = self.level_classifiers[level](full_embedding)
            level_predictions.append(level_logits)

        return level_predictions, None


class HYDRAGlobal(nn.Module):
    """HYDRA with Local Heads + Global Head.

    Adds a unified global classifier over the full label set.
    """

    def __init__(self, label_dims: List[int], hierarchy_info: HierarchyInfo,
                 encoder: str = 'allenai/scibert_scivocab_uncased',
                 pooling: str = 'cls', project_embedding: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        self.num_levels = len(label_dims)
        d = self.encoder.config.hidden_size
        self.hierarchy_info = hierarchy_info
        self.pooling = pooling

        self.project_embedding = project_embedding
        self.embedding_size = d
        if project_embedding:
            self.embedding_projector = nn.Linear(d, d * self.num_levels)
            self.embedding_size = d * self.num_levels

        self.level_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size * 2),
                nn.LayerNorm(self.embedding_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_size * 2, label_dims[level]),
            ) for level in range(self.num_levels)
        ])

        total_labels = sum(label_dims)
        total_input = d * self.num_levels if project_embedding else d
        self.unified_classifier = nn.Sequential(
            nn.Linear(total_input, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d * 2, total_labels),
        )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            return hidden_states.mean(dim=1)
        elif self.pooling == 'max':
            return hidden_states.max(dim=1).values
        elif self.pooling == 'last':
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.project_embedding:
            hidden = self.embedding_projector(outputs.last_hidden_state)
        else:
            hidden = outputs.last_hidden_state

        full_embedding = self._pool(hidden)

        level_predictions = []
        for level in range(self.num_levels):
            level_logits = self.level_classifiers[level](full_embedding)
            level_predictions.append(level_logits)

        unified_logits = self.unified_classifier(full_embedding)

        return level_predictions, unified_logits


class HYDRANested(nn.Module):
    """HYDRA with Local Heads + Nested Head.

    The global head takes concatenated local head outputs as input.
    """

    def __init__(self, label_dims: List[int], hierarchy_info: HierarchyInfo,
                 encoder: str = 'allenai/scibert_scivocab_uncased',
                 pooling: str = 'cls', project_embedding: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        self.num_levels = len(label_dims)
        d = self.encoder.config.hidden_size
        self.hierarchy_info = hierarchy_info
        self.pooling = pooling

        self.project_embedding = project_embedding
        self.embedding_size = d
        if project_embedding:
            self.embedding_projector = nn.Linear(d, d * self.num_levels)
            self.embedding_size = d * self.num_levels

        self.level_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size * 2),
                nn.LayerNorm(self.embedding_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.embedding_size * 2, label_dims[level]),
            ) for level in range(self.num_levels)
        ])

        total_labels = sum(label_dims)
        self.unified_classifier = nn.Sequential(
            nn.Linear(total_labels, d * 2),
            nn.LayerNorm(d * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d * 2, total_labels),
        )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'cls':
            return hidden_states[:, 0, :]
        elif self.pooling == 'mean':
            return hidden_states.mean(dim=1)
        elif self.pooling == 'max':
            return hidden_states.max(dim=1).values
        elif self.pooling == 'last':
            return hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.project_embedding:
            hidden = self.embedding_projector(outputs.last_hidden_state)
        else:
            hidden = outputs.last_hidden_state

        full_embedding = self._pool(hidden)

        level_predictions = []
        for level in range(self.num_levels):
            level_logits = self.level_classifiers[level](full_embedding)
            level_predictions.append(level_logits)

        concat_local_logits = torch.cat(level_predictions, dim=1)
        unified_logits = self.unified_classifier(concat_local_logits)

        return level_predictions, unified_logits
