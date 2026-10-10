"""BiLSTM token-classification model for Pakistani address parsing.

Word-level BiLSTM tagger trained on synthetic labeled addresses.
Labels: BIO scheme over {HOUSE, STREET, BLOCK, COLONY, CITY, O}.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AddressTagger(nn.Module):
    def __init__(self, vocab_size: int, n_labels: int,
                 emb_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ids: (B, T), mask: (B, T) with 1 for real tokens
        emb = self.embedding(ids)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True,
                                                  total_length=ids.size(1))
        return self.classifier(self.dropout(out))  # (B, T, n_labels)
