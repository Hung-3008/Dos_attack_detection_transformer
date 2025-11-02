import math
from typing import List

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_vocab_sizes: List[int],
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        self.cat_vocab_sizes = cat_vocab_sizes
        self.d_model = d_model

        # numeric projection: project full numeric vector to token embeddings for each numeric feature
        # we will project each scalar separately by a small linear layer
        if num_numeric > 0:
            self.num_project = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_numeric)])
        else:
            self.num_project = None

        # categorical embeddings: separate embedding per categorical feature
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size + 1, d_model, padding_idx=0) for vocab_size in cat_vocab_sizes]
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # positional embeddings for sequence length = num_numeric + num_categorical + 1 (CLS)
        seq_len = 1 + num_numeric + len(cat_vocab_sizes)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, dim_feedforward=4 * d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, numeric_tensor, categorical_tensor):
        # numeric_tensor: (B, num_numeric)
        # categorical_tensor: (B, num_cat)
        batch_size = numeric_tensor.size(0)
        token_list = []

        if self.num_project is not None:
            # create per-feature token embeddings for numerics
            for i, proj in enumerate(self.num_project):
                x = numeric_tensor[:, i : i + 1]  # (B,1)
                emb = proj(x)  # (B, d_model)
                token_list.append(emb.unsqueeze(1))

        # categorical embeddings
        for j, emb_layer in enumerate(self.cat_embeddings):
            ids = categorical_tensor[:, j]  # (B,)
            emb = emb_layer(ids)  # (B, d_model)
            token_list.append(emb.unsqueeze(1))

        # stack tokens: list of (B,1,d_model) -> (B, seq_tokens, d_model)
        if token_list:
            tokens = torch.cat(token_list, dim=1)
        else:
            tokens = torch.zeros(batch_size, 0, self.d_model, device=numeric_tensor.device)

        # prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # add positional embedding
        tokens = tokens + self.pos_embedding[:, : tokens.size(1), :]

        # Transformer expects (S, B, E)
        x = tokens.permute(1, 0, 2)
        x = self.encoder(x)  # (S, B, E)
        cls_out = x[0]  # (B, E)

        logits = self.classifier(cls_out)
        return logits
