from __future__ import annotations

import torch
from torch import nn

from microscopy_cv_research.models.heads import ClassificationHead, RegressionHead


class HybridMultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = ClassificationHead(embedding_dim, num_classes)
        self.regressor = RegressionHead(embedding_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encoder(inputs)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        return {
            "classification_logits": self.classifier(embeddings),
            "regression_output": self.regressor(embeddings),
            "embeddings": embeddings,
        }
