from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models as tv_models


def _strip_classifier(model: nn.Module, attribute_names: list[str]) -> tuple[nn.Module, int]:
    for attribute_name in attribute_names:
        if hasattr(model, attribute_name):
            module = getattr(model, attribute_name)
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                setattr(model, attribute_name, nn.Identity())
                return model, in_features
    raise ValueError("Could not strip classifier from encoder")


def create_backbone(name: str, *, pretrained: bool = False) -> tuple[nn.Module, int, str]:
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        model, embedding_dim = _strip_classifier(model, ["fc"])
        return model, embedding_dim, "torchvision"

    try:
        import timm
        model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return model, int(model.num_features), "timm"
    except Exception as exc:
        warnings.warn(f"Falling back to torchvision resnet18 because encoder '{name}' is unavailable: {exc}")
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet18(weights=weights)
        model, embedding_dim = _strip_classifier(model, ["fc"])
        return model, embedding_dim, "torchvision-fallback"


class EncoderWithHead(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(inputs)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        return self.head(embeddings)
