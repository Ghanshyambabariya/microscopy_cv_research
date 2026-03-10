from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models as tv_models


def make_dropout(p: float) -> nn.Module:
    return nn.Dropout2d(p) if p > 0 else nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            make_dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            make_dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels, dropout)
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout)
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4, dropout)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2, dropout)
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels, dropout)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.head(x)


class TorchvisionSegWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.model.classifier[4] = nn.Conv2d(self.model.classifier[4].in_channels, num_classes, kernel_size=1)
        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier:
            self.model.aux_classifier[4] = nn.Conv2d(self.model.aux_classifier[4].in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["out"]


class TimmSegHead(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("timm is required for Swin backbones") from exc
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, out_indices=(3,))
        in_channels = self.backbone.feature_info.channels()[-1]
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[-1]
        logits = self.head(feats)
        return nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def _micronet_backbone() -> nn.Module:
    try:
        import pretrained_microscopy_models as pmm
        import torch.utils.model_zoo as model_zoo
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pretrained_microscopy_models is required for MicroNet") from exc
    backbone = tv_models.resnet50(weights=None)
    url = pmm.util.get_pretrained_microscopynet_url("resnet50", "micronet")
    backbone.load_state_dict(model_zoo.load_url(url))
    return backbone


def create_segmentation_model(name: str = "unet_small", *, num_classes: int = 3, base_channels: int = 32, dropout: float = 0.1) -> nn.Module:
    if name == "unet_small":
        return UNetSmall(num_classes=num_classes, base_channels=base_channels, dropout=dropout)
    if name == "deeplab_resnet50":
        weights = tv_models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        model = tv_models.segmentation.deeplabv3_resnet50(weights=weights)
        return TorchvisionSegWrapper(model, num_classes)
    if name == "swin_tiny":
        return TimmSegHead("swin_tiny_patch4_window7_224", num_classes=num_classes, pretrained=True)
    if name == "micronet_deeplab":
        model = tv_models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
        try:
            backbone = _micronet_backbone()
            model.backbone.load_state_dict(backbone.state_dict(), strict=False)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Falling back to ImageNet backbone because MicroNet is unavailable: {exc}")
        return TorchvisionSegWrapper(model, num_classes)
    raise ValueError(f"Unknown segmentation model: {name}")
