from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EncoderSpec:
    name: str
    source: str
    family: str
    notes: str


ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "micronet": EncoderSpec("micronet", "https://github.com/nasa/pretrained-microscopy-models", "microscopy-specialized", "Microscopy-specialized family to benchmark when local weights are available."),
    "cytoself": EncoderSpec("cytoself", "https://github.com/royerlab/cytoself", "microscopy-specialized", "Strong reference for morphology-centric embeddings."),
    "uni": EncoderSpec("uni", "https://github.com/KatherLab/uni", "biomedical-foundation", "Useful biomedical foundation benchmark."),
    "titan": EncoderSpec("titan", "https://github.com/mahmoodlab/TITAN", "biomedical-foundation", "Multimodal-oriented biomedical encoder family."),
    "dinov2_vitb14": EncoderSpec("dinov2_vitb14", "generic-self-supervised", "general-foundation", "Strong generic SSL encoder benchmark."),
    "convnextv2_base": EncoderSpec("convnextv2_base", "timm", "general-supervised", "Strong image benchmark backbone."),
    "resnet18": EncoderSpec("resnet18", "torchvision", "general-supervised", "Lightweight runnable baseline."),
}


def get_encoder_spec(name: str) -> EncoderSpec:
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder: {name}")
    return ENCODER_REGISTRY[name]
