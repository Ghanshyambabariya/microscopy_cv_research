from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from microscopy_cv_research.config import load_config
from microscopy_cv_research.data.datasets import MicroscopyHybridDataset
from microscopy_cv_research.data.splits import make_group_train_val_test_split
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.models.backbones import create_backbone
from microscopy_cv_research.models.encoder_registry import get_encoder_spec
from microscopy_cv_research.models.hybrid import HybridMultiTaskModel
from microscopy_cv_research.training.engine import get_device, run_hybrid_epoch, save_checkpoint, save_json
from microscopy_cv_research.training.synthetic import generate_synthetic_images
from microscopy_cv_research.utils.repro import set_seed


def _build_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform


def plan_hybrid_experiment(config_path: str | Path) -> dict:
    config = load_config(config_path)
    return {
        "experiment_name": config["experiment_name"],
        "encoder": asdict(get_encoder_spec(config["encoder_name"])),
        "classification_target": config["classification_target"],
        "regression_target": config["regression_target"],
        "synthetic_mix_ratio": config["synthetic_mix_ratio"],
    }


def run_hybrid_experiment(config_path: str | Path, synthetic_config_path: str | Path | None = None) -> dict:
    config = load_config(config_path)
    set_seed(config["seed"])
    table = pd.read_csv(config["dataset_csv"])
    train_df, val_df, test_df = make_group_train_val_test_split(table, group_column=config["group_column"], random_state=config["seed"])

    synthetic_report = None
    if synthetic_config_path is not None and config.get("synthetic_mix_ratio", 0) > 0:
        synthetic_report = generate_synthetic_images(synthetic_config_path)
        synthetic_table = pd.read_csv(Path(synthetic_report["table_path"]))
        synthetic_take = max(1, int(len(train_df) * config["synthetic_mix_ratio"]))
        synthetic_subset = synthetic_table.sample(n=min(synthetic_take, len(synthetic_table)), random_state=config["seed"]).copy()
        synthetic_subset["image_root_override"] = synthetic_report["image_dir"]
        synthetic_subset[config["group_column"]] = synthetic_subset["specimen_id"].astype(str)
        train_df = pd.concat([train_df, synthetic_subset], ignore_index=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([table[config["classification_target"]], train_df[config["classification_target"]]], ignore_index=True))
    train_df = train_df.assign(**{config["classification_target"]: label_encoder.transform(train_df[config["classification_target"]]).astype("int64")})
    val_df = val_df.assign(**{config["classification_target"]: label_encoder.transform(val_df[config["classification_target"]]).astype("int64")})
    test_df = test_df.assign(**{config["classification_target"]: label_encoder.transform(test_df[config["classification_target"]]).astype("int64")})

    train_transform, eval_transform = _build_transforms(config["image_size"])
    device = get_device()
    encoder_model, embedding_dim, backbone_source = create_backbone(config["encoder_name"], pretrained=config.get("pretrained", False))
    model = HybridMultiTaskModel(encoder_model, embedding_dim, config["num_classes"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_dataset = MicroscopyHybridDataset(train_df, config["image_root"], config["classification_target"], config["regression_target"], train_transform)
    val_dataset = MicroscopyHybridDataset(val_df, config["image_root"], config["classification_target"], config["regression_target"], eval_transform)
    test_dataset = MicroscopyHybridDataset(test_df, config["image_root"], config["classification_target"], config["regression_target"], eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    history = []
    best_metric = float("-inf")
    best_state = None
    for epoch in range(1, config["epochs"] + 1):
        train_loss, _, _, _, _ = run_hybrid_epoch(model, train_loader, optimizer, device, config["classification_weight"], config["regression_weight"], config["consistency_weight"])
        val_loss, class_preds, class_targets, reg_preds, reg_targets = run_hybrid_epoch(model, val_loader, None, device, config["classification_weight"], config["regression_weight"], config["consistency_weight"])
        class_metrics = classification_metrics(class_targets, class_preds)
        reg_metrics = regression_metrics(reg_targets, reg_preds)
        primary_metric = class_metrics["macro_f1"] + max(reg_metrics["r2"], 0.0)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **class_metrics, **reg_metrics})
        if primary_metric > best_metric:
            best_metric = primary_metric
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, class_preds, class_targets, reg_preds, reg_targets = run_hybrid_epoch(model, test_loader, None, device, config["classification_weight"], config["regression_weight"], config["consistency_weight"])
    report = {
        "experiment_name": config["experiment_name"],
        "encoder": asdict(get_encoder_spec(config["encoder_name"])),
        "runtime_backbone_source": backbone_source,
        "synthetic_report": synthetic_report,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "history": history,
        "test_loss": test_loss,
        "classification_metrics": classification_metrics(class_targets, class_preds),
        "regression_metrics": regression_metrics(reg_targets, reg_preds),
        "label_mapping": {int(i): label for i, label in enumerate(label_encoder.classes_)},
    }
    save_checkpoint({"model_state": model.state_dict(), "config": config, "report": report}, config["save_path"])
    save_json(report, config["report_path"])
    return report
