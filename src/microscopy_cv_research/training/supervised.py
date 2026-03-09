from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from microscopy_cv_research.config import load_config
from microscopy_cv_research.data.datasets import MicroscopyImageDataset
from microscopy_cv_research.data.splits import make_group_train_val_test_split
from microscopy_cv_research.evaluation.metrics import classification_metrics, regression_metrics
from microscopy_cv_research.models.backbones import EncoderWithHead, create_backbone
from microscopy_cv_research.models.encoder_registry import get_encoder_spec
from microscopy_cv_research.models.heads import ClassificationHead, RegressionHead
from microscopy_cv_research.training.engine import get_device, run_supervised_epoch, save_checkpoint, save_json
from microscopy_cv_research.utils.repro import set_seed


def _build_transforms(image_size: int):
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()])
    eval_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return train_transform, eval_transform


def plan_supervised_experiment(config_path: str | Path) -> dict:
    config = load_config(config_path)
    encoder = get_encoder_spec(config["encoder_name"])
    dataset_path = Path(config["dataset_csv"])
    report = {
        "experiment_name": config["experiment_name"],
        "task_type": config["task_type"],
        "encoder": asdict(encoder),
        "group_column": config["group_column"],
        "target_column": config["target_column"],
        "dataset_csv": str(dataset_path),
        "dataset_present": dataset_path.exists(),
    }
    if not dataset_path.exists():
        report["next_action"] = "Add labels.csv and image paths to enable training."
        return report
    table = pd.read_csv(dataset_path)
    train_df, val_df, test_df = make_group_train_val_test_split(table, group_column=config["group_column"], random_state=config["seed"])
    report["num_train_samples"] = len(train_df)
    report["num_val_samples"] = len(val_df)
    report["num_test_samples"] = len(test_df)
    return report


def run_supervised_experiment(config_path: str | Path) -> dict:
    config = load_config(config_path)
    set_seed(config["seed"])
    table = pd.read_csv(config["dataset_csv"])
    train_df, val_df, test_df = make_group_train_val_test_split(table, group_column=config["group_column"], random_state=config["seed"])
    train_transform, eval_transform = _build_transforms(config["image_size"])
    device = get_device()
    encoder_model, embedding_dim, backbone_source = create_backbone(config["encoder_name"], pretrained=config.get("pretrained", False))

    if config.get("freeze_encoder", False):
        for parameter in encoder_model.parameters():
            parameter.requires_grad = False

    if config["task_type"] == "classification":
        label_encoder = LabelEncoder()
        label_encoder.fit(table[config["target_column"]])
        train_df = train_df.assign(**{config["target_column"]: label_encoder.transform(train_df[config["target_column"]]).astype("int64")})
        val_df = val_df.assign(**{config["target_column"]: label_encoder.transform(val_df[config["target_column"]]).astype("int64")})
        test_df = test_df.assign(**{config["target_column"]: label_encoder.transform(test_df[config["target_column"]]).astype("int64")})
        head = ClassificationHead(embedding_dim, config["num_classes"])
        criterion = nn.CrossEntropyLoss()
    else:
        label_encoder = None
        head = RegressionHead(embedding_dim)
        criterion = nn.MSELoss()

    model = EncoderWithHead(encoder_model, head).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_dataset = MicroscopyImageDataset(train_df, config["image_root"], config["target_column"], train_transform)
    val_dataset = MicroscopyImageDataset(val_df, config["image_root"], config["target_column"], eval_transform)
    test_dataset = MicroscopyImageDataset(test_df, config["image_root"], config["target_column"], eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    history = []
    best_metric = float("-inf")
    best_state = None
    for epoch in range(1, config["epochs"] + 1):
        train_loss, _, _ = run_supervised_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = run_supervised_epoch(model, val_loader, criterion, None, device)
        metrics = classification_metrics(val_targets, val_preds) if config["task_type"] == "classification" else regression_metrics(val_targets, val_preds)
        primary_metric = metrics.get("macro_f1", metrics.get("r2", float("-inf")))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})
        if primary_metric > best_metric:
            best_metric = primary_metric
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_preds, test_targets = run_supervised_epoch(model, test_loader, criterion, None, device)
    test_metrics = classification_metrics(test_targets, test_preds) if config["task_type"] == "classification" else regression_metrics(test_targets, test_preds)
    report = {
        "experiment_name": config["experiment_name"],
        "task_type": config["task_type"],
        "encoder": asdict(get_encoder_spec(config["encoder_name"])),
        "runtime_backbone_source": backbone_source,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "history": history,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "label_mapping": None if label_encoder is None else {int(i): label for i, label in enumerate(label_encoder.classes_)},
    }
    save_checkpoint({"model_state": model.state_dict(), "config": config, "report": report}, config["save_path"])
    save_json(report, config["report_path"])
    return report
