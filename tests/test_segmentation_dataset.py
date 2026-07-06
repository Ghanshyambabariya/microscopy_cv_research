import numpy as np
import torch
from pathlib import Path
from PIL import Image

from microscopy_cv_research.data.segmentation import SegmentationSample, SemSegmentationDataset
from microscopy_cv_research.evaluation.metrics import segmentation_metrics
from microscopy_cv_research.training.segmentation import compute_class_weights


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_threshold_and_mask_map(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.array([[0, 1, 2, 255] * 1] * 4, dtype=np.uint8)
    image_path = tmp_path / "im.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(img).save(image_path)
    _write_mask(mask_path, mask)

    samples = [SegmentationSample(image_path=image_path, mask_path=mask_path, dataset_name="dummy", split="train")]
    ds = SemSegmentationDataset(samples, threshold=1, mask_map={255: 0, 2: 1}, ignore_index=255, image_size=4)
    item = ds[0]
    mask_tensor = item["mask"]
    assert set(mask_tensor.numpy().reshape(-1).tolist()) == {0, 1}


def test_compute_class_weights_handles_imbalance():
    class DummyDS:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            if idx == 0:
                mask = torch.zeros((2, 2), dtype=torch.long)
            else:
                mask = torch.zeros((2, 2), dtype=torch.long)
                mask[0, 0] = 1
            return {"mask": mask}

    weights = compute_class_weights(DummyDS(), num_classes=2)
    assert weights.shape[0] == 2
    assert weights[1] > weights[0], "Minority class should receive higher weight"


def test_segmentation_metrics_ignore_index():
    y_true = np.array([[0, 1, 255]])
    y_pred = np.array([[0, 1, 0]])
    metrics = segmentation_metrics(y_true, y_pred, num_classes=2, ignore_index=255)
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["mean_iou_fg"] == 1.0
