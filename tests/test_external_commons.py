from __future__ import annotations

from pathlib import Path

from PIL import Image

from microscopy_cv_research.training.external_commons import build_manifest


def test_build_commons_manifest_infers_folder_labels(tmp_path) -> None:
    for label in ["smooth", "rough"]:
        folder = tmp_path / label
        folder.mkdir()
        Image.new("RGB", (8, 8), color="white").save(folder / f"{label}_sample.png")
    manifest = build_manifest(tmp_path, {"manifest_path": str(tmp_path / "manifest.csv"), "max_images": 10})
    assert sorted(manifest["label"].unique().tolist()) == ["rough", "smooth"]

