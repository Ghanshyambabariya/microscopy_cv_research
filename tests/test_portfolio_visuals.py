from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_visual_assets_are_valid_svg() -> None:
    expected = [
        ROOT / "assets/system_architecture.svg",
        ROOT / "assets/active_learning_loop.svg",
        ROOT / "assets/portfolio_evidence_map.svg",
    ]
    for path in expected:
        assert path.exists(), f"Missing visual asset: {path}"
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
