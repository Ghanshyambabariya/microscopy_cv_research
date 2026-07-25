from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_visual_assets_are_valid_svg() -> None:
    expected = {
        ROOT / "assets/system_architecture.svg": (1400, 760),
        ROOT / "assets/active_learning_loop.svg": (1400, 460),
        ROOT / "assets/portfolio_evidence_map.svg": (1400, 560),
    }
    for path, minimum_size in expected.items():
        assert path.exists(), f"Missing visual asset: {path}"
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        width = int(root.attrib["width"])
        height = int(root.attrib["height"])
        assert width >= minimum_size[0]
        assert height >= minimum_size[1]


def test_portfolio_visual_text_is_current() -> None:
    evidence_map = (ROOT / "assets/portfolio_evidence_map.svg").read_text(encoding="utf-8")
    assert "13 tests" in evidence_map
    assert "11 tests" not in evidence_map
