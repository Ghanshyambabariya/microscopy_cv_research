from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK_PATTERNS = [
    re.compile(r"!\[[^\]]*\]\(([^)]+)\)"),
    re.compile(r"(?<!!)(?<!\])\[[^\]]+\]\(([^)]+)\)"),
]


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:")) or target.startswith("<")


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md"]
    markdown_files.extend((ROOT / "docs").rglob("*.md"))
    markdown_files.extend((ROOT / "projects").rglob("*.md"))

    missing: list[str] = []
    for markdown_path in markdown_files:
        text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in MARKDOWN_LINK_PATTERNS:
            for match in pattern.finditer(text):
                target = match.group(1).split("#", 1)[0]
                if not target or _is_external(target):
                    continue
                if not (markdown_path.parent / target).resolve().exists():
                    rel = markdown_path.relative_to(ROOT).as_posix()
                    missing.append(f"{rel} -> {target}")

    assert not missing, "Broken local markdown links/images:\n" + "\n".join(missing)
