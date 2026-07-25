from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_project_index_lists_separate_projects_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    projects = (ROOT / "projects" / "README.md").read_text(encoding="utf-8")
    public_text = readme + "\n" + projects

    assert "01 Microscopy CV" in public_text
    assert "02 Process Signal ML" in public_text
    assert "03 Tool-Wear Benchmarks" in public_text
    assert "04 Materials Property ML" in public_text
    assert "05 Multimodal Platform" not in public_text
    assert "combined benchmark" not in public_text.lower()


def test_public_markdown_avoids_guidance_language() -> None:
    markdown_files = [ROOT / "README.md"]
    markdown_files.extend((ROOT / "docs").rglob("*.md"))
    markdown_files.extend((ROOT / "projects").rglob("*.md"))

    blocked = [
        "If the role is about",
        "Share this project page",
        "Application Links",
        "CV Ready",
        "portfolio-style",
        "recruiter",
        "Short CV",
    ]

    hits: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for phrase in blocked:
            if phrase.lower() in text.lower():
                hits.append(f"{path.relative_to(ROOT).as_posix()}: {phrase}")

    assert not hits, "Public documentation contains guidance-style wording:\n" + "\n".join(hits)
