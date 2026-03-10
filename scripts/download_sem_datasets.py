from __future__ import annotations

import subprocess
from pathlib import Path


REPOS = {
    "emps": "https://github.com/by256/emps.git",
    "automatic-sem-image-segmentation": "https://github.com/BAMresearch/automatic-sem-image-segmentation.git",
    "MudrockNet": "https://github.com/abhishekdbihani/MudrockNet.git",
}


def clone_repo(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "external"
    for name, url in REPOS.items():
        clone_repo(url, root / name)


if __name__ == "__main__":
    main()
