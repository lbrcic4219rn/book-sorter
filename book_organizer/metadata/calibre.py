import subprocess
from pathlib import Path


def get_raw_metadata(path: Path) -> str:
    result = subprocess.run(
        ["ebook-meta", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout