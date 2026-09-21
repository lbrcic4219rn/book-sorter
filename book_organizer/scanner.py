from pathlib import Path
from typing import List


def scan_books(directory: Path, extensions: set[str]) -> List[Path]:
    books = []

    for file in directory.rglob("*"):
        if file.is_dir(): continue
        if file.suffix.lower() in extensions:
            books.append(file)

    return books
