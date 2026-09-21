from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BookMetadata:
    path: Path
    title: str | None = None
    authors: str | None = None
    isbn: str | None = None
    genre: str | None = None