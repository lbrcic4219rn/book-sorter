import re
import subprocess
from pathlib import Path

from book_organizer.metadata.models import BookMetadata
from book_organizer.metadata.transliterate import transliterate_cyrillic


def sanitize(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return re.sub(r"\s+", " ", name).strip()


def build_filename(meta: BookMetadata) -> str:
    author = meta.authors or "Unknown"
    title = meta.title or meta.path.stem
    # Transliterate Cyrillic characters to Latin
    author = transliterate_cyrillic(author)
    title = transliterate_cyrillic(title)
    return sanitize(f"{author} - {title}{meta.path.suffix}")


def apply_metadata(meta: BookMetadata) -> None:
    """Write corrected title/author/isbn/genre back into the file via ebook-meta."""
    cmd = ["ebook-meta", str(meta.path)]
    if meta.title:
        cmd += ["--title", meta.title]
    if meta.authors:
        cmd += ["--authors", meta.authors]
    if meta.isbn:
        cmd += ["--identifier", f"isbn:{meta.isbn}"]
    if meta.genre:
        cmd += ["--tags", meta.genre]
    subprocess.run(cmd, capture_output=True, text=True)


def rename_book(meta: BookMetadata) -> Path:
    """Rename the file on disk to a clean 'Author - Title.ext' pattern.
    Returns the new path. meta.path is NOT mutated here — caller decides
    whether to update it."""
    new_path = meta.path.with_name(build_filename(meta))
    meta.path.rename(new_path)
    return new_path