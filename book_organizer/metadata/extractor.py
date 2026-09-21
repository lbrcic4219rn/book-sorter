import re
from pathlib import Path

from book_organizer.metadata.calibre import get_raw_metadata
from book_organizer.metadata.models import BookMetadata
from book_organizer.metadata.transliterate import transliterate_cyrillic, remove_titles_and_expand_initials

ISBN_RE = re.compile(
    r"(97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d|\d{9}[\dXx])"
)


def extract_metadata(path: Path) -> BookMetadata:
    raw = get_raw_metadata(path)
    meta = BookMetadata(path=path)

    for line in raw.splitlines():
        if line.startswith("Title"):
            title = line.split(":", 1)[1].strip()
            meta.title = transliterate_cyrillic(title)
        elif line.startswith("Author(s)"):
            authors = line.split(":", 1)[1].strip().split(" [")[0]
            authors = transliterate_cyrillic(authors)
            # Remove titles and expand abbreviations from extracted author names
            if authors:
                meta.authors = remove_titles_and_expand_initials(authors)
        elif "isbn" in line.lower():
            m = ISBN_RE.search(line)
            if m:
                meta.isbn = m.group(0).replace("-", "").replace(" ", "")

    return meta
