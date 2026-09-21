import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from book_organizer.metadata.models import BookMetadata

ISBN_RE = re.compile(
    r"(97[89][-\s]?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d|\d{9}[\dXx])"
)

OPF_NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "opf": "http://www.idpf.org/2007/opf",
}


def find_isbn_in_filename(path: Path) -> str | None:
    m = ISBN_RE.search(path.stem)
    return m.group(0).replace("-", "").replace(" ", "") if m else None


def fetch_online_metadata(
    path: Path,
    isbn: str | None = None,
    title: str | None = None,
    authors: str | None = None,
) -> BookMetadata | None:
    """Query online sources via Calibre's fetch-ebook-metadata and return
    the best match, or None if nothing usable came back."""
    cmd = ["fetch-ebook-metadata", "--opf"]
    if isbn:
        cmd += ["--isbn", isbn]
    else:
        if title:
            cmd += ["--title", title]
        if authors:
            cmd += ["--authors", authors]
    if len(cmd) == 2:  # nothing to search on
        return None

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    return _parse_opf(result.stdout, path, fallback_isbn=isbn)


def _parse_opf(opf_xml: str, path: Path, fallback_isbn: str | None) -> BookMetadata | None:
    try:
        root = ET.fromstring(opf_xml)
    except ET.ParseError:
        return None

    title_el = root.find(".//dc:title", OPF_NS)
    author_el = root.find(".//dc:creator", OPF_NS)
    isbn_el = root.find(".//dc:identifier[@opf:scheme='ISBN']", OPF_NS)
    subject_els = root.findall(".//dc:subject", OPF_NS)
    genre = ", ".join(s.text for s in subject_els if s.text) or None

    return BookMetadata(
        path=path,
        title=title_el.text if title_el is not None else None,
        authors=author_el.text if author_el is not None else None,
        isbn=isbn_el.text if isbn_el is not None else fallback_isbn,
        genre=genre,
    )