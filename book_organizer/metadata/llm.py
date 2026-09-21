import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from shutil import which

import requests

from book_organizer.metadata.models import BookMetadata
from book_organizer.metadata.transliterate import transliterate_cyrillic, remove_titles_and_expand_initials

OLLAMA_URL = "http://localhost:11434/api/chat"


def call_ollama(prompt: str, model: str) -> str:
    """Call a local Ollama model (must be running: `ollama serve`) and return its reply."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "think": False,  # skip reasoning overhead on hybrid models — not needed for extraction
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(
            f"Could not reach Ollama at {OLLAMA_URL} ({e}).\n"
            f"Make sure `ollama serve` is running and you've pulled the model: "
            f"`ollama pull {model}`"
        )
    return resp.json()["message"]["content"]


def _extract_genre(data: dict) -> str | None:
    """Models don't always respect the exact key name ('genre' vs 'genres'),
    and sometimes return a list instead of a string. Normalize both."""
    genre = data.get("genre", data.get("genres"))
    if isinstance(genre, list):
        genre = ", ".join(str(g) for g in genre if g)
    return genre or None


def guess_from_filename(path: Path, model: str) -> BookMetadata:
    """Ask the local LLM to guess title/author/genre from a messy filename."""
    prompt = (
        "You are cleaning up a messy ebook filename (it may contain "
        "underscores, dots, release-group tags, years, or other junk). "
        "Guess the most likely real book title, author, and genre.\n"
        'Respond ONLY with JSON: {"title": "...", "authors": "...", "genre": "..."}. '
        "Use null for anything you can't confidently determine.\n\n"
        f"Filename: {path.stem}"
    )
    raw = call_ollama(prompt, model)
    try:
        data = json.loads(raw)
        title = data.get("title")
        authors = data.get("authors")
        # Transliterate Cyrillic to Latin
        if title:
            title = transliterate_cyrillic(title)
        if authors:
            authors = transliterate_cyrillic(authors)
            # Remove titles and expand abbreviations
            authors = remove_titles_and_expand_initials(authors)
        return BookMetadata(
            path=path,
            title=title,
            authors=authors,
            genre=_extract_genre(data),
        )
    except (json.JSONDecodeError, AttributeError):
        return BookMetadata(path=path)


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _extract_epub_snippet(path: Path, max_chars: int) -> str:
    """EPUB is just a zip of XHTML — read the first few spine docs directly,
    far faster than a full conversion."""
    try:
        with zipfile.ZipFile(path) as zf:
            html_files = sorted(
                n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))
            )
            text = ""
            for name in html_files[:4]:  # title/copyright page is usually in the first few docs
                text += _strip_html(zf.read(name).decode("utf-8", errors="ignore"))
                if len(text) >= max_chars:
                    break
            return text[:max_chars]
    except (zipfile.BadZipFile, KeyError):
        return ""


def _extract_pdf_snippet(path: Path, max_pages: int = 5) -> str:
    pdftotext = which("pdftotext")
    if not pdftotext:
        return ""
    try:
        result = subprocess.run(
            [pdftotext, "-l", str(max_pages), str(path), "-"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout[:3000]
    except subprocess.TimeoutExpired:
        return ""


def _extract_via_calibre_convert(path: Path, max_chars: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "snippet.txt"
        try:
            subprocess.run(
                ["ebook-convert", str(path), str(out_path)],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return ""
        if out_path.exists():
            return out_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    return ""


def extract_text_snippet(path: Path, max_chars: int = 3000) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_snippet(path)
    if suffix == ".epub":
        return _extract_epub_snippet(path, max_chars)
    return _extract_via_calibre_convert(path, max_chars)  # mobi, azw3, azw, etc.


def guess_from_text(path: Path, text: str, model: str) -> BookMetadata:
    """Ask the local LLM to identify a book from extracted early-page text."""
    prompt = (
        "Below is text extracted from the first pages of a book (may include "
        "a cover, title page, or copyright page). Identify the book's real "
        "title, author, and genre.\n"
        'Respond ONLY with JSON: {"title": "...", "authors": "...", "genre": "..."}. '
        "Use null if unclear.\n\n"
        f"{text}"
    )
    raw = call_ollama(prompt, model)
    try:
        data = json.loads(raw)
        title = data.get("title")
        authors = data.get("authors")
        # Transliterate Cyrillic to Latin
        if title:
            title = transliterate_cyrillic(title)
        if authors:
            authors = transliterate_cyrillic(authors)
            # Remove titles and expand abbreviations
            authors = remove_titles_and_expand_initials(authors)
        return BookMetadata(
            path=path,
            title=title,
            authors=authors,
            genre=_extract_genre(data),
        )
    except (json.JSONDecodeError, AttributeError):
        return BookMetadata(path=path)


def verify_match(old: BookMetadata, new: BookMetadata, model: str) -> bool:
    """Ask the local LLM whether two BookMetadata records plausibly describe the same book."""
    prompt = (
        "Do these two book records plausibly refer to the SAME book, allowing "
        "for reformatting, abbreviated names, translation, or missing details?\n\n"
        f"Record A: title={old.title!r}, author={old.authors!r}\n"
        f"Record B: title={new.title!r}, author={new.authors!r}\n\n"
        'Respond ONLY with JSON: {"same_book": true} or {"same_book": false}.'
    )
    raw = call_ollama(prompt, model)
    try:
        return bool(json.loads(raw).get("same_book"))
    except (json.JSONDecodeError, AttributeError):
        return False