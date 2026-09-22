"""Candidate author names from three sources: filename (+folder), tags, first pages."""
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from shutil import which

from booksorter.formats import detect
from booksorter.mobi import mobi_text
from booksorter.names import display_clean, is_title_like, split_names, looks_like_name, transliterate

_BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_NOISE = re.compile(r"\b(19|20)\d{2}\b|\b\d+\b|\+")


def _clean_stem(stem: str) -> str:
    stem = re.sub(r"\b([OoDd])_(?=[A-Z])", r"\1'", transliterate(stem))  # O_Brien -> O'Brien
    stem = _BRACKETS.sub(" ", stem).replace(" _ ", " & ").replace("_", " ")
    return re.sub(r"\s+", " ", _NOISE.sub(" ", stem)).strip(" -.")


def book_stem(path: Path) -> str:
    """Filename without extension; a 'suffix' like '.Jerome-tri čovjeka u čamcu' is part of the name."""
    suffix = path.suffix
    return path.name if len(suffix) > 6 or " " in suffix[1:].strip() and len(suffix) > 6 else path.stem


def from_filename(path: Path) -> list[str]:
    stem = _clean_stem(book_stem(path))
    if " - " in stem or " -" in stem or "- " in stem:
        parts = re.split(r"\s*-\s+|\s+-\s*", stem)
    elif stem.count("-") == 1:
        parts = stem.split("-")
    elif "-" in stem:  # 'Samanta-Jang-Jamajska-staza' — guess name size
        words = stem.split("-")
        parts = [" ".join(words[:2]), " ".join(words[:3]), " ".join(words[-2:]), " ".join(words[-3:])]
    else:
        parts = [stem]
    # 'Author. Title' / 'Title, Author' fallbacks for a single chunk
    if len(parts) == 1 and "." in stem:
        parts = stem.split(".", 1)
    if len(parts) == 1 and "," in stem:
        parts = [p for p in stem.split(",") if p.strip()]
    out = []
    for part in parts:
        for p in split_names(part):
            p = display_clean(p)
            if looks_like_name(p) and p not in out:
                out.append(p)
    return out


def from_folder(path: Path) -> list[str]:
    name = display_clean(_clean_stem(path.parent.name))
    return [name] if looks_like_name(name) else []


def _split_authors(s: str) -> list[str]:
    return split_names(s)


def author_groups(path: Path, tags: dict) -> list[list[str]]:
    """Co-author sets as written in the filename or tags, in their original order."""
    raw = re.split(r"\s+-\s+", _clean_stem(book_stem(path))) + [tags.get("authors", "")]
    return [g for g in (split_names(x) for x in raw) if len(g) > 1]


def read_tags(path: Path) -> dict:
    """Title/author from embedded metadata via Calibre's ebook-meta."""
    if detect(path) in ("doc", "txt", None):  # no usable metadata
        return {}
    try:
        out = subprocess.run(["ebook-meta", str(path)], capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, OSError):
        return {}
    tags = {}
    for line in out.splitlines():
        field, _, value = line.partition(":")
        field, value = field.strip(), value.strip()
        if field == "Title":
            tags["title"] = value
        elif field == "Author(s)":
            tags["authors"] = re.sub(r"\s*\[[^\]]*\]", "", value)  # drop sort-name
    return tags


def from_tags(tags: dict) -> list[str]:
    """Tag authors, minus the common PDF mistake of the title stored as author."""
    title = tags.get("title", "")
    out = []
    for a in _split_authors(tags.get("authors", "")):
        a = display_clean(a)
        if looks_like_name(a) and not is_title_like(a, title):
            out.append(a)
    return out


# ---------- first pages ----------

def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _epub_text(path: Path, n: int) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            opf = next((x for x in zf.namelist() if x.endswith(".opf")), None)
            docs = []
            if opf:  # follow the spine order
                content = zf.read(opf).decode("utf-8", "ignore")
                base = opf.rsplit("/", 1)[0] + "/" if "/" in opf else ""
                hrefs = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', content))
                hrefs |= {i: h for h, i in re.findall(r'<item[^>]*href="([^"]+)"[^>]*id="([^"]+)"', content)}
                docs = [base + hrefs[i] for i in re.findall(r'<itemref[^>]*idref="([^"]+)"', content) if i in hrefs]
            docs = [d for d in docs if d in zf.namelist()] or sorted(
                x for x in zf.namelist() if x.lower().endswith((".xhtml", ".html", ".htm")))
            text = ""
            for d in docs[:8]:
                text += " " + _strip_html(zf.read(d).decode("utf-8", "ignore"))
                if len(text) >= n:
                    break
            return text[:n]
    except (zipfile.BadZipFile, KeyError, OSError):
        return ""


def _pdf_text(path: Path, n: int) -> str:
    if not which("pdftotext"):  # brew install poppler — much faster than calibre
        return _calibre_text(path, n)
    try:
        r = subprocess.run(["pdftotext", "-l", "5", str(path), "-"], capture_output=True, text=True, timeout=60)
        return r.stdout[:n]
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _calibre_text(path: Path, n: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.txt"
        try:
            subprocess.run(["ebook-convert", str(path), str(out)], capture_output=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError):
            return ""
        return out.read_text("utf-8", "ignore")[:n] if out.exists() else ""


def _textutil_text(path: Path, n: int) -> str:
    """macOS textutil reads .doc/.docx/.rtf/.html quickly (Calibre can't open .doc at all)."""
    try:
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                           capture_output=True, timeout=60)
        return r.stdout.decode("utf-8", "ignore")[:n]
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _plain_text(path: Path, n: int) -> str:
    raw = path.read_bytes()[: n * 4]
    for enc in ("utf-8", "cp1250", "cp1251"):
        try:
            return raw.decode(enc)[:n]
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")[:n]


def first_pages(path: Path, n: int = 5000) -> str:
    fmt = detect(path)
    if fmt == "epub":
        return _epub_text(path, n)
    if fmt == "pdf":
        return _pdf_text(path, n)
    if fmt == "mobi":
        text = _strip_html(mobi_text(path))
        if len(text) > 200:
            return text[:n]
    if fmt in ("doc", "docx", "rtf", "html") and which("textutil"):
        return re.sub(r"\s+", " ", _textutil_text(path, n * 2)).strip()[:n]
    if fmt == "txt":
        return _plain_text(path, n)
    return _calibre_text(path, n)


# ---------- copyright / impressum page ----------

_ORIGINAL = re.compile(
    r"(?:naslov (?:originala|izvornika|orginala)|originalni naslov|izvorni naslov|original title|"
    r"title of the original)\s*[:.]?\s*(.{0,120})", re.I)
_COPYRIGHT = re.compile(r"(?:copyright|©|\(c\))\s*(?:©\s*)?(?:\d{4}\s*)?(?:by\s+)?(.{0,80})", re.I)
_WORD = r"(?:[A-ZČĆŠŽĐ]\.|[A-ZČĆŠŽĐ][a-zčćšžđäöüéèáíóúñ'’-]+)"
_NAME_RUN = re.compile(rf"{_WORD}(?:\s+(?:{_WORD}|de|la|van|von|der|le|di|da|du)){{1,3}}")
_STOP = re.compile(r"\b(Copyright|Prevela|Preveo|Prevod|Naslov|All|Sva|Svako|Za|Izdavač\w*|Urednik|Urednica|Glavni|Direktor|Štampa|Tiraž|Published|by)\b.*", re.S)


def from_impressum(snippet: str) -> list[str]:
    """Author names from 'Naslov originala: Runaway / Alice Munro' and '© 2004 by Alice Munro'.
    Serbian editions print these in the original spelling, so they beat transcriptions."""
    text = transliterate(snippet)
    out = []
    for pattern in (_ORIGINAL, _COPYRIGHT):
        for m in pattern.finditer(text):
            chunk = m.group(1)
            if len(chunk) >= 80:  # hit the length cap: drop the possibly cut-off last word
                chunk = chunk.rsplit(" ", 1)[0]
            for name in _NAME_RUN.findall(_STOP.sub("", chunk)):
                name = display_clean(re.sub(r"\s+\d{4}.*$", "", name))
                if looks_like_name(name) and name not in out:
                    out.append(name)
    return out
