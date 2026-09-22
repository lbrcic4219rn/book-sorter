"""Detect a file's real format from its bytes (extensions in this library are often wrong)
and unpack archives so the books inside can be scanned."""
import hashlib
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path

BOOK_FORMATS = {"epub", "pdf", "mobi", "doc", "docx", "rtf", "html", "txt"}
ARCHIVE_FORMATS = {"rar", "zip", "7z"}
_TEXT_EXTS = {".txt": "txt", ".htm": "html", ".html": "html", ".xhtml": "html"}
_SKIP_NAMES = {"metadata.opf", "filelist.txt", ".ds_store", "desktop.ini", "thumbs.db"}


@lru_cache(maxsize=None)
def detect(path: Path) -> str | None:
    if path.name.lower() in _SKIP_NAMES:
        return None
    try:
        with path.open("rb") as f:
            head = f.read(4096)
    except OSError:
        return None
    if head[60:68] in (b"BOOKMOBI", b"TEXtREAd"):
        return "mobi"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return "doc"
    if head.startswith(b"Rar!"):
        return "rar"
    if head.startswith(b"7z\xbc\xaf"):
        return "7z"
    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
                if "mimetype" in names and b"epub" in zf.read("mimetype"):
                    return "epub"
                if "word/document.xml" in names:
                    return "docx"
        except (zipfile.BadZipFile, OSError, KeyError):
            return None
        return "zip"
    ext = _TEXT_EXTS.get(path.suffix.lower())
    if ext and b"\x00" not in head:
        return ext
    return None


def extract(archive: Path, root: Path) -> list[Path]:
    """Unpack once into root/<hash>/; later runs reuse the folder. Returns the books inside."""
    st = archive.stat()
    tag = hashlib.sha1(f"{archive}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    dest = root / tag
    if not dest.exists():
        tmp = root / (tag + ".partial")
        subprocess.run(["unar", "-q", "-f", "-D", "-o", str(tmp), str(archive)],
                       capture_output=True, timeout=300)
        tmp.mkdir(parents=True, exist_ok=True)
        tmp.rename(dest)
    return sorted(p for p in dest.rglob("*") if p.is_file() and detect(p) in BOOK_FORMATS)
