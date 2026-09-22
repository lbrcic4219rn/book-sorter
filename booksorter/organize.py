"""Stage 2: put each confident book into <output>/<Author>/ as EPUB (PDF stays PDF),
write the author into the file's tags, and park the original in <output>/_Originals/.
Nothing is ever deleted; every step is logged so `undo` can reverse it."""
import json
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

from booksorter import formats
from booksorter.sources import book_stem, first_pages

KEEP_AS_IS = {"epub", "pdf"}
CONVERTIBLE = {"mobi", "doc", "docx", "rtf", "html", "txt"}


@dataclass
class Job:
    src: Path             # the book file to process (may be inside .booksorter_extracted/)
    original: Path        # what gets parked in _Originals (the archive, for archived books)
    author: str
    coauthors: list[str]
    title: str = ""


def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return re.sub(r"\s+", " ", name) or "Unknown"


def clean_stem(path: Path) -> str:
    """Filename without extension and download junk ('+', '(1)', '.original_epub')."""
    stem = book_stem(path)
    stem = re.sub(r"\.original_epub$", "", stem)
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    return safe_name(stem.rstrip(" +").strip(" .-_"))


_RESERVED: set[Path] = set()
_RESERVE_LOCK = threading.Lock()


def _unique(path: Path) -> Path:
    """A free file name, reserved so parallel conversions can't pick the same one."""
    with _RESERVE_LOCK:
        n, out = 1, path
        while out.exists() or out in _RESERVED:
            out = path.with_name(f"{path.stem} ({n}){path.suffix}")
            n += 1
        _RESERVED.add(out)
        return out


def _taken(path: Path) -> bool:
    with _RESERVE_LOCK:
        return path.exists() or path in _RESERVED


def convert_to_epub(src: Path, dest: Path) -> str | None:
    """Convert with Calibre; .doc goes through macOS textutil first (Calibre can't read .doc).
    Returns an error message, or None on success."""
    fmt = formats.detect(src)
    with tempfile.TemporaryDirectory() as tmp:
        source = src
        if fmt == "doc":
            source = Path(tmp) / "book.docx"
            r = subprocess.run(["textutil", "-convert", "docx", "-output", str(source), str(src)],
                               capture_output=True, text=True, timeout=300)
            if r.returncode or not source.exists():
                return f"textutil failed: {r.stderr.strip()[:200]}"
        elif src.suffix.lower() not in {".mobi", ".azw", ".azw3", ".prc", ".docx", ".rtf", ".htm", ".html", ".txt"}:
            # broken extension ('. mobi', no extension): Calibre picks the reader by extension
            ext = {"mobi": ".mobi", "docx": ".docx", "rtf": ".rtf", "html": ".html", "txt": ".txt"}[fmt]
            source = Path(tmp) / f"book{ext}"
            shutil.copy2(src, source)
        out = Path(tmp) / "out.epub"
        try:
            r = subprocess.run(["ebook-convert", str(source), str(out)],
                               capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return "ebook-convert timed out"
        if r.returncode or not out.exists():
            return f"ebook-convert failed: {(r.stderr or r.stdout).strip()[-200:]}"
        problem = verify_epub(out)
        if problem:
            return problem
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), str(dest))
    return None


def verify_epub(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as zf:
            if zf.testzip() is not None:
                return "converted EPUB is corrupt"
    except zipfile.BadZipFile:
        return "converted EPUB is not a valid zip"
    if len(first_pages(path, 2000).strip()) < 200:
        return "converted EPUB has almost no text"
    return None


def write_tags(path: Path, authors: list[str], title: str = "") -> str | None:
    cmd = ["ebook-meta", str(path), "--authors", " & ".join(authors)]
    if title:
        cmd += ["--title", title]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return None if r.returncode == 0 else f"ebook-meta failed: {r.stderr.strip()[:200]}"


def process(job: Job, out_root: Path, library: Path) -> dict:
    """Place one book. Returns a log entry (status 'done' / 'duplicate' / 'failed')."""
    fmt = formats.detect(job.src)
    author_dir = out_root / safe_name(job.author)
    # 'Author - Title', or the cleaned old name if no title could be found
    stem = safe_name(f"{job.author} - {job.title}") if job.title else clean_stem(job.src)
    target_ext = ".pdf" if fmt == "pdf" else ".epub"
    dest = author_dir / f"{stem}{target_ext}"
    entry = {"src": str(job.src), "original": str(job.original), "author": job.author, "format": fmt}

    if _taken(dest):  # same book already placed (e.g. .mobi and .epub of one title)
        dest = out_root / "_Duplicates" / safe_name(job.author) / f"{stem}{target_ext}"
        entry["status"] = "duplicate"
    dest = _unique(dest)

    if fmt in KEEP_AS_IS:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job.src, dest)
    elif fmt in CONVERTIBLE:
        error = convert_to_epub(job.src, dest)
        if error:
            return entry | {"status": "failed", "error": error}
    else:
        return entry | {"status": "failed", "error": f"unsupported format {fmt}"}

    error = write_tags(dest, [job.author, *job.coauthors], job.title)
    if error:
        entry["tag_error"] = error
    entry["dest"] = str(dest)
    entry.setdefault("status", "done")
    return entry


def park_original(original: Path, out_root: Path, library: Path) -> Path | None:
    """Move the original into _Originals, keeping its path relative to the library."""
    if not original.exists():
        return None  # already parked (archive with several books)
    try:
        rel = original.relative_to(library)
    except ValueError:
        rel = Path(original.name)
    parked = _unique(out_root / "_Originals" / rel)
    parked.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(original), str(parked))
    return parked


def log_line(log: Path, entry: dict) -> None:
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def placed_sources(log: Path) -> set[str]:
    """Every book file that has been placed so far, according to the log."""
    if not log.exists():
        return set()
    out = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        e = json.loads(line) if line.strip() else {}
        if e.get("dest") and Path(e["dest"]).exists():
            out.add(e["src"])
    return out


def undo(log: Path) -> tuple[int, int]:
    """Reverse a run: originals go back to where they were, created files are removed."""
    if not log.exists():
        return 0, 0
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    restored = removed = 0
    for e in reversed(entries):
        parked, original = e.get("parked"), e.get("original")
        if parked and Path(parked).exists() and not Path(original).exists():
            Path(original).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(parked, original)
            restored += 1
        dest = e.get("dest")
        if dest and Path(dest).exists():
            Path(dest).unlink()  # a file this tool created (converted/copied); the original is back
            removed += 1
    # remove folders this left empty (author folders, _Originals/…), never the output root itself
    for d in sorted((p for p in log.parent.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        junk = [f for f in d.iterdir() if f.name == ".DS_Store"]
        if all(f.name == ".DS_Store" for f in d.iterdir()):
            for f in junk:
                f.unlink()
            d.rmdir()
    log.rename(log.with_suffix(".undone.log"))
    return restored, removed


def prune_empty_dirs(root: Path) -> int:
    """Delete folders under root that hold nothing (or only .DS_Store / Thumbs.db). Files are
    never touched; root itself is kept."""
    removed = 0
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        entries = list(d.iterdir())
        if all(e.is_file() and e.name in (".DS_Store", "Thumbs.db") for e in entries):
            for e in entries:
                e.unlink()
            d.rmdir()
            removed += 1
    return removed
