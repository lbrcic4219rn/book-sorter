"""Pick a clean book title from the filename (minus the author) and the title tag."""
import difflib
import re
from pathlib import Path

from booksorter.names import fold, same_person, split_names, tokens, transliterate
from booksorter.sources import book_stem

_JUNK = [
    (re.compile(r"^\d{6,}[-_ ]"), ""),                        # '112881508-John-Grisham-Tvrtka'
    (re.compile(r"\.original_epub$", re.I), ""),
    (re.compile(r"\s*\(\d+\)\s*$"), ""),                       # 'Knjiga (1)'
    (re.compile(r"[\s+$]+$"), ""),                             # trailing '+', '$'
    (re.compile(r"\s*\d+\s*str\.?$", re.I), ""),               # '52str'
    (re.compile(r"[\s\-_(\[]*\b(lat|cir|latinica|cirilica|ćirilica)\b[)\]]?$", re.I), ""),  # '- lat'
    (re.compile(r"^microsoft word\s*-\s*", re.I), ""),         # PDF title tags
    (re.compile(r"\s*[-–]?\s*(www\.\S+|https?://\S+)", re.I), ""),  # ' - www.sftim.com'
    (re.compile(r"\.(docx?|rtf|pdf|txt|html?)$", re.I), ""),
]


def _tidy(s: str) -> str:
    s = transliterate(s).replace(" _ ", " & ").replace("_", " ")
    for pattern, repl in _JUNK:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip(" -.,")
    letters = [c for c in s if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) > 3:  # 'POKONDIRENA TIKVA'
        s = s[0] + s[1:].lower()
    return s


def _is_author(part: str, authors: list[str]) -> bool:
    pieces = split_names(part)  # 'Tina Fras i Boris Rasheta': a group of names with one of ours in it
    if len(pieces) > 1:
        return any(_is_author(p, authors) for p in pieces)
    pt = set(tokens(part))
    return bool(pt) and any(same_person(part, a) or pt <= set(tokens(a)) for a in authors)


def _split(stem: str) -> list[str]:
    if re.search(r"\s[-–]\s|\s[-–]|[-–]\s", stem):
        return [p for p in re.split(r"\s*[-–]\s+|\s+[-–]\s*", stem) if p.strip()]
    if stem.count("-") == 1:
        return stem.split("-")
    return [stem]


def _strip_leading_author(stem: str, authors: list[str]) -> str:
    """'Nora Roberts-Hotel BoonsBoro-1...' -> 'Hotel BoonsBoro-1...' (any separator)."""
    for a in sorted(authors, key=len, reverse=True):
        at = tokens(a)
        if not at:
            continue
        m = re.match(r"\s*" + r"[\s.,_-]+".join(re.escape(t) for t in at) + r"\s*[-–,:.]?\s*",
                     fold(stem))
        if m and m.end() < len(stem):
            return stem[m.end():].lstrip(" -–,:.")
    return stem


def _strip_author(stem: str, authors: list[str]) -> str:
    stem = _strip_leading_author(stem, authors)
    parts = _split(stem)
    if len(parts) == 1 and "-" in stem and " " not in stem:  # 'John-Grisham-Tvrtka'
        words = stem.split("-")
        for n in (3, 2):
            if _is_author(" ".join(words[:n]), authors):
                return " ".join(words[n:])
            if _is_author(" ".join(words[-n:]), authors):
                return " ".join(words[:-n])
    if len(parts) == 1 and "," in stem:  # 'Колиба, Вилијам Јанг' — only if a piece IS the author,
        pieces = [p.strip() for p in stem.split(",")]  # never for 'Ljubav, struja, voda i telefon'
        if any(_is_author(p, authors) for p in pieces):
            parts = pieces
    rest = [p for p in parts if not _is_author(p, authors)]
    if len(rest) == len(parts):
        return stem  # nothing was an author: keep the title exactly as written
    rest = [re.sub(r"^\d{1,3}\s*$", "", p) for p in rest]  # lone series numbers: '13 - Trag'
    rest = [re.sub(r"^\d{1,2}\s+(?=[^\d\s.])", "", p) for p in rest]  # '02 Odjek u tami' (keeps '1.', '2001')
    return " - ".join(p for p in rest if p)


def _from_filename(path: Path, authors: list[str]) -> str:
    return _strip_author(_tidy(book_stem(path)), authors)


def _from_tag(tag_title: str, authors: list[str]) -> str:
    title = _tidy(tag_title or "")
    if not title or _is_author(title, authors):
        return ""
    return _strip_author(title, authors)


def _similar(a: str, b: str) -> bool:
    fa, fb = fold(a), fold(b)
    return fa in fb or fb in fa or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.8


def decide_title(path: Path, archive: Path | None, tags: dict, authors: list[str]) -> str:
    """Filename is usually the most deliberate source; the tag wins when it agrees with it
    (it tends to have proper diacritics and the full title, e.g. 'tikva' -> 'Pokondirena tikva')."""
    from_name = _from_filename(archive or path, authors) or (archive and _from_filename(path, authors)) or ""
    from_tag = _from_tag(tags.get("title", ""), authors)
    if from_name and from_tag and _similar(from_name, from_tag):
        if len(_split(from_tag)) > len(_split(from_name)):  # 'tvrdjava - imam' vs 'Tvrđava'
            return from_name
        return from_tag if len(from_tag) >= len(from_name) - 3 else from_name
    return from_name or from_tag or _tidy(book_stem(path))
