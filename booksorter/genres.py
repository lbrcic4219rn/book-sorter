"""Decide 1-3 genres per book with the local LLM, restricted to a fixed taxonomy."""
import difflib
import json
import re
import subprocess
import urllib.error
import urllib.request

from booksorter.llm import OLLAMA_URL
from booksorter.names import fold

PROMPT = """Classify this book. It is probably Serbian/Croatian/Bosnian, often a translation.

Author: {author}
Title: {title}
Existing tags: {tags}
{hint}
First pages:
\"\"\"{snippet}\"\"\"

Pick 1-{max_genres} genres from this list only, most important first:
{taxonomy}

Rules: pick the most SPECIFIC genres that fit (Science Fiction, Mystery, Romance, Young Adult…). \
Use "Literary Fiction" only when no specific genre fits, never together with one. \
One genre is enough when the book is clearly one thing.

Respond ONLY with JSON: {{"genres": ["..."]}}"""


def _ask(prompt: str, model: str) -> list[str]:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "format": "json", "think": False, "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(json.loads(resp.read())["message"]["content"])
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
        return []
    genres = data.get("genres") or data.get("genre") or []
    return [genres] if isinstance(genres, str) else [g for g in genres if isinstance(g, str)]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", fold(s)).strip()


def _match(name: str, taxonomy: list[str]) -> str | None:
    """Map whatever the model said onto the taxonomy ('sci-fi', 'naučna fantastika' -> 'Science Fiction')."""
    f = _norm(name)
    for g in taxonomy:
        if _norm(g) == f:
            return g
    aliases = {"sci fi": "Science Fiction", "scifi": "Science Fiction", "sf": "Science Fiction",
               "thriller": "Crime & Thriller", "crime": "Crime & Thriller", "biography": "Biography & Memoir",
               "memoir": "Biography & Memoir", "religion": "Religion & Spirituality", "satire": "Humor & Satire",
               "humor": "Humor & Satire", "humour": "Humor & Satire", "drama": "Drama & Plays",
               "play": "Drama & Plays", "childrens": "Children's", "kids": "Children's",
               "self help": "Self-Help", "politics": "Politics & Society", "society": "Politics & Society",
               "business": "Business & Economics", "economics": "Business & Economics",
               "art": "Art & Music", "music": "Art & Music", "nature": "Nature & Environment",
               "education": "Reference & Education", "reference": "Reference & Education",
               "erotic": "Erotica", "ya": "Young Adult", "comics": "Graphic Novel",
               # Serbian / Croatian names the model sometimes answers with
               "naucna fantastika": "Science Fiction", "sf": "Science Fiction",
               "fantastika": "Fantasy", "epska fantastika": "Fantasy", "horor": "Horror",
               "krimi": "Crime & Thriller", "triler": "Crime & Thriller", "kriminalisticki roman": "Crime & Thriller",
               "ljubavni roman": "Romance", "istorijski roman": "Historical Fiction",
               "istorija": "History", "filozofija": "Philosophy", "psihologija": "Psychology",
               "poezija": "Poetry", "drama": "Drama & Plays", "biografija": "Biography & Memoir",
               "memoari": "Biography & Memoir", "religija": "Religion & Spirituality",
               "pripovetke": "Short Stories", "price": "Short Stories", "satira": "Humor & Satire",
               "putopis": "Travel", "avantura": "Adventure", "klasik": "Classic Literature",
               "klasicna knjizevnost": "Classic Literature", "roman": "Literary Fiction",
               "publicistika": "Essays", "nauka": "Science", "sport": "Sports", "kuvar": "Cooking"}
    if f in aliases:
        return aliases[f]
    close = difflib.get_close_matches(f, [_norm(g) for g in taxonomy], n=1, cutoff=0.85)
    return next((g for g in taxonomy if _norm(g) == close[0]), None) if close else None


def collection_hint(old_tags: str, hints: dict) -> str:
    """'Polaris CD2' means the book came from an SF collection — a hint, not a rule."""
    low = fold(old_tags or "")
    for pattern, genre in hints.items():
        if fold(pattern) in low:
            return genre
    return ""


def decide_genres(author: str, title: str, snippet: str, tags: dict, taxonomy: list[str],
                  model: str, max_genres: int = 3, hint: str = "") -> list[str]:
    hint_line = (f"This book comes from a {hint} collection — likely, but check the text; "
                 f"a {hint} collection can still contain other genres." if hint else "")
    prompt = PROMPT.format(
        author=author or "unknown", title=title or "unknown", hint=hint_line,
        tags=tags.get("tags", "") or "none", snippet=(snippet or "")[:1500] or "(no text available)",
        max_genres=max_genres, taxonomy="\n".join(f"- {g}" for g in taxonomy))
    out, seen = [], set()
    for name in _ask(prompt, model):
        g = _match(name, taxonomy)
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    out = out[:max_genres]
    specific = [g for g in out if g != "Literary Fiction"]
    return specific or out  # 'Literary Fiction' is the fallback, not an extra label


def read_tags(path) -> dict:
    """Current Tags field, so we can log what we overwrite."""
    try:
        out = subprocess.run(["ebook-meta", str(path)], capture_output=True, text=True, timeout=60).stdout
    except (subprocess.TimeoutExpired, OSError):
        return {}
    for line in out.splitlines():
        if line.startswith("Tags"):
            return {"tags": line.split(":", 1)[1].strip()}
    return {}


def write_genres(path, genres: list[str]) -> str | None:
    r = subprocess.run(["ebook-meta", str(path), "--tags", ", ".join(genres)],
                       capture_output=True, text=True, timeout=120)
    return None if r.returncode == 0 else f"ebook-meta failed: {r.stderr.strip()[:200]}"


def harmonize(by_book: dict[str, list[str]], same_book: dict[str, list[str]],
              by_series: dict[str, list[str]], by_author: dict[str, list[str]]) -> dict[str, list[str]]:
    """Three passes. First: copies of ONE book (same author and title, different format) must carry
    the same tags. Then: a genre most of an author's books share is added to the others, so a
    series isn't split between 'Romance' and 'Young Adult, Fantasy'. Then the same per series,
    then per author."""
    from collections import Counter
    fixed = dict(by_book)

    for copies in same_book.values():
        if len(copies) < 2:
            continue
        counts = Counter(g for b in copies for g in fixed.get(b, []))
        merged = [g for g, _ in counts.most_common(3)]
        for b in copies:
            fixed[b] = merged

    for books in by_series.values():  # 'Hotel BoonsBoro 1/2/3' must agree
        if len(books) < 2:
            continue
        counts = Counter(g for b in books for g in fixed.get(b, []))
        merged = [g for g, _ in counts.most_common(3)]
        for b in books:
            fixed[b] = merged

    for books in by_author.values():
        if len(books) < 3:
            continue
        counts = Counter(g for b in books for g in set(fixed.get(b, [])))
        shared = [g for g, n in counts.items() if n >= max(2, len(books) * 0.6)]
        for b in books:
            current = fixed[b]
            fixed[b] = (current + [g for g in shared if g not in current])[:3] or current
    return fixed


def series_key(author: str, stem: str) -> tuple | None:
    """'Hotel BoonsBoro-1.Sada i zauvijek' -> ('Nora Roberts', 'hotel boonsboro'): the part of the
    title before the volume number, when there is one."""
    title = stem.split(" - ", 1)[-1]
    m = re.search(r"^(.{4,}?)[\s\-_,.#]*\b(\d{1,2})[\s\-_.:)]", title)
    return (author, _norm(m.group(1))) if m else None
