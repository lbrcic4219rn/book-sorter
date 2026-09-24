"""BookSorter: decide each book's author by comparing filename, tags and first pages.

  python main.py scan [--limit N] [--only SUBFOLDER]   -> report.csv (review/edit it)
  python main.py apply [--limit N] [--min-confidence X] -> moves books into output_path/<Author>/
  python main.py undo                                   -> reverses the last apply
"""
import argparse
import csv
import re
import json
import os
import shutil
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from booksorter import formats, organize, sources
from booksorter.cache import Cache
from booksorter.llm import identify_author, pick_author
from booksorter.names import transliterate, appears_in, strip_credits, display_clean, looks_like_name, COMMON_FIRST_NAMES, first_name, fix_order, fold, key, same_person
from booksorter.titles import decide_title
from booksorter.resolve import canonicalize, decide

ROOT = Path(__file__).parent
FIELDS = ["path", "archive", "author", "coauthors", "title", "confidence", "sources", "candidates", "status"]


EXTRACT_DIR = ROOT / ".booksorter_extracted"
ARCHIVE_OF: dict[Path, Path] = {}  # extracted book -> the archive it came from


def find_books(root: Path, limit: int | None = None) -> tuple[list[Path], list[Path]]:
    """Every book under root, by real format (not extension); archives are unpacked."""
    books, skipped = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        fmt = formats.detect(p)
        if fmt in formats.BOOK_FORMATS:
            books.append(p)
        elif fmt in formats.ARCHIVE_FORMATS:
            inner = formats.extract(p, EXTRACT_DIR)
            for b in inner:
                ARCHIVE_OF[b] = p
            books += inner
            if not inner:
                skipped.append(p)
        else:
            skipped.append(p)
        if limit and len(books) >= limit:
            return books[:limit], skipped
    return books, skipped


OVERRIDES = ROOT / "overrides.csv"


def load_overrides() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """overrides.csv: 'from,to'. 'from' is either an author spelling (every book by that person
    gets 'to') or a file name / path fragment (that one book gets 'to')."""
    names, files = [], []
    if not OVERRIDES.exists():
        return names, files
    with open(OVERRIDES, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src, dst = (row.get("from") or "").strip(), (row.get("to") or "").strip()
            if not src or src.startswith("#"):
                continue
            is_file = "/" in src or formats.detect(Path(src)) or Path(src).suffix.lower() in (
                ".epub", ".pdf", ".mobi", ".azw3", ".azw", ".doc", ".docx", ".rtf", ".rar", ".zip")
            (files if is_file else names).append((unicodedata.normalize("NFC", src), dst))
    return names, files


def load_config() -> dict:
    path = Path(os.environ.get("BOOKSORTER_CONFIG", ROOT / "config.json"))  # e.g. a sandbox config
    return json.loads(path.read_text(encoding="utf-8"))


def caps_title_suspect(author: str, r: dict) -> bool:
    c = r["candidates"]
    # only when the filename is ONE segment ('stepski vuk.doc'), not 'Dejvid Gemel - Legenda'
    if r.get("name_parts", 1) > 1 or len(c["filename"]) != 1 or not same_person(author, c["filename"][0]):
        return False
    if any(any(same_person(author, n) for n in c.get(s, [])) for s in ("folder", "impressum", "swappedtags")):
        return False
    raw = sources.book_stem(ARCHIVE_OF.get(r["book"], r["book"]))
    if raw == raw.lower() and not any(any(same_person(author, n) for n in c.get(s, []))
                                      for s in ("tags", "pdftags")):
        return True  # 'drevni_egipat.pdf': people's names are capitalised
    head = transliterate(r["snippet"][:300])
    return author.upper() in head.upper() and any(
        w.isupper() and len(w) > 2 and fold(w) in {fold(t) for t in author.split()} for w in head.split())


ANTHOLOGY_AUTHOR = "Razni autori"
_ANTHOLOGY_WORDS = re.compile(
    r"\b(antologij\w*|zbornik\w*|almanah\w*|zbirka pri[cč]\w*|[cč]asopis\w*|best of|the best|"
    r"sirius \d+|galaksija \d+|politikin zabavnik|razni autori|grupa autora|vi[sš]e autora)\b", re.I)
_VARIOUS = {"various", "various authors", "razni autori", "grupa autora", "vise autora", "više autora", "anthology"}


def is_anthology(r: dict) -> bool:
    """Magazines and collections by many writers: title words, a 'various' tag, or a table of
    contents naming 3+ different authors ('SADRŽAJ: Walter M. Miller: ŽRTVOVANI 3 Dmitrij Bilenkin…')."""
    name = transliterate(sources.book_stem(ARCHIVE_OF.get(r["book"], r["book"])))
    if _ANTHOLOGY_WORDS.search(name) or _ANTHOLOGY_WORDS.search(transliterate(r["tags"].get("title", ""))):
        return True
    if fold(r["tags"].get("authors", "")).strip() in _VARIOUS:
        return True
    head = transliterate(r["snippet"][:3000])
    toc = re.search(r"(?i)\b(sadr[zž]aj|contents)\b[:\s](.{0,1500})", head, re.S)
    if toc:
        names = {fold(n) for n in re.findall(r"([A-ZČĆŠŽĐ][a-zčćšžđ]+(?: [A-Z]\.)? [A-ZČĆŠŽĐ][a-zčćšžđ]+)\s*[:–-]", toc.group(2))}
        if len(names) >= 3:
            return True
    return False


STRONG = {"tags", "filename", "folder", "text", "impressum", "swappedtags"}


def settled(d) -> bool:
    """Plain sources already agree: 3+ independent sources and no real competitor."""
    return len(STRONG & set(d.sources)) >= 3 and d.confidence >= 0.6


def ask_llm(book: Path, tags: dict, snippet: str, cache: Cache, model: str) -> list[str]:
    field = f"llm:{model}"
    answer = cache.get(book, field)
    if answer is None:
        answer = identify_author(book.name, book.parent.name, tags, snippet, model) or ""
        cache.set(book, field, answer)
    answer = display_clean(answer)
    return [answer] if looks_like_name(answer) else []


def gather(book: Path, cache: Cache, llm_model: str | None = None) -> dict:
    tags = cache.get(book, "tags")
    if tags is None:
        tags = sources.read_tags(book)
        cache.set(book, "tags", tags)
    snippet = cache.get(book, "text")
    if snippet is None:
        snippet = sources.first_pages(book)
        cache.set(book, "text", snippet)
    origin = ARCHIVE_OF.get(book, book)  # for archived books, the archive's name/folder count too
    folder = sources.from_folder(origin)
    from_name = sources.from_filename(book)
    if origin is not book:
        from_name += [n for n in sources.from_filename(origin) if n not in from_name]
    title = tags.get("title", "")
    # Swapped fields: title tag holds the author ('A. J. Molloy'), confirmed by the filename.
    # Only when the filename has another part too — otherwise the match is just the title itself.
    # Swapped fields: the title tag holds the author. Trust it only if the author tag is junk
    # ('Sanja', 'Unknown') or is itself the other half of the filename (a clean swap).
    tag_authors = sources.from_tags(tags)
    swapped = []
    if looks_like_name(title) and len(from_name) >= 2:
        hits = [n for n in from_name if same_person(display_clean(title), n)]
        rest = [n for n in from_name if n not in hits]
        clean_swap = (all(any(same_person(a, n) for n in rest) for a in tag_authors)
                      and any(any(same_person(h, f) for f in folder) or first_name(h) in COMMON_FIRST_NAMES
                              for h in hits))  # a normal book looks identical, so need outside proof
        if hits and (not tag_authors or clean_swap):
            swapped, tag_authors = hits, []
    # A single-part filename equal to the title tag is just the title ('ZASTO_SE_NISTE_UBILI').
    # Only when the author tag holds a real name — otherwise the title tag may be the author (swapped).
    if len(from_name) == 1 and title and tag_authors and same_person(display_clean(title), from_name[0]):
        from_name = []
    # PDF title tags are often junk (the author's name, 'Microsoft Word - ...'), so don't trust them
    is_pdf = formats.detect(book) == "pdf"
    if is_pdf or any(same_person(title, f) for f in folder):
        title = ""
    return {
        "book": book, "snippet": snippet, "title": title,
        "candidates": {
            ("pdftags" if is_pdf else "tags"): tag_authors,
            "filename": from_name,
            "swappedtags": swapped,
            "folder": folder,
            "impressum": sources.from_impressum(snippet),
            "llm": ask_llm(book, tags, snippet, cache, llm_model) if llm_model else [],
        },
        "tags": tags,
        "name_parts": sources.filename_parts(origin),
        "groups": sources.author_groups(book, tags),
    }


def scan(args, cfg):
    library = Path(cfg["library_path"]).expanduser()
    root = library / args.only if args.only else library
    books, skipped = find_books(root, args.limit)
    print(f"{len(books)} books to scan ({len(skipped)} non-book files skipped)")

    cache = Cache(ROOT / ".booksorter_cache.json")
    llm = cfg["llm_model"] if cfg.get("use_llm") else None
    llm_mode = cfg.get("llm_mode", "auto")  # auto = only when sources disagree; always; off
    records = []
    with ThreadPoolExecutor(cfg.get("workers", 6)) as pool:
        futures = [pool.submit(gather, b, cache, llm if llm_mode == "always" else None) for b in books]
        for i, f in enumerate(as_completed(futures), 1):
            records.append(f.result())
            if i % 50 == 0 or i == len(books):
                print(f"  extracted {i}/{len(books)}")
                cache.save()

    # A folder only counts as an author folder if most of its books agree with it
    # (weeds out genre folders like 'Chic lit.komedije.ljubići').
    by_folder = defaultdict(list)
    for r in records:
        if r["candidates"]["folder"]:
            by_folder[r["book"].parent].append(r)
    for rs in by_folder.values():
        name = rs[0]["candidates"]["folder"][0]
        # confirmed by filenames/tags, or by the book's own text ('Patricia D.Cornwel/13 - Trag')
        agree = sum(any(same_person(name, n) for n in r["candidates"]["filename"] + r["candidates"].get("tags", [])
                        + r["candidates"].get("pdftags", []) + r["candidates"]["impressum"])
                    or appears_in(name, strip_credits(fold(r["snippet"])))
                    for r in rs)
        if agree < max(1, len(rs) / 2):
            for r in rs:
                r["candidates"]["folder"] = []

    # How many books mention each author anywhere — a library-wide prior.
    library_counts = Counter()
    for r in records:
        library_counts.update({key(n) for names in r["candidates"].values() for n in names})

    # First names we trust: built-in list + author folders (tags are too often the title).
    first_names = set(COMMON_FIRST_NAMES)
    for r in records:
        for n in r["candidates"]["folder"]:
            if fold(n.split()[-1]) not in COMMON_FIRST_NAMES:  # skip 'Asimov Isak'-style folders
                first_names.add(first_name(n))
    first_names.discard("")

    threshold = cfg.get("min_confidence", 0.4)

    def ask_pick(r, options):
        field = f"pick:{llm}:{'|'.join(options)}"
        choice = cache.get(r["book"], field)
        if choice is None:
            choice = pick_author(options, r["snippet"], r["book"].name, llm) or ""
            cache.set(r["book"], field, choice)
        return [choice] if choice else []

    for r in records:
        r["decision"] = decide(r["candidates"], r["snippet"], library_counts, r["title"], first_names, r["groups"])

    # Stage 2: ask the LLM only where the plain sources don't clearly agree.
    if llm and llm_mode == "auto":
        unsure = [r for r in records if not settled(r["decision"])]
        print(f"LLM voter ({llm}) on {len(unsure)}/{len(records)} books where sources disagree")
        with ThreadPoolExecutor(cfg.get("llm_workers", 2)) as pool:
            futures = {pool.submit(ask_llm, r["book"], r["tags"], r["snippet"], cache, llm): r for r in unsure}
            for i, f in enumerate(as_completed(futures), 1):
                r = futures[f]
                r["candidates"]["llm"] = f.result()
                r["decision"] = decide(r["candidates"], r["snippet"], library_counts, r["title"], first_names, r["groups"])
                if i % 25 == 0 or i == len(unsure):
                    print(f"  llm {i}/{len(unsure)}")
                    cache.save()

    # How often each exact word order was seen (llm counts too: it knows 'Clifford Simak').
    order_counts = Counter()
    for r in records:
        for src, ns in r["candidates"].items():
            for n in ns:
                order_counts[n] += 2 if src in ("impressum", "llm") else 1

    name_overrides, file_overrides = load_overrides()

    # Last resort: hand the shortlist to the LLM and let it point at the author.
    if llm:
        unsure = [r for r in records if r["decision"].confidence < threshold and len(r["decision"].candidates) > 1]
        print(f"LLM tie-break on {len(unsure)} books that are still unclear")
        with ThreadPoolExecutor(cfg.get("llm_workers", 2)) as pool:
            futures = {pool.submit(ask_pick, r, [c for c, _ in r["decision"].candidates][:6]): r for r in unsure}
            for i, f in enumerate(as_completed(futures), 1):
                r = futures[f]
                r["candidates"]["llmpick"] = f.result()
                r["decision"] = decide(r["candidates"], r["snippet"], library_counts, r["title"],
                                       first_names, r["groups"])
                if i % 25 == 0 or i == len(unsure):
                    print(f"  pick {i}/{len(unsure)}")
                    cache.save()

    # A name that only a one-part filename gives, and which the book prints in CAPITALS at the
    # start, is its title ('stepski vuk.doc' -> 'STEPSKI VUK Ovo je knjiga…'): send to review.
    for r in records:
        d = r["decision"]
        if d.author and caps_title_suspect(d.author, r):
            d.confidence = min(d.confidence, 0.3)

    for r in records:  # anthologies all go to one 'Razni autori' folder (off until we decide)
        if cfg.get("anthologies") and is_anthology(r):
            d = r["decision"]
            d.author, d.coauthors, d.confidence, d.sources = ANTHOLOGY_AUTHOR, [], 1.0, ["anthology"]

    mapping = canonicalize([a for r in records for a in [r["decision"].author, *r["decision"].coauthors] if a])
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda r: str(r["book"])):
            d = r["decision"]
            author = mapping.get(d.author, d.author)
            author = fix_order(author, first_names, order_counts) if author else author
            author = next((dst for src, dst in name_overrides if author and same_person(author, src)), author)
            where = unicodedata.normalize("NFC", str(ARCHIVE_OF.get(r["book"], r["book"])))
            forced = next((dst for src, dst in file_overrides if src in where), None)
            if forced:
                author, d.confidence, d.sources = forced, 1.0, ["override"]
            w.writerow({
                "path": str(r["book"]), "archive": str(ARCHIVE_OF.get(r["book"], "")),
                "author": author or "",
                "title": decide_title(r["book"], ARCHIVE_OF.get(r["book"]), r["tags"],
                                      [a for a in [author, d.author, *d.coauthors] if a]),
                "coauthors": "; ".join(
                    next((dst for src, dst in name_overrides if same_person(c, src)), c)
                    for c in (fix_order(mapping.get(c, c), first_names, order_counts) for c in d.coauthors)),
                "confidence": d.confidence, "sources": "+".join(d.sources),
                "candidates": " | ".join(f"{n} ({s})" for n, s in d.candidates),
                "status": "ok" if author and d.confidence >= threshold else "review",
            })
    confs = [r["decision"].confidence for r in records]
    ok = sum(c >= threshold for c in confs)
    print(f"Wrote {args.report}: {ok}/{len(records)} confident, "
          f"{len(set(mapping.values()))} distinct authors")


def apply(args, cfg):
    """Move confident books into <output>/<Author>/ (as EPUB, PDF stays PDF), tag them,
    and park originals in <output>/_Originals/. Review rows are left untouched."""
    library = Path(cfg["library_path"]).expanduser()
    out = Path(args.output or cfg["output_path"]).expanduser()
    log = out / "moves.log"
    out.mkdir(parents=True, exist_ok=True)

    with open(args.report, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:  # learn each archive's full contents from the extraction cache
        if row["archive"] and Path(row["archive"]).exists():
            for b in formats.extract(Path(row["archive"]), EXTRACT_DIR):
                ARCHIVE_OF[b] = Path(row["archive"])
    already = organize.placed_sources(log)  # placed by an earlier run (e.g. a trial)
    jobs, held_back = [], 0
    by_original: dict[Path, list] = defaultdict(list)
    for row in rows:
        original = Path(row["archive"] or row["path"])
        ready = row["status"] == "ok" and row["author"] and float(row["confidence"]) >= args.min_confidence
        by_original[original].append(ready)
        if ready and Path(row["path"]).exists() and row["path"] not in already:
            jobs.append(organize.Job(Path(row["path"]), original, row["author"],
                                     [c for c in row["coauthors"].split("; ") if c], row["title"]))
        elif not ready:
            held_back += 1
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} books to place, {held_back} left in the library for review")

    results = defaultdict(list)
    counts = Counter()
    with ThreadPoolExecutor(cfg.get("convert_workers", 4)) as pool:
        futures = {pool.submit(organize.process, job, out, library): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:  # never let one bad file stop the run
                entry = {"src": str(job.src), "original": str(job.original), "status": "failed", "error": repr(e)}
            results[job.original].append(entry)
            counts[entry["status"]] += 1
            if entry["status"] == "failed":
                organize.log_line(log, entry)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  done={counts['done']} duplicates={counts['duplicate']} failed={counts['failed']}")

    # Log placed books first, then park an original only once EVERY book it holds has been
    # placed — in this run or an earlier one (an archive may hold several books).
    for entries in results.values():
        for e in entries:
            if e["status"] != "failed":
                organize.log_line(log, e | {"parked": None})
    placed = organize.placed_sources(log)
    parked = 0
    for original in results:
        inside = [b for b, a in ARCHIVE_OF.items() if a == original] or [original]
        if original.exists() and all(str(b) in placed for b in inside):
            where = organize.park_original(original, out, library)
            organize.log_line(log, {"original": str(original), "parked": str(where)})
            parked += 1

    emptied = organize.prune_empty_dirs(library)
    print(f"Removed {emptied} empty folders from the library")
    print(f"Placed {counts['done']} books ({counts['duplicate']} duplicates -> _Duplicates), "
          f"{counts['failed']} failed (kept in library, see {log}). Originals parked: {parked}")


def undo(args, cfg):
    out = Path(args.output or cfg["output_path"]).expanduser()
    restored, removed = organize.undo(out / "moves.log")
    print(f"Restored {restored} originals to the library, removed {removed} placed files")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--limit", type=int)
    s.add_argument("--only", help="subfolder of the library to scan")
    s.add_argument("--report", type=Path, default=ROOT / "report.csv")
    a = sub.add_parser("apply")
    a.add_argument("--report", type=Path, default=ROOT / "report.csv")
    a.add_argument("--output")
    a.add_argument("--limit", type=int, help="only place the first N books (for a trial run)")
    a.add_argument("--min-confidence", type=float, default=cfg.get("min_confidence", 0.4))
    u = sub.add_parser("undo", help="put everything from the last apply back")
    u.add_argument("--output")
    args = p.parse_args()
    {"scan": scan, "apply": apply, "undo": undo}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
