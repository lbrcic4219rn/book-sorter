"""Accuracy check: in 'Knjige 1/<Author>/' the folder is the answer key.
Hide the folder vote and see how often filename+tags+text alone get it right.
Run after `python main.py scan` (uses the extraction cache)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from booksorter import resolve
from booksorter.cache import Cache
from booksorter.names import COMMON_FIRST_NAMES, key, same_person

cfg = main.load_config()
cache = Cache(main.ROOT / ".booksorter_cache.json")
# the books may already be sorted; _Originals keeps the original folder structure (the answer key)
root = Path(cfg["library_path"]) / "Knjige 1"
originals = Path(cfg["output_path"]) / "_Originals" / "Knjige 1"
if originals.exists():
    root = originals
records = []
for book in main.find_books(root)[0]:
    r = main.gather(book, cache)
    if r["candidates"]["folder"]:
        r["answer"] = r["candidates"]["folder"][0]
        r["candidates"]["folder"] = []
        records.append(r)
cache.save()
counts = Counter()
for r in records:
    counts.update({key(n) for ns in r["candidates"].values() for n in ns})
names = set(COMMON_FIRST_NAMES)
right = confident = confident_right = 0
misses = []
for r in records:
    d = resolve.decide(r["candidates"], r["snippet"], counts, r["title"], names, r["groups"])
    if cfg.get("use_llm") and not main.settled(d):
        r["candidates"]["llm"] = main.ask_llm(r["book"], r["tags"], r["snippet"], cache, cfg["llm_model"])
        d = resolve.decide(r["candidates"], r["snippet"], counts, r["title"], names, r["groups"])
    ok = bool(d.author) and same_person(d.author, r["answer"])
    right += ok
    if d.confidence >= cfg["min_confidence"]:
        confident += 1
        confident_right += ok
    if not ok:
        misses.append((r["answer"], d.author, d.candidates[:3]))
print(f"{len(records)} books | correct {right / len(records):.0%} | confident {confident} "
      f"({confident_right / max(confident, 1):.1%} of those correct)")
for m in misses[:10]:
    print("  miss:", m)
cache.save()
