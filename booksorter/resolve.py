"""Vote across sources to pick one author per book, then canonicalize library-wide."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from booksorter.names import looks_serbian_transcribed, is_initial, tokens, strip_credits, first_name, appears_in, foreignness, fold, has_diacritics, is_title_like, key, same_person

WEIGHTS = {"tags": 3, "filename": 2, "folder": 2, "text": 2, "library": 3, "name": 2, "pdftags": 1, "llm": 3, "swappedtags": 2, "impressum": 4, "llmpick": 3}
MAX_SCORE = sum(WEIGHTS.values()) - WEIGHTS["pdftags"] - WEIGHTS["swappedtags"] - WEIGHTS["impressum"]


def _middle_initials(name: str) -> int:
    """'Patricia D. Cornwell' -> 1. Only counts after a written-out first name, so the initials
    in 'J. K. Rowling' or 'H. P. Lovecraft' are part of the name, not optional extras."""
    toks = tokens(name)
    if len(toks) < 3 or is_initial(toks[0]):
        return 0
    return sum(is_initial(t) for t in toks[1:-1])


def _full_tokens(name: str) -> list[str]:
    return [t for t in tokens(name) if not is_initial(t)]


def pick_display(counts: Counter) -> str:
    """One display spelling. A more original-looking spelling only beats its own transcription
    (same full words: 'Alice Munro' > 'Alis Manro', 'Arthur C. Clarke' > 'Artur Klark');
    A Serbian-lettered spelling loses to a genuinely different Latin one ('Dž.' vs 'George'), but
    not to the same name with diacritics stripped ('Nušić' beats 'Nusic').
    Otherwise: complete > no middle initial > diacritics > most books > longest."""
    names = list(counts)
    originals = [n for n in names if not any(
        (len(_full_tokens(o)) == len(_full_tokens(n)) and foreignness(o) > foreignness(n))
        # 'Dž. R. R. Martin' is a transcription of 'George R. R. Martin'; 'Nusic' is not an original of 'Nušić'
        or (looks_serbian_transcribed(n) and not looks_serbian_transcribed(o) and key(o) != key(n))
        for o in names)] or names
    return max(originals, key=lambda n: (not _truncated(n, counts), -_middle_initials(n),
                                         has_diacritics(n), counts[n], len(n)))


def _truncated(name: str, spellings) -> bool:
    """'Halil Džubr' is a cut-off 'Halil Džubran'."""
    return any(o != name and o.startswith(name) for o in spellings)


@dataclass
class Cluster:
    names: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)

    weak_filename: bool = False  # filename gave several parts; one of them is the title

    @property
    def score(self) -> int:
        if self.sources == {"filename"} and self.weak_filename:
            return 1  # an unconfirmed filename part is probably the title
        return sum(WEIGHTS[s] for s in self.sources)

    @property
    def best_name(self) -> str:
        # prefer diacritics, then longest (fullest) spelling, then most frequent
        counts = Counter(self.names)
        return pick_display(counts)


@dataclass
class Decision:
    author: str | None
    confidence: float
    sources: list[str]
    candidates: list[tuple[str, int]]
    coauthors: list[str] = field(default_factory=list)


def cluster(candidates: dict[str, list[str]]) -> list[Cluster]:
    clusters: list[Cluster] = []
    for source, names in candidates.items():
        for n in names:
            c = next((c for c in clusters if any(same_person(n, m) for m in c.names)), None)
            if c is None:
                c = Cluster()
                clusters.append(c)
            c.names.append(n)
            c.sources.add(source)
    return clusters


def decide(candidates: dict[str, list[str]], snippet: str, library_counts: Counter,
           title: str = "", first_names: set[str] = frozenset(),
           groups: list[list[str]] = ()) -> Decision:
    if title:  # whatever matches the embedded title is the title, not the author
        candidates = {s: [n for n in ns if not is_title_like(n, title)] for s, ns in candidates.items()}
    clusters = cluster(candidates)
    folded = strip_credits(fold(snippet))
    for c in clusters:
        if folded and any(appears_in(n, folded) for n in c.names):
            c.sources.add("text")
        if max(library_counts[key(n)] for n in c.names) >= 3:
            c.sources.add("library")
        if any(first_name(n) in first_names for n in c.names):
            c.sources.add("name")
    ambiguous = len(candidates.get("filename", [])) > 1
    for c in clusters:
        c.weak_filename = ambiguous
    if not clusters:
        return Decision(None, 0.0, [], [])
    clusters.sort(key=lambda c: c.score, reverse=True)
    win = clusters[0]
    runner = clusters[1].score if len(clusters) > 1 else 0
    # half from absolute evidence, half from margin over the runner-up
    confidence = 0.5 * win.score / MAX_SCORE + 0.5 * (win.score - runner) / win.score
    coauthors = []
    for g in groups:
        if any(same_person(n, m) for n in g for m in win.names):
            # the author listed first in the book is the primary one when votes are tied
            first = next((c for c in clusters if any(same_person(g[0], m) for m in c.names)), None)
            if first is not None and first is not win and first.score >= win.score:
                win = first
            for n in g:
                if any(same_person(n, m) for m in win.names) or any(same_person(n, c) for c in coauthors):
                    continue
                own = next((c for c in clusters if any(same_person(n, m) for m in c.names)), None)
                # a real person needs more than the filename: another source or a known first name
                if (own and own.sources - {"filename", "text", "library"}) or first_name(n) in first_names:
                    coauthors.append(n)
    return Decision(
        win.best_name, round(min(confidence, 1.0), 2), sorted(win.sources),
        [(c.best_name, c.score) for c in clusters], coauthors,
    )


def canonicalize(authors: list[str]) -> dict[str, str]:
    """Map every spelling to one display form per person across the library."""
    groups: dict[str, list[str]] = defaultdict(list)  # exact key -> spellings
    for a in authors:
        groups[key(a)].append(a)
    reps = sorted(groups, key=lambda k: -len(groups[k]))
    merged: list[list[str]] = []
    for k in reps:
        target = next((m for m in merged if any(same_person(groups[k][0], x) for x in set(m))), None)
        (target.extend(groups[k]) if target else merged.append(list(groups[k])))
    mapping = {}
    for spellings in merged:
        counts = Counter(spellings)
        best = pick_display(counts)
        mapping.update({s: best for s in spellings})
    return mapping
