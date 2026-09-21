import difflib
import json
import re
from collections import defaultdict

import requests

from book_organizer.metadata.llm import call_ollama
from book_organizer.metadata.transliterate import remove_titles_and_expand_initials


def normalize_name(name: str) -> str:
    """Rough normalization: 'Last, First' -> 'First Last', collapse whitespace."""
    name = name.strip()
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    return re.sub(r"\s+", " ", name)


def _fingerprint(name: str) -> str:
    """Key for clustering: lowercase, drop periods/hyphens, letters only."""
    name = normalize_name(name).lower()
    name = re.sub(r"[.\-]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def cluster_authors(all_authors: list[str]) -> dict[str, list[str]]:
    """Group name variants likely referring to the same person: exact
    fingerprint match first ('J.K. Rowling' vs 'JK Rowling'), then a fuzzy
    pass for near-misses (typos, minor abbreviation differences)."""
    fingerprints: dict[str, list[str]] = defaultdict(list)
    for name in all_authors:
        if name:
            fingerprints[_fingerprint(name)].append(name)

    keys = list(fingerprints.keys())
    merged: dict[str, list[str]] = {}
    used = set()

    for i, key in enumerate(keys):
        if key in used:
            continue
        group = list(fingerprints[key])
        used.add(key)
        for other in keys[i + 1:]:
            if other not in used and difflib.SequenceMatcher(None, key, other).ratio() > 0.88:
                group.extend(fingerprints[other])
                used.add(other)
        merged[key] = group

    return merged


def _lookup_canonical_online(name: str) -> str | None:
    """Query Open Library's author search for the authoritative spelling.

    OPTIMIZATION: Calculate similarity only once instead of twice."""
    try:
        resp = requests.get(
            "https://openlibrary.org/search/authors.json",
            params={"q": name}, timeout=10,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
    except requests.RequestException:
        return None
    if not docs:
        return None

    # Calculate best match with similarity in one pass
    best_name = None
    best_similarity = 0.0
    best_work_count = 0

    for doc in docs:
        similarity = difflib.SequenceMatcher(None, name.lower(), doc.get("name", "").lower()).ratio()
        work_count = doc.get("work_count", 0)

        # Choose doc if it has better similarity, or same similarity with more works
        if similarity > best_similarity or (similarity == best_similarity and work_count > best_work_count):
            best_similarity = similarity
            best_work_count = work_count
            best_name = doc["name"]

    return best_name if best_similarity > 0.7 else None


def _best_local_variant(variants: list[str]) -> str:
    """Fallback if no online match: prefer fuller names (fewer bare initials)."""
    return sorted(variants, key=lambda v: (v.count("."), -len(v)))[0]


def _normalize_author_with_llm(name: str, model: str) -> str | None:
    """Use LLM as last resort to normalize author name to canonical form.
    Explicitly requests full names without abbreviations or titles.

    OPTIMIZATION: Don't call remove_titles_and_expand_initials again here,
    since it was already done in build_author_mapping. The LLM request itself
    asks for expanded names, so we just parse the result."""
    prompt = (
        "You are normalizing an author name. Given a possibly misspelled, abbreviated, "
        "or oddly-formatted author name, provide the most likely FULL canonical spelling.\n"
        "IMPORTANT: Expand all initials to full names (e.g., 'F. Scott Fitzgerald' -> 'Francis Scott Fitzgerald')\n"
        "Remove any titles like Dr., Prof., Mr., Mrs., etc.\n"
        'Respond ONLY with JSON: {"canonical_name": "..."}\n\n'
        f"Author name: {name}"
    )
    try:
        raw = call_ollama(prompt, model)
        data = json.loads(raw)
        canonical = data.get("canonical_name")
        return canonical.strip() if canonical else None
    except (json.JSONDecodeError, AttributeError, Exception):
        return None


def build_author_mapping(all_authors: list[str], llm_model: str | None = None) -> dict[str, str]:
    """Return {original_variant: canonical_name} across the whole library.
    Falls back to LLM if canonical name not found online and llm_model is provided.
    Enforces full names without abbreviations or titles."""
    clusters = cluster_authors(all_authors)
    mapping: dict[str, str] = {}
    for variants in clusters.values():
        representative = normalize_name(_best_local_variant(variants))
        
        # Remove titles and expand abbreviations from representative
        representative = remove_titles_and_expand_initials(representative, llm_model)
        
        # Try online lookup
        canonical = _lookup_canonical_online(representative)
        
        # Clean up the canonical name from online source
        if canonical:
            canonical = remove_titles_and_expand_initials(canonical, llm_model)
        
        # Use LLM as last resort if online lookup failed and LLM is available
        if not canonical and llm_model:
            canonical = _normalize_author_with_llm(representative, llm_model)
        
        # Fall back to representative if all else fails
        canonical = canonical or representative
        
        for v in variants:
            mapping[v] = canonical
    return mapping