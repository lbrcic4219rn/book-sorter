import difflib

from book_organizer.metadata.llm import verify_match
from book_organizer.metadata.models import BookMetadata


def confidence_score(old: BookMetadata, new: BookMetadata) -> float:
    """Rough 0-1 similarity between old and new title+author strings."""
    old_str = f"{old.title or ''} {old.authors or ''}".lower()
    new_str = f"{new.title or ''} {new.authors or ''}".lower()
    return difflib.SequenceMatcher(None, old_str, new_str).ratio()


def resolve_match(
    old: BookMetadata,
    new: BookMetadata,
    threshold: float,
    use_llm: bool,
    llm_model: str,
) -> tuple[float, bool]:
    """Return (score, accepted) — accepted True means confident enough to apply.

    Fuzzy string matching alone misses cases like reordered author names or
    translated titles, so scores in the gray zone just below the threshold
    get a second opinion from the LLM before being rejected.
    """
    score = confidence_score(old, new)

    gray_zone = use_llm and (threshold - 0.25) <= score < threshold
    if gray_zone and verify_match(old, new, llm_model):
        return threshold, True

    return score, score >= threshold