import json

from book_organizer.metadata.llm import call_ollama

_cache: dict[str, str] = {}


def classify_genre(raw_genre: str, taxonomy: list[str], model: str) -> str:
    """Map a free-text genre string onto the closest entry in a fixed taxonomy.
    Cached, since many books share identical raw genre strings."""
    if not raw_genre:
        return "Unknown"
    if raw_genre in _cache:
        return _cache[raw_genre]

    prompt = (
        "Classify the following raw genre/tag text into EXACTLY ONE of these "
        f"categories: {', '.join(taxonomy)}.\n"
        "Pick the closest match even if imperfect. If truly none fit, use "
        '"Unknown".\n'
        'Respond ONLY with JSON: {"category": "..."}.\n\n'
        f"Raw text: {raw_genre}"
    )
    raw = call_ollama(prompt, model)
    try:
        category = json.loads(raw).get("category", "Unknown")
    except (json.JSONDecodeError, AttributeError):
        category = "Unknown"

    if category not in taxonomy:
        category = "Unknown"

    _cache[raw_genre] = category
    return category