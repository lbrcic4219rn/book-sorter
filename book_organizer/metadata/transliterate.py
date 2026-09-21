"""Cyrillic to Latin transliteration utilities."""

import json
import re


# Cyrillic to Latin mapping (Serbian/general)
CYRILLIC_TO_LATIN = {
    # ...existing code...
}

# Cache for LLM-based initial expansions to avoid redundant calls
_initial_expansion_cache: dict[str, str] = {
    # Uppercase
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
    'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
    'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
    'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'C',
    'Ш': 'S', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
    'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya', 'Ћ': 'Ć', 'Ђ': 'Đ', 'Ј': 'J', 'Љ': 'Lj', 'Њ': 'Nj',
    # Lowercase
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
    'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'c',
    'ш': 's', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya', 'ћ': 'ć', 'ђ': 'đ', 'ј': 'j', 'љ': 'lj', 'њ': 'nj',
}


def transliterate_cyrillic(text: str) -> str:
    """Convert Cyrillic script to Latin script."""
    if not text:
        return ""

    result = []
    for char in text:
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        else:
            result.append(char)

    return ''.join(result)


def remove_titles_and_expand_initials(name: str, model: str | None = None) -> str:
    """Remove titles (Dr., Prof., Mr., etc.) and expand common abbreviated initials.
    Ensures author names are in full form without honorifics.
    If model is provided, uses LLM to expand any remaining initials intelligently."""
    if not name:
        return ""
    
    # Remove common titles
    titles_pattern = r'\b(Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Sir|Dame|Saint|St\.)\s*'
    name = re.sub(titles_pattern, '', name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()

    # Check if name has unresolved initials (letters followed by dots)
    has_initials = bool(re.search(r'\b[A-Z]\.\s', name))

    # If we have initials and a model is available, use LLM to expand them
    if has_initials and model:
        return _expand_initials_with_llm(name, model)

    return name


def _expand_initials_with_llm(name: str, model: str) -> str:
    """Use LLM to expand initials in an author name to full first/middle names.
    Results are cached to avoid redundant LLM calls for the same pattern."""
    # Check cache first
    if name in _initial_expansion_cache:
        return _initial_expansion_cache[name]

    from book_organizer.metadata.llm import call_ollama

    prompt = (
        "You are expanding abbreviated author names. Given an author name with initials, "
        "provide the most likely full version of the name with all initials expanded to full names.\n"
        'Respond ONLY with JSON: {"full_name": "..."}\n\n'
        f"Author name: {name}"
    )
    try:
        raw = call_ollama(prompt, model)
        data = json.loads(raw)
        full_name = data.get("full_name")
        if full_name:
            # Clean up the result
            result = re.sub(r"\s+", " ", full_name.strip())
            _initial_expansion_cache[name] = result
            return result
        return name
    except (json.JSONDecodeError, AttributeError, Exception):
        return name
