"""Local LLM (Ollama) as one voter: reads all the evidence and names the author."""
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

PROMPT = """You identify the AUTHOR of an ebook. Books are mostly Serbian/Croatian/Bosnian, \
in Latin or Cyrillic script; many are translations, so the author may be foreign.

Evidence (any of it may be wrong or missing):
- File name: {filename}
- Folder: {folder}
- Embedded metadata: title={title!r}, author={tag_author!r}
- First pages of the book:
\"\"\"{snippet}\"\"\"

Rules:
- The file name may be "Author - Title" OR "Title - Author"; decide which part is the person.
- Metadata fields are often wrong: the author field may hold the title, a single first name of whoever scanned the book (e.g. "Sanja"), or junk like "Unknown"/"Admin"; the title field may hold the author's name. A real author almost always has a first AND last name.
- Prefer the name printed on the title page. Ignore translators, editors, illustrators, publishers.
- Write the name in Latin script, first name first. If the book gives the original foreign \
spelling (e.g. "Alice Munro" rather than "Alis Manro"), use the original.
- If you cannot tell, answer null. Do not guess.

Respond ONLY with JSON: {{"author": "First Last" or null}}"""


def identify_author(filename: str, folder: str, tags: dict, snippet: str, model: str) -> str | None:
    prompt = PROMPT.format(
        filename=filename, folder=folder, title=tags.get("title", ""),
        tag_author=tags.get("authors", ""), snippet=snippet[:3000] or "(no text available)",
    )
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "format": "json", "think": False, "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            reply = json.loads(resp.read())["message"]["content"]
        author = json.loads(reply).get("author")
        return author.strip() if isinstance(author, str) and author.strip() else None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, AttributeError):
        return None
