import json
import threading
from pathlib import Path


class Cache:
    """Per-file extraction results keyed by path+size+mtime."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def _key(file: Path) -> str:
        st = file.stat()
        return f"{file}|{st.st_size}|{int(st.st_mtime)}"

    def get(self, file: Path, field: str):
        return self.data.get(self._key(file), {}).get(field)

    def set(self, file: Path, field: str, value) -> None:
        with self._lock:
            self.data.setdefault(self._key(file), {})[field] = value

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
