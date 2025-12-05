from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class CacheManager:
    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._data: dict[str, Any] = {"version": 1, "files": {}}
        self._loaded = False
        self._dirty = False

    def load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                text = self._path.read_text(encoding="utf-8")
                self._data = json.loads(text)
            except (json.JSONDecodeError, OSError):
                self._data = {"version": 1, "files": {}}
        self._loaded = True

    def clear(self) -> None:
        self._data = {"version": 1, "files": {}}
        self._dirty = True

    def _relativize(self, file_path: Path) -> str:
        try:
            return str(file_path.relative_to(self._path.parent))
        except ValueError:
            return str(file_path)

    def should_skip(self, file_path: Path, digest: str, signature: str) -> bool:
        self.load()
        files = self._data.get("files", {})
        entry = files.get(self._relativize(file_path))
        return bool(
            entry
            and entry.get("hash") == digest
            and entry.get("signature") == signature
        )

    def update(self, file_path: Path, digest: str, signature: str) -> None:
        self.load()
        files = self._data.setdefault("files", {})
        files[self._relativize(file_path)] = {
            "hash": digest,
            "signature": signature,
            "timestamp": time.time(),
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._data, indent=2, sort_keys=True)
        self._path.write_text(text, encoding="utf-8")
        self._dirty = False
