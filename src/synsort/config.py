from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ORDER: list[str] = ["globals", "dunder", "public", "private"]
DEFAULT_HEADERS: dict[str, str] = {
    "globals": "# --- Globals ---",
    "dunder": "# --- Dunder Members ---",
    "public": "# --- Public Members ---",
    "private": "# --- Private Members ---",
}


def _normalize_order(order: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    if order:
        for raw in order:
            item = raw.strip().lower()
            if item in DEFAULT_ORDER and item not in cleaned:
                cleaned.append(item)
    for fallback in DEFAULT_ORDER:
        if fallback not in cleaned:
            cleaned.append(fallback)
    return cleaned


def _merge_headers(overrides: dict[str, str] | None) -> dict[str, str]:
    headers = DEFAULT_HEADERS.copy()
    if overrides:
        for key, value in overrides.items():
            normalized = key.strip().lower()
            if normalized in headers and value.strip():
                headers[normalized] = value.strip()
    return headers


@dataclass
class SynsortConfig:
    order: list[str] = field(default_factory=lambda: DEFAULT_ORDER.copy())
    section_headers: dict[str, str] = field(
        default_factory=lambda: DEFAULT_HEADERS.copy()
    )
    cache_file: str = ".synsort_cache.json"

    @classmethod
    def load(cls, root: Path) -> SynsortConfig:
        pyproject = root / "pyproject.toml"
        order: list[str] | None = None
        headers: dict[str, str] | None = None
        cache_file = ".synsort_cache.json"

        if pyproject.exists():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            tool_cfg = data.get("tool", {}).get("synsort", {})
            order = tool_cfg.get("order")
            headers = tool_cfg.get("section_headers")
            cache_file = tool_cfg.get("cache_file", cache_file)

        return cls(
            order=_normalize_order(order),
            section_headers=_merge_headers(headers),
            cache_file=cache_file,
        )

    @property
    def signature(self) -> str:
        payload: dict[str, str | list[str] | dict[str, str]] = {
            "order": self.order,
            "section_headers": self.section_headers,
            "cache_file": self.cache_file,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def header_for(self, category: str) -> str:
        return self.section_headers.get(category, f"# --- {category.title()} ---")
