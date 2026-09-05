"""An on-disk cache in front of a :class: LiteratureSearch"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crepidinem.logging_setup import get_logger
from crepidinem.research.search import LiteratureSearch, SearchHit

__all__ = ["CacheStats", "CachedLiteratureSearch", "default_cache_dir"]

log = get_logger(__name__)

#datasheet specifications change on the order of product revisions
DEFAULT_TTL_HOURS = 24 * 14


def _hits_from(stored: list[Any]) -> tuple[SearchHit, ...] | None:
    """rebuild hits from stored json, or None if anything is malformed"""
    hits: list[SearchHit] = []
    for item in stored:
        if not isinstance(item, dict):
            return None
        
        try:
            hits.append(
                SearchHit(
                    title=str(item["title"]),
                    url=str(item["url"]),
                    snippet=str(item["snippet"]),
                    score=float(item.get("score") or 0.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            return None
    return tuple(hits)


def default_cache_dir() -> Path:
    """per user cache location"""
    local = os.environ.get("LOCALAPPDATA")
    root = Path(local) if local else Path.home() / ".cache"
    return root / "crepidinem" / "research-cache"


@dataclass(slots=True)
class CacheStats:
    """billed requests this cache avoided"""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def describe(self, *, credits_per_search: int = 2) -> str:
        """one line a human can read in the unit that is actually billed"""
        if self.total == 0:
            return "No searches were issued"
        
        saved = self.hits * credits_per_search
        spent = self.misses * credits_per_search
        return (
            f"{self.total} search(es): {self.misses} live, {self.hits} from cache "
            f"(~{spent} credit(s) spent, ~{saved} saved)."
        )


class CachedLiteratureSearch:
    """serve repeated queries"""

    def __init__(
        self,
        inner: LiteratureSearch,
        *,
        directory: Path | None = None,
        ttl_hours: float = DEFAULT_TTL_HOURS,
        refresh: bool = False,
    ) -> None:
        self._inner = inner
        self._dir = directory or default_cache_dir()
        self._ttl_s = max(0.0, ttl_hours * 3600.0)
        self._refresh = refresh
        self.stats = CacheStats()

    #keys

    def _path_for(self, query: str, max_results: int) -> Path:
        raw = json.dumps(
            {"q": query.strip().lower(), "n": max_results},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return self._dir / f"{digest}.json"

    #storage

    def _read(self, path: Path) -> tuple[SearchHit, ...] | None:
        """return stored hits"""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        created = payload.get("created")
        if not isinstance(created, (int, float)):
            return None
        if self._ttl_s and (time.time() - created) > self._ttl_s:
            log.debug("Cache entry expired: %s", path.name)
            return None

        stored = payload.get("hits")
        if not isinstance(stored, list):
            return None
        return _hits_from(stored)

    def _write(self, path: Path, query: str, max_results: int, hits: tuple[SearchHit, ...]) -> None:
        """store hits treating any failure as simply not caching"""
        payload: dict[str, Any] = {
            "created": time.time(),
            "query": query,
            "max_results": max_results,
            "hits": [
                {"title": h.title, "url": h.url, "snippet": h.snippet, "score": h.score}
                for h in hits
            ],
        }

        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:12]}.tmp")
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            log.debug("Could not cache %r: %s", query, exc)
            tmp.unlink(missing_ok=True)

    #search

    async def search(self, query: str, *, max_results: int = 5) -> tuple[SearchHit, ...]:
        """answer from disk when possible"""
        path = self._path_for(query, max_results)

        if not self._refresh:
            cached = self._read(path)
            if cached is not None:
                self.stats.hits += 1
                log.info("Search served from cache (no credit spent): %r", query)
                return cached

        hits = await self._inner.search(query, max_results=max_results)
        self.stats.misses += 1

        if hits:
            self._write(path, query, max_results, hits)
        return hits

    async def aclose(self) -> None:
        """close the wrapped search"""
        await self._inner.aclose()
