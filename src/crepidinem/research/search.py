"""Web search behind a narrow protocol"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from crepidinem.config import Settings
from crepidinem.exceptions import ResearchError
from crepidinem.logging_setup import get_logger

__all__ = [
    "LiteratureSearch",
    "NullLiteratureSearch",
    "SearchHit",
    "TavilyLiteratureSearch",
    "build_search",
]

log = get_logger(__name__)

_MAX_SNIPPET_CHARS = 1200


@dataclass(frozen=True, slots=True)
class SearchHit:
    """one retrieved document"""

    title: str
    url: str
    snippet: str
    score: float = 0.0

    def as_context(self, index: int) -> str:
        """Render for inclusion in an extraction prompt."""
        return f"[{index}] {self.title}\nURL: {self.url}\n{self.snippet}"


@runtime_checkable
class LiteratureSearch(Protocol):
    """anything that can answer a query with documents"""

    async def search(self, query: str, *, max_results: int = 5) -> tuple[SearchHit, ...]:
        """retrieve documents relevant to query"""
        ...

    async def aclose(self) -> None:
        """release held connections"""
        ...


class NullLiteratureSearch:
    """research is disabled or no key is configured."""

    async def search(self, query: str, *, max_results: int = 5) -> tuple[SearchHit, ...]:
        del max_results
        log.debug("Literature search disabled; ignoring query %r", query)
        return ()

    async def aclose(self) -> None:
        return


class TavilyLiteratureSearch:
    """Tavily backed search"""

    def __init__(
        self,
        api_key: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        search_depth: str = "advanced",
    ) -> None:
        from tavily import AsyncTavilyClient

        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=30.0)
        self._client = AsyncTavilyClient(api_key=api_key, client=self._http)
        self._search_depth = search_depth

    async def search(self, query: str, *, max_results: int = 5) -> tuple[SearchHit, ...]:
        """Run one Tavily search"""

        log.info("Tavily search: %r (max_results=%d)", query, max_results)
        try:
            raw: Any = await self._client.search(
                query,
                search_depth=self._search_depth,
                max_results=max_results,
            )

        except Exception as exc:
            msg = f"Tavily search failed for {query!r}: {exc}"
            raise ResearchError(msg) from exc

        if not isinstance(raw, dict):
            msg = f"Tavily returned {type(raw).__name__}, expected an object."
            raise ResearchError(msg)

        results = raw.get("results")
        if not isinstance(results, list):
            log.warning("Tavily answered without a results array for %r", query)
            return ()

        hits: list[SearchHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=str(item.get("title") or "untitled").strip(),
                    url=url,
                    snippet=str(item.get("content") or "").strip()[:_MAX_SNIPPET_CHARS],
                    score=float(item.get("score") or 0.0),
                )
            )
        log.info("Tavily returned %d usable hit(s) for %r", len(hits), query)
        return tuple(hits)

    async def aclose(self) -> None:
        """close HTTP pool if this object opened it"""
        if self._owns_http:
            await self._http.aclose()


def build_search(settings: Settings) -> LiteratureSearch:
    """pick a search implementation for the current config"""

    if not settings.research_available:
        reason = "disabled" if not settings.research_enabled else "no TAVILY_API_KEY"
        log.info("Literature research is off (%s); using the cell's default limits", reason)
        return NullLiteratureSearch()

    live = TavilyLiteratureSearch(settings.require_tavily_key())
    if not settings.research_cache_enabled:
        return live

    from crepidinem.research.cache import CachedLiteratureSearch

    return CachedLiteratureSearch(
        live,
        directory=Path(settings.research_cache_dir) if settings.research_cache_dir else None,
        ttl_hours=settings.research_cache_ttl_hours,
        refresh=settings.research_cache_refresh,
    )
