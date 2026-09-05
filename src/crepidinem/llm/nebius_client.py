"""client for token factory inference"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, Self, TypedDict, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from crepidinem.config import Settings
from crepidinem.exceptions import (
    LLMError,
    LLMTransportError,
    ReasoningBudgetExhaustedError,
)
from crepidinem.logging_setup import get_logger

__all__ = ["ChatMessage", "LLMResponse", "NebiusLLMClient", "TokenSink"]


TokenSink = Callable[[str, str], None]
log = get_logger(__name__)

#strip reasoning section
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ORPHAN_THINK = re.compile(r"^.*?</think>", re.DOTALL)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    raw_content: str = field(default="", repr=False)

    @property
    def truncated(self) -> bool:
        """true when the model ran out of budget mid-answer"""
        return self.finish_reason == "length"


def _split_reasoning(text: str) -> tuple[str, str]:
    """separate an inline thinking"""
    if "</think>" not in text.lower():
        return text.strip(), ""

    thoughts = [match.group(0) for match in _THINK_BLOCK.finditer(text)]
    stripped = _THINK_BLOCK.sub("", text)

    #everything before tag is reasoning
    orphan = _ORPHAN_THINK.match(stripped)
    if orphan:
        thoughts.append(orphan.group(0))
        stripped = stripped[orphan.end() :]

    return stripped.strip(), "\n".join(t.strip() for t in thoughts).strip()


def _as_openai_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """key our messages into for sdk rules"""
    return [cast("ChatCompletionMessageParam", dict(m)) for m in messages]


class NebiusLLMClient:
    """chat completions against token factory"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = 300.0,
        max_retries: int = 3,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """build client from resolved config"""
        return cls(
            api_key=settings.require_nebius_key(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
        )

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        label: str = "",
        on_token: TokenSink | None = None,
    ) -> LLMResponse:
        who = label or model
        log.info(
            "LLM call [%s] model=%s messages=%d max_tokens=%d stream=%s",
            who,
            model,
            len(messages),
            max_tokens,
            on_token is not None,
        )
        if on_token is not None:
            return await self._complete_streaming(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                who=who,
                on_token=on_token,
            )
        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=_as_openai_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.APIStatusError as exc:
            msg = f"[{who}] {model} returned HTTP {exc.status_code}: {exc.message}"
            raise LLMTransportError(msg, status_code=exc.status_code) from exc
        except openai.APIError as exc:
            msg = f"[{who}] {model} was unreachable: {exc}"
            raise LLMTransportError(msg) from exc
        latency = time.perf_counter() - started

        if not response.choices:
            msg = f"[{who}] {model} returned no choices"
            raise LLMError(msg)

        choice = response.choices[0]
        raw = choice.message.content or ""
        content, inline_reasoning = _split_reasoning(raw)
        reasoning = str(getattr(choice.message, "reasoning_content", "") or "") or inline_reasoning

        usage = response.usage
        result = LLMResponse(
            content=content,
            model=model,
            reasoning=reasoning,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=latency,
            finish_reason=choice.finish_reason,
            raw_content=raw,
        )
        return self._finalise(result, who=who, max_tokens=max_tokens)

    async def _complete_streaming(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        who: str,
        on_token: TokenSink,
    ) -> LLMResponse:
        started = time.perf_counter()
        answer: list[str] = []
        thoughts: list[str] = []
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=_as_openai_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = usage.prompt_tokens or prompt_tokens
                    completion_tokens = usage.completion_tokens or completion_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                trace = str(getattr(delta, "reasoning_content", "") or "")
                if trace:
                    thoughts.append(trace)
                    on_token("reasoning", trace)
                piece = delta.content or ""
                if piece:
                    answer.append(piece)
                    on_token("answer", piece)
        except openai.APIStatusError as exc:
            msg = f"[{who}] {model} returned HTTP {exc.status_code}: {exc.message}"
            raise LLMTransportError(msg, status_code=exc.status_code) from exc
        except openai.APIError as exc:
            msg = f"[{who}] {model} was unreachable: {exc}"
            raise LLMTransportError(msg) from exc

        raw = "".join(answer)
        content, inline_reasoning = _split_reasoning(raw)
        result = LLMResponse(
            content=content,
            model=model,
            reasoning="".join(thoughts).strip() or inline_reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=time.perf_counter() - started,
            finish_reason=finish_reason,
            raw_content=raw,
        )
        return self._finalise(result, who=who, max_tokens=max_tokens)

    def _finalise(self, result: LLMResponse, *, who: str, max_tokens: int) -> LLMResponse:
        """log call and refuse a response that carries no answer"""

        log.info(
            "LLM done [%s] %.2fs tokens=%d/%d finish=%s chars=%d",
            who,
            result.latency_s,
            result.prompt_tokens,
            result.completion_tokens,
            result.finish_reason,
            len(result.content),
        )
        if result.truncated:
            log.warning(
                "[%s] %s hit the %d-token ceiling; the answer is cut off",
                who,
                result.model,
                max_tokens,
            )
        if not result.content.strip():
            msg = (
                f"[{who}] {result.model} returned an empty answer "
                f"(finish_reason={result.finish_reason!r}; "
                f"{result.completion_tokens} completion tokens, all reasoning). "
                f"Raise the token budget for this agent."
            )
            raise ReasoningBudgetExhaustedError(msg)
        return result

    async def aclose(self) -> None:
        """close the underlying connection pool if this client owns it"""
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
