"""Pluggable LLM client: Claude / GPT / Gemini, selected via config."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from knowledge_assistant.infrastructure.observability.langsmith import trace


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str: ...

    @abstractmethod
    async def stream(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> AsyncIterator[str]: ...


class ClaudeClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    @trace(run_type="llm", name="Claude-Complete")
    async def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:

        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    async def stream(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self._model, system=system, messages=messages, max_tokens=max_tokens
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIChatClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    def _to_openai_messages(self, system: str, messages: list[dict]) -> list[dict]:
        return [{"role": "system", "content": system}, *messages]

    @trace(run_type="llm", name="OpenAI-Complete")
    async def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai_messages(system, messages),
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai_messages(system, messages),
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def _to_prompt(self, system: str, messages: list[dict]) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return f"{system}\n\n{transcript}"

    @trace(run_type="llm", name="Gemini-Complete")
    async def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> str:
        response = await self._model.generate_content_async(
            self._to_prompt(system, messages),
            generation_config={"max_output_tokens": max_tokens},
        )
        try:
            return response.text
        except Exception:
            try:
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    return "".join(p.text for p in response.candidates[0].content.parts if hasattr(p, "text"))
            except Exception:
                pass
            return ""

    async def stream(
        self, system: str, messages: list[dict], max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        response = await self._model.generate_content_async(
            self._to_prompt(system, messages),
            generation_config={"max_output_tokens": max_tokens},
            stream=True,
        )
        async for chunk in response:
            try:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text
                elif hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
            except Exception:
                pass


def get_llm_client(provider: str, api_key: str, model: str | None = None) -> LLMClient:
    if provider == "claude":
        return ClaudeClient(api_key, model) if model else ClaudeClient(api_key)
    if provider == "openai":
        return OpenAIChatClient(api_key, model) if model else OpenAIChatClient(api_key)
    if provider == "gemini":
        return GeminiClient(api_key, model) if model else GeminiClient(api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")
