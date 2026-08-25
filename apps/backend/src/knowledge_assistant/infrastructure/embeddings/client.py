"""
Pluggable embedding client.

Default: OpenAI text-embedding-3-small (1536 dims) since it's cheap, fast,
and widely available. Swap EMBEDDING_PROVIDER in config to add others
(e.g. Voyage AI, which Anthropic recommends for RAG) -- add a branch here,
the rest of the system only depends on `EmbeddingClient.embed`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingClient(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class GeminiEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001", dim: int = 3072):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = model if model.startswith("models/") else f"models/{model}"
        self.dim = dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import google.generativeai as genai

        results = []
        for text in texts:
            res = genai.embed_content(
                model=self._model,
                content=text,
            )
            emb = res["embedding"]
            if not self.dim:
                self.dim = len(emb)
            results.append(emb)
        return results


def get_embedding_client(
    provider: str,
    api_key: str,
    model: str | None = None,
    dim: int | None = None,
) -> EmbeddingClient:
    if provider == "openai":
        kwargs: dict = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        if dim:
            kwargs["dim"] = dim
        return OpenAIEmbeddingClient(**kwargs)
    if provider == "gemini":
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        if dim:
            kwargs["dim"] = dim
        return GeminiEmbeddingClient(**kwargs)
    raise ValueError(f"Unsupported embedding provider: {provider}")

