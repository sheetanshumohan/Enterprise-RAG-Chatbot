"""
Evaluation framework.

Two families of metrics, computed per benchmark query:

Retrieval metrics (computed against a human-labeled `relevant_document_ids`
set, no LLM needed -- fast, deterministic, cheap to run on every commit):
  - precision:          fraction of retrieved chunks that came from a relevant document
  - recall:             fraction of relevant documents that were represented in retrieval
  - context_precision:  precision restricted to the top-ranked chunk (is #1 actually relevant?)

Generation metrics (LLM-as-judge, using the same pluggable LLMClient --
slower and non-deterministic, so these are for a benchmark suite / CI
smoke run, not per-request):
  - groundedness:       fraction of answer sentences supported by the retrieved context
  - answer_relevance:   does the answer actually address the question asked
  - hallucination_rate: 1 - groundedness (kept as a separate named metric since
                         it's what's called out explicitly in reporting/dashboards)

latency_ms is just measured wall-clock time from the orchestrator, no
judging involved.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from knowledge_assistant.application.use_cases.ask_question import AskQuestionUseCase
from knowledge_assistant.domain.entities import EvaluationResult
from knowledge_assistant.domain.repositories import EvaluationRepository
from knowledge_assistant.infrastructure.llm.client import LLMClient


@dataclass
class BenchmarkQuery:
    query: str
    relevant_document_ids: set[str]
    collection_id: str | None = None


JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a RAG system. Given a question, the
context chunks the system retrieved, and the answer it generated, score the answer on two
axes from 0.0 to 1.0:
  - groundedness: what fraction of factual claims in the answer are directly supported by
    the provided context? (1.0 = every claim is supported, 0.0 = answer is entirely
    unsupported/fabricated)
  - answer_relevance: does the answer actually address what was asked, regardless of
    correctness? (1.0 = fully on-topic and responsive, 0.0 = off-topic or non-answer)
Respond ONLY with JSON: {"groundedness": float, "answer_relevance": float, "reasoning": string}
"""


class RagEvaluator:
    def __init__(self, ask_use_case: AskQuestionUseCase, judge_llm: LLMClient, eval_repo: EvaluationRepository):
        self._ask = ask_use_case
        self._judge = judge_llm
        self._eval_repo = eval_repo

    async def evaluate_query(
        self, session_id: str, user_id: str, benchmark: BenchmarkQuery
    ) -> EvaluationResult:
        t0 = time.monotonic()
        result = await self._ask.run(
            session_id=session_id,
            user_id=user_id,
            collection_id=benchmark.collection_id,
            question=benchmark.query,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        retrieved_doc_ids = {c.document_id for c in result.citations}
        precision, recall, context_precision = _retrieval_metrics(
            retrieved_doc_ids, benchmark.relevant_document_ids, result.citations
        )
        groundedness, answer_relevance = await self._judge_answer(
            benchmark.query, result.answer, [c.snippet for c in result.citations]
        )

        evaluation = EvaluationResult(
            query=benchmark.query,
            precision=precision,
            recall=recall,
            groundedness=groundedness,
            context_precision=context_precision,
            answer_relevance=answer_relevance,
            hallucination_rate=round(1.0 - groundedness, 4),
            latency_ms=latency_ms,
        )
        return await self._eval_repo.save(evaluation)

    async def _judge_answer(self, question: str, answer: str, context_snippets: list[str]) -> tuple[float, float]:
        if not answer.strip():
            return 0.0, 0.0
        context_block = "\n---\n".join(context_snippets) or "(no context retrieved)"
        raw = await self._judge.complete(
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:\n{answer}",
                }
            ],
            max_tokens=300,
        )
        data = _safe_json(raw, default={"groundedness": 0.5, "answer_relevance": 0.5})
        return float(data.get("groundedness", 0.5)), float(data.get("answer_relevance", 0.5))


def _retrieval_metrics(
    retrieved_doc_ids: set[str], relevant_doc_ids: set[str], citations: list
) -> tuple[float, float, float]:
    if not relevant_doc_ids:
        # No ground truth for this query -- can't score retrieval, only generation.
        return 0.0, 0.0, 0.0

    true_positives = retrieved_doc_ids & relevant_doc_ids
    precision = len(true_positives) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
    recall = len(true_positives) / len(relevant_doc_ids)

    context_precision = 0.0
    if citations:
        context_precision = 1.0 if citations[0].document_id in relevant_doc_ids else 0.0

    return round(precision, 4), round(recall, 4), round(context_precision, 4)


def _safe_json(raw: str, default: dict) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return default


def aggregate(results: list[EvaluationResult]) -> dict:
    """Summary stats across a benchmark run -- what a CI job or dashboard would show."""
    if not results:
        return {}
    n = len(results)
    return {
        "n_queries": n,
        "avg_precision": round(sum(r.precision for r in results) / n, 4),
        "avg_recall": round(sum(r.recall for r in results) / n, 4),
        "avg_context_precision": round(sum(r.context_precision for r in results) / n, 4),
        "avg_groundedness": round(sum(r.groundedness for r in results) / n, 4),
        "avg_answer_relevance": round(sum(r.answer_relevance for r in results) / n, 4),
        "avg_hallucination_rate": round(sum(r.hallucination_rate for r in results) / n, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
        "p95_latency_ms": round(sorted(r.latency_ms for r in results)[int(n * 0.95) if n > 1 else 0], 1),
    }
