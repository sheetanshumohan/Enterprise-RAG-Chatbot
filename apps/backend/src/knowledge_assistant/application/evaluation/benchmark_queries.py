"""
Sample benchmark queries for the evaluation harness.

In a real deployment these would be hand-labeled against the user's actual
document set (relevant_document_ids populated from ground truth). This
file ships a template + a synthetic example set so `scripts/run_evaluation.py`
is runnable out of the box against demo data (see docs/EVALUATION.md).
"""
from __future__ import annotations

from knowledge_assistant.application.evaluation.evaluator import BenchmarkQuery


def load_benchmark_queries(document_id_map: dict[str, str]) -> list[BenchmarkQuery]:
    """document_id_map maps a human-readable label (e.g. 'employee_handbook')
    to the actual document_id assigned at ingestion time in this environment,
    since document IDs are UUIDs generated at upload time and can't be
    hardcoded."""
    return [
        BenchmarkQuery(
            query="How many paid sick leave days do employees get per year?",
            relevant_document_ids={document_id_map.get("employee_handbook", "")},
        ),
        BenchmarkQuery(
            query="What is the process for requesting reimbursement for a team lunch?",
            relevant_document_ids={document_id_map.get("employee_handbook", "")},
        ),
        BenchmarkQuery(
            query="Summarize the main contribution of the research paper.",
            relevant_document_ids={document_id_map.get("research_paper", "")},
        ),
    ]
