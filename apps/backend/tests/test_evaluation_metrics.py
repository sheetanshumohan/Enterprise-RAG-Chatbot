from knowledge_assistant.application.evaluation.evaluator import _retrieval_metrics, aggregate
from knowledge_assistant.domain.entities import EvaluationResult


def _citation(document_id: str):
    from knowledge_assistant.domain.entities import Citation

    return Citation(chunk_id="c1", document_id=document_id, document_filename="f.txt", snippet="...", score=1.0)


def test_perfect_retrieval_scores_1_0():
    retrieved = {"docA"}
    relevant = {"docA"}
    citations = [_citation("docA")]
    precision, recall, context_precision = _retrieval_metrics(retrieved, relevant, citations)
    assert precision == 1.0
    assert recall == 1.0
    assert context_precision == 1.0


def test_irrelevant_retrieval_scores_0():
    retrieved = {"docX"}
    relevant = {"docA"}
    citations = [_citation("docX")]
    precision, recall, context_precision = _retrieval_metrics(retrieved, relevant, citations)
    assert precision == 0.0
    assert recall == 0.0
    assert context_precision == 0.0


def test_partial_recall_when_only_some_relevant_docs_found():
    retrieved = {"docA"}
    relevant = {"docA", "docB"}
    citations = [_citation("docA")]
    precision, recall, context_precision = _retrieval_metrics(retrieved, relevant, citations)
    assert precision == 1.0
    assert recall == 0.5


def test_no_ground_truth_returns_zeros_not_crash():
    precision, recall, context_precision = _retrieval_metrics({"docA"}, set(), [_citation("docA")])
    assert (precision, recall, context_precision) == (0.0, 0.0, 0.0)


def test_aggregate_computes_averages():
    results = [
        EvaluationResult(query="q1", precision=1.0, recall=1.0, groundedness=0.9, latency_ms=100),
        EvaluationResult(query="q2", precision=0.5, recall=0.5, groundedness=0.7, latency_ms=200),
    ]
    summary = aggregate(results)
    assert summary["n_queries"] == 2
    assert summary["avg_precision"] == 0.75
    assert summary["avg_groundedness"] == 0.8
    assert summary["avg_latency_ms"] == 150.0


def test_aggregate_empty_list_returns_empty_dict():
    assert aggregate([]) == {}
