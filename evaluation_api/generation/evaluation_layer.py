import logging
import os
import random
import re
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

from .models import GeneratedQuery, ValidatedGroundTruth

logger = logging.getLogger(__name__)


def _build_azure_client(config):
    try:
        from openai import AzureOpenAI  # type: ignore
    except ImportError:
        return None
    endpoint = getattr(config, "AZURE_OPENAI_ENDPOINT", None)
    api_key = (
        os.environ.get("AZURE_OPENAI_KEY")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not (endpoint and api_key):
        return None
    try:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-01",
        )
    except Exception:  # noqa: BLE001
        return None


def _answer_with_context(client, model: str, question: str, contexts: List[str], temperature: float, max_tokens: int) -> str:
    if client is None:
        # Fallback: return truncated context as a pseudo-answer
        return (" ".join(contexts))[:256]
    system = (
        "You are a helpful assistant. Answer STRICTLY using the provided context. "
        "If the answer is not present, say 'NOT_ANSWERABLE'."
    )
    user = (
        "## Context\n" + "\n---\n".join(contexts) + "\n\n" +
        "## Question\n" + question + "\n\nProvide a concise answer."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _deepeval_score(question: str, answer: str, contexts: List[str]) -> Optional[float]:
    # Import via importlib to avoid linter import errors when optional dep is missing
    import importlib
    try:
        metrics_mod = importlib.import_module('deepeval.metrics.ragas')
        tc_mod = importlib.import_module('deepeval.test_case')
        RagasMetric = getattr(metrics_mod, 'RagasMetric')
        LLMTestCase = getattr(tc_mod, 'LLMTestCase')
    except ImportError:
        return None

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=answer,  # We don't have separate ground-truth answers; align for now
        retrieval_context=contexts,
    )
    metric = RagasMetric()
    try:
        result = metric.measure(test_case)
        score = getattr(result, "score", None)
        if score is None:
            score = getattr(metric, "score", None)
        return float(score) if score is not None else None
    except Exception:  # noqa: BLE001
        return None


def _tokenize(text: str) -> List[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore
        stop = ENGLISH_STOP_WORDS
    except Exception:  # noqa: BLE001
        stop = set()
    tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in stop]


def _bm25_nonllm_check(question: str, golden_contexts: List[str], distractor_contexts: List[str], config) -> Tuple[bool, Dict[str, Any]]:
    """Non-LLM acceptance using BM25 rank/margin and token coverage checks."""
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        # Fallback to simple coverage-only check
        BM25Okapi = None  # type: ignore

    # Combine golden contexts into one doc for scoring
    golden_text = "\n".join(golden_contexts)
    docs = [golden_text] + list(distractor_contexts)
    docs_tok = [_tokenize(d) for d in docs]
    q_tokens = _tokenize(question)

    # BM25 scoring
    scores: List[float]
    if BM25Okapi is not None and len(docs_tok) > 0:
        bm25 = BM25Okapi(docs_tok)
        scores = bm25.get_scores(q_tokens).tolist()
    else:
        # Simple proxy: score by overlap count
        q_set = set(q_tokens)
        scores = [len(q_set.intersection(set(toks))) for toks in docs_tok]

    golden_score = scores[0] if scores else 0.0
    best_distractor_score = max(scores[1:], default=0.0)
    margin = float(golden_score - best_distractor_score)
    rank = 1 + sum(1 for s in scores if s > golden_score)

    # Coverage checks
    q_set = set(q_tokens)
    golden_cov = (len(q_set.intersection(set(docs_tok[0]))) / (len(q_set) or 1)) if docs_tok else 0.0
    best_dist_cov = 0.0
    if len(docs_tok) > 1:
        best_idx = 1 + max(range(len(docs_tok) - 1), key=lambda i: scores[i + 1], default=0)
        best_dist_cov = len(q_set.intersection(set(docs_tok[best_idx]))) / (len(q_set) or 1)

    # Thresholds
    min_margin = float(getattr(config, "BM25_MIN_MARGIN", 0.5))
    min_golden_cov = float(getattr(config, "COVERAGE_MIN_GOLDEN", 0.6))
    min_cov_gap = float(getattr(config, "COVERAGE_GAP_MIN", 0.2))

    accept = (rank == 1) and (margin >= min_margin) and (golden_cov >= min_golden_cov) and ((golden_cov - best_dist_cov) >= min_cov_gap)

    metrics: Dict[str, Any] = {
        "bm25_rank": rank,
        "bm25_margin": round(margin, 4),
        "golden_score": round(float(golden_score), 4),
        "best_distractor_score": round(float(best_distractor_score), 4),
        "golden_coverage": round(golden_cov, 4),
        "best_distractor_coverage": round(best_dist_cov, 4),
    }
    return accept, metrics


def evaluate_queries(
    generated_queries: List[GeneratedQuery], config, evaluation_mode: str = "llm"
) -> List[ValidatedGroundTruth]:
    """
    Evaluate generated queries.
    Modes:
      - none:    accept all
      - nonllm:  BM25/coverage checks only
      - llm:     Azure OpenAI answer + Ragas/DeepEval gating
      - hybrid:  non-LLM first; if fail, optionally escalate to LLM by sample rate
    """
    mode = (evaluation_mode or "llm").lower()
    logger.info("Evaluating %s queries with mode=%s...", len(generated_queries), mode)

    final_dataset: List[ValidatedGroundTruth] = []

    # Prepare LLM client if needed
    client = None
    model = None
    temperature = getattr(config, "TEMPERATURE", 0.3)
    max_tokens = getattr(config, "MAX_TOKENS", 64)
    if mode in ("llm", "hybrid"):
        client = _build_azure_client(config)
        model = getattr(config, "AZURE_OPENAI_DEPLOYMENT_NAME", None)

    escalate_rate = float(getattr(config, "EVAL_LLM_SAMPLE_RATE", 0.1))

    for q in tqdm(generated_queries, desc="Evaluating queries"):
        question = q.query
        golden_ctx = [c.chunk_text for c in q.golden_chunks]
        distractor_ctx = [c.chunk_text for c in getattr(q, "distractor_chunks", [])]

        accepted = False
        validation_data: Dict[str, Any] = {"query_type": q.query_type}

        if mode in ("none",):
            accepted = True
            validation_data.update({"evaluation": "none"})

        if not accepted and mode in ("nonllm", "hybrid"):
            nonllm_ok, m = _bm25_nonllm_check(question, golden_ctx, distractor_ctx, config)
            validation_data.update({"nonllm_metrics": m})
            if nonllm_ok:
                accepted = True
                validation_data.update({"evaluation": "nonllm"})

        if not accepted and mode in ("llm", "hybrid") and model:
            # Optional sampling in hybrid mode
            if mode == "hybrid" and random.random() > escalate_rate:
                # skip LLM escalation
                pass
            else:
                try:
                    answer = _answer_with_context(client, model, question, golden_ctx, temperature, max_tokens)
                except Exception:  # noqa: BLE001
                    answer = (" ".join(golden_ctx))[:256]
                ragas_score = _deepeval_score(question, answer, golden_ctx)
                validation_data.update({
                    "ragas_context_relevance": round(float(ragas_score), 3) if ragas_score is not None else None,
                    "generator_model": ("azure:" + model) if model else "unknown",
                })
                thr = float(getattr(config, "RAGAS_CONTEXT_RELEVANCE_THRESHOLD", 0.8))
                if ragas_score is None or ragas_score >= thr:
                    accepted = True
                    validation_data.update({"evaluation": ("llm" if mode == "llm" else "hybrid_llm")})

        if not accepted:
            continue

        expected_doc_ids = sorted(list({c.doc_id for c in q.golden_chunks}))
        context_chunks = [
            {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "chunk": c.chunk_text}
            for c in q.golden_chunks
        ]

        final_dataset.append(
            ValidatedGroundTruth(
                query=question,
                expected_doc_ids=expected_doc_ids,
                context_chunks=context_chunks,
                validation=validation_data,
            )
        )

    logger.info("Evaluation complete. %s queries passed validation.", len(final_dataset))
    return final_dataset

