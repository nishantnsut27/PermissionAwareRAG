"""Jina reranking stage.

Security note: this runs AFTER the authorization re-check, so only already
authorized passages are ever sent to the reranker. The reranker orders and
filters by relevance; it can never grant access to a passage, and a reranker
failure degrades relevance quality without weakening authorization.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import requests

from . import config, jina_keys
from .deadline import remaining_timeout
from .http_deadline import provider_deadline, session

logger = logging.getLogger(__name__)
HTTP = session()

MAX_DOC_CHARS = 2000
WINDOW_OVERLAP_CHARS = 300


@dataclass
class RerankOutcome:
    items: list[tuple[object, float | None]]
    applied: bool
    error: str | None = None
    scored_items: list[tuple[object, float | None]] | None = None

    def __post_init__(self):
        if self.scored_items is None:
            self.scored_items = list(self.items)


def evidence_windows(text: str, max_chars: int = MAX_DOC_CHARS,
                    overlap: int = WINDOW_OVERLAP_CHARS) -> list[str]:
    """Split a passage into bounded overlapping windows including its tail."""
    if not isinstance(text, str) or max_chars < 1 or not 0 <= overlap < max_chars:
        raise ValueError("invalid passage window settings")
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    step = max_chars - overlap
    starts = list(range(0, max(1, len(text) - max_chars + 1), step))
    tail_start = len(text) - max_chars
    if not starts or starts[-1] != tail_start:
        starts.append(tail_start)
    return [text[start:start + max_chars] for start in starts]


def rerank(query: str, chunks: list, threshold: float | None = None) -> RerankOutcome:
    """Order authorized chunks by relevance and drop those below threshold.

    Returns the original order with no scores if reranking is unavailable.
    """
    if not chunks:
        return RerankOutcome(items=[], applied=False)
    if (threshold is not None
            and (not isinstance(threshold, (int, float))
                 or isinstance(threshold, bool) or not math.isfinite(threshold)
                 or not 0.0 <= threshold <= 1.0)):
        raise ValueError("rerank threshold must be finite and in [0, 1]")
    if not config.RERANK_ENABLED or not jina_keys.pool().size:
        return RerankOutcome([(c, None) for c in chunks], False, "disabled")

    documents: list[str] = []
    window_to_chunk: list[int] = []
    for chunk_index, chunk in enumerate(chunks):
        windows = evidence_windows(getattr(chunk, "text", "") or "")
        documents.extend(windows)
        window_to_chunk.extend([chunk_index] * len(windows))
    if not documents:
        return RerankOutcome([(c, None) for c in chunks], False, "empty passages")
    try:
        with provider_deadline(config.RERANK_TIMEOUT_S), jina_keys.request('reranking') as call:
            response = HTTP.post(
                config.JINA_RERANK_URL,
                headers=call.headers,
                json={
                    "model": config.JINA_RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
                timeout=remaining_timeout(config.RERANK_TIMEOUT_S),
            )
            call.response(response.status_code)
            try:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Jina rerank error {response.status_code}: "
                        f"{response.text[:200]}")
                body = response.json()
            finally:
                close = getattr(response, "close", None)
                if close:
                    close()
            return _rank_response(chunks, body, window_to_chunk, documents, threshold)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        if jina_keys.strict_evaluation():
            raise
        detail = jina_keys.pool().redact(exc)
        logger.warning("Reranking unavailable, falling back to fusion order: %s",
                       detail)
        return RerankOutcome([(c, None) for c in chunks], False, detail[:200])


def _rank_response(chunks, body, window_to_chunk, documents, threshold):
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise RuntimeError("Jina rerank response must contain a results list")
    results = body["results"]
    if not results:
        raise RuntimeError("Jina rerank returned no results")
    scores_by_chunk: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("invalid rerank item")
        index = item.get("index")
        score = item.get("relevance_score")
        if (not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(documents)
                or not isinstance(score, (int, float)) or isinstance(score, bool)
                or not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise RuntimeError("invalid rerank index or score")
        chunk_index = window_to_chunk[index]
        scores_by_chunk[chunk_index] = max(scores_by_chunk.get(chunk_index, 0.0),
                                           float(score))
    if not scores_by_chunk:
        raise RuntimeError("no usable scores")

    scored = [(chunks[index], score) for index, score in scores_by_chunk.items()]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    selected = scored if threshold is None else [pair for pair in scored if pair[1] >= threshold]
    return RerankOutcome(items=selected, applied=True, scored_items=scored)
