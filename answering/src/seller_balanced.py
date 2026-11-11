"""Conservative multi-account routing and bounded per-seller reranking."""
from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor

from . import config, reranker
from .deadline import check_deadline, request_deadline


def relevant_sellers(question: str, effective_sellers: list[str]) -> list[str]:
    """Query text may narrow server scope, never supply or widen identity.

    Ambiguous and single-account questions retain the global baseline. Prior
    turns are deliberately excluded from routing: they are untrusted context.
    """
    explicit = set(re.findall(r"\bS\d{3}\b", question.upper()))
    if explicit:
        relevant = [seller for seller in effective_sellers if seller in explicit]
        return relevant if len(relevant) > 1 else []
    plural = re.search(
        r"\b(?:my|our|all|both|each|multiple|across|compare|comparison\s+of)\s+"
        r"(?:the\s+)?(?:authorized\s+)?seller(?:s|\s+accounts)\b"
        r"|\bseller\s+accounts\b", question, re.IGNORECASE)
    return list(effective_sellers) if plural and len(effective_sellers) > 1 else []


def rerank_sellers(query: str, chunks: list, sellers: list[str],
                   threshold: float) -> reranker.RerankOutcome:
    groups = {seller: [chunk for chunk in chunks
                       if chunk.metadata.get("seller_id") == seller]
              for seller in sellers}

    def rank(seller):
        check_deadline()
        outcome = reranker.rerank(query, groups[seller], threshold=threshold)
        check_deadline()
        # The final independent permission gate still runs in the service.
        # Also constrain the reranker to the exact original partition inputs.
        originals = {id(chunk) for chunk in groups[seller]}
        seen, items = set(), []
        for chunk, score in outcome.items:
            if (id(chunk) in originals and id(chunk) not in seen
                    and chunk.metadata.get("seller_id") == seller):
                seen.add(id(chunk))
                items.append((chunk, score))
        outcome.items = items
        outcome.scored_items = [pair for pair in outcome.scored_items
                                if id(pair[0]) in originals
                                and pair[0].metadata.get("seller_id") == seller]
        return outcome

    active = [seller for seller in sellers if groups[seller]]
    # All reranker calls share ONE existing stage timeout, as well as the
    # enclosing API deadline. No per-seller retries or timeout multiplication.
    with request_deadline(config.RERANK_TIMEOUT_S):
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="seller-rerank") as pool:
            futures = {seller: pool.submit(contextvars.copy_context().run, rank, seller)
                       for seller in active}
            outcomes = {seller: future.result() for seller, future in futures.items()}

    supported = [seller for seller in active if outcomes[seller].items]
    # Reserve the strongest surviving passage from each seller first. Fill
    # remaining slots by relevance; absent scores preserve partition order.
    heads = [outcomes[seller].items[0] for seller in supported]
    heads.sort(key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
    tails = [pair for seller in supported for pair in outcomes[seller].items[1:]]
    tails.sort(key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
    applied = bool(outcomes) and all(outcome.applied for outcome in outcomes.values())
    scored = [pair for seller in active for pair in outcomes[seller].scored_items]
    scored.sort(key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
    return reranker.RerankOutcome(heads + tails, applied,
                                  None if applied else "seller reranking degraded", scored)
