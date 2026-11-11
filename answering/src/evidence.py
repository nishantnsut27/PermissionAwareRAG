"""Choose relevant documents and their neighboring passages as one operation."""
from __future__ import annotations

import re

from .query_plan import QueryPlan


def chunk_index(chunk):
    value = chunk.metadata.get("chunk_index")
    if type(value) is int:
        return value
    match = re.search(r"_chunk_(\d+)$", chunk.chunk_id)
    return int(match[1]) if match else None


def source_family(dtype):
    return "Project Document" if dtype in {"Project Document", "Project Status Report",
        "Implementation Plan", "Requirements Document", "Decision Record", "Technical Notes"} else dtype


def matches_source_type(requested, actual):
    return requested == actual or (requested == "Project Document" and source_family(actual) == requested)


def prioritize(outcome, plan: QueryPlan) -> tuple[list, list[dict]]:
    """Keep all eligible alternatives for packing/backfill, not only five chunks.

    A below-threshold passage can complete an adjacent, relevant document. It
    never anchors a new document or account. All inputs have passed the same
    authorization and scope gates. Retrieval order itself is never mutated.
    """
    anchors = {id(chunk) for chunk, _ in outcome.items}
    all_items = list(outcome.scored_items)
    groups = {}
    for pair in all_items:
        groups.setdefault(pair[0].document_id, []).append(pair)
    eligible, omitted = {}, []
    for doc, pairs in groups.items():
        relevant = [pair for pair in pairs if id(pair[0]) in anchors]
        if not relevant:
            omitted.extend({"chunk_id": chunk.chunk_id, "document_id": doc,
                            "reason": "no_relevant_document_anchor", "score": score}
                           for chunk, score in pairs)
            continue
        anchor_indices = {chunk_index(chunk) for chunk, _ in relevant}
        completed = []
        for chunk, score in pairs:
            index = chunk_index(chunk)
            if id(chunk) in anchors or (index is not None and any(
                    other is not None and abs(index - other) == 1 for other in anchor_indices)):
                completed.append((chunk, score))
            else:
                omitted.append({"chunk_id": chunk.chunk_id, "document_id": doc,
                                "reason": "nonadjacent_below_threshold", "score": score})
        eligible[doc] = completed

    def source_priority(chunk):
        dtype = chunk.metadata.get("document_type")
        preferred = (0 if dtype == "Account Overview" else 1) if plan.current_profile else (
            (0 if dtype in {"Incident Report", "Security Incident Report"} else
             1 if dtype == "Support Ticket" else 2) if plan.subject == "incident" else 0)
        return (chunk.document_id not in plan.requested_documents,
                not (not plan.requested_documents and any(matches_source_type(t, dtype) for t in plan.source_types)), preferred)

    def priority(pair):
        chunk, score = pair
        dtype = chunk.metadata.get("document_type")
        summary = ((plan.current_profile and dtype == "Account Overview") or
                   (plan.subject == "incident" and dtype in {"Incident Report", "Security Incident Report"}))
        return (not (summary and chunk_index(chunk) == 1),
                id(chunk) not in anchors, score is None, -(score or 0.0))

    # Only the strongest document per requested seller is reserved. Neighbor
    # completion and unused budget are then shared across the remaining pool.
    heads = {doc: min(pairs, key=priority) for doc, pairs in eligible.items()}
    def document_priority(doc):
        scores = [score for chunk, score in eligible[doc] if id(chunk) in anchors and score is not None]
        return (*source_priority(heads[doc][0]), not scores, -max(scores) if scores else 0)
    documents = sorted(eligible, key=document_priority)
    reservations = []
    for seller in plan.searched_sellers if plan.scope in {"seller", "mixed"} else ():
        choices = [doc for doc in documents if heads[doc][0].metadata.get("seller_id") == seller]
        if choices:
            reservations.append(heads[choices[0]])
    # A requested policy in a mixed question gets the same reservation as an
    # account, provided it has a relevant anchor in the authorized candidate set.
    if plan.scope == "mixed":
        org = [doc for doc in documents if heads[doc][0].metadata.get("seller_id") == "Organization-wide"]
        if org:
            reservations.append(heads[org[0]])
    # Requests for a support history or several formal records need document
    # diversity before any one record consumes the remainder with neighbors.
    reserved_types = set()
    for doc in documents:
        head = heads[doc]
        matches = {t for t in plan.source_types if matches_source_type(t, head[0].metadata.get("document_type"))}
        if (doc in plan.requested_documents or (not plan.requested_documents
                and matches and (plan.source_coverage == "history" or matches - reserved_types))):
            if all(pair[0].document_id != doc for pair in reservations):
                reservations.append(head)
            reserved_types.update(matches)
    if not reservations and documents:
        reservations.append(heads[documents[0]])

    ordered = list(reservations)
    visited_docs = list(dict.fromkeys(pair[0].document_id for pair in reservations))
    def neighbors_for(doc):
        head = heads[doc]
        neighbors = [p for p in eligible[doc] if p[0] is not head[0]]
        neighbors.sort(key=lambda pair: (
            abs((chunk_index(pair[0]) or 0) - (chunk_index(head[0]) or 0)),
            chunk_index(pair[0]) or 0))
        return neighbors

    # Give each required document its nearest continuation before any one
    # consumes the budget with distant sections. Unspent capacity is shared.
    queues = [neighbors_for(doc) for doc in visited_docs]
    for i in range(max((len(queue) for queue in queues), default=0)):
        ordered.extend(queue[i] for queue in queues if i < len(queue))
    for doc in documents:
        if doc not in visited_docs:
            ordered.append(heads[doc])
            ordered.extend(neighbors_for(doc))
    seen, unique = set(), []
    for pair in ordered:
        if pair[0].chunk_id not in seen:
            unique.append(pair)
            seen.add(pair[0].chunk_id)
    return unique, omitted
