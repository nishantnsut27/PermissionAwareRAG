"""Permission-aware retrieval: scope filter -> Qdrant -> re-check -> gate."""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indexing.src.retrieval import (
    RetrievalResult,
    retrieve as phase3_retrieve,
    retrieve_partitioned,
)

from .audit import log_request
from .engine import PermissionEngine
from .policy import IdentityError, ScopeDeniedError
from .users import User, get_user

REFUSAL_MESSAGE = (
    "I couldn't find information available to you "
    "that can answer this request."
)

MAX_TOP_K = 40
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 40

_engine = PermissionEngine()


class AuthorizationInvariantError(RuntimeError):
    """Raised if an unauthorized chunk ever reaches the LLM boundary."""


@dataclass
class AccessResponse:
    user_id: str
    query: str
    request_id: str
    chunks: list[RetrievalResult] = field(default_factory=list)
    withheld_count: int = 0
    refused: bool = False
    stage_trace: dict = field(default_factory=dict)

    def build_llm_context(self) -> str:
        """The only sanctioned way to hand retrieved text to an LLM."""
        if self.refused or not self.chunks:
            return REFUSAL_MESSAGE
        return "\n\n".join(
            f"[{chunk.document_id} / {chunk.chunk_id}]\n{chunk.text}"
            for chunk in self.chunks
        )


def _resolve(user: User | str) -> User:
    if isinstance(user, str):
        return get_user(user)
    if isinstance(user, User):
        return user
    raise IdentityError(
        f"user must be a User or a user_id, got {type(user).__name__}")


def authorized_retrieve(user: User | str, query: str, top_k: int = 5,
                        request_id: str | None = None,
                        requested_sellers=None, organization_only=False) -> AccessResponse:
    identity = _resolve(user)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    request_id = request_id or uuid.uuid4().hex[:12]

    # Raises for a malformed identity or a request outside the authorized
    # scope, rather than issuing a widened query.
    scope = _engine.get_authorized_scope(identity, requested_sellers)
    if organization_only:
        scope = {"must": [scope, {"seller_id": {"value": "Organization-wide"}}]}
    candidates = phase3_retrieve(
        query,
        top_k=min(max(top_k * CANDIDATE_MULTIPLIER, 10), MAX_CANDIDATES),
        filters=scope,
    )
    allowed, denied = _engine.authorize_chunks(identity, candidates, request_id)
    allowed = [chunk for chunk in allowed
               if (not organization_only or chunk.metadata.get("seller_id") == "Organization-wide")
               and (requested_sellers is None or chunk.metadata.get("seller_id") in
                    (*requested_sellers, "Organization-wide"))]
    selected = allowed[:top_k]

    for chunk in selected:
        if not _engine.can_access(identity, chunk.metadata).allow:
            raise AuthorizationInvariantError(
                f"chunk {chunk.chunk_id} passed the filter but is not "
                f"authorized for {identity.user_id}")

    response = AccessResponse(
        user_id=identity.user_id,
        query=query,
        request_id=request_id,
        chunks=selected,
        withheld_count=len(denied),
        refused=not selected,
        stage_trace={
            "index_revision": next((chunk.metadata.get("_index_revision")
                                     for chunk in candidates
                                     if chunk.metadata.get("_index_revision")), None),
            "policy_version": "permission-engine-v1",
            "raw_retrieval_candidates": [_chunk_ref(chunk) for chunk in candidates],
            "postfilter_allowed_candidates": [_chunk_ref(chunk) for chunk in allowed],
            "postfilter_denied_candidates": [_chunk_ref(chunk) for chunk in denied],
        },
    )
    log_request(identity.user_id, request_id, len(candidates), len(selected),
                len(denied), response.refused)
    return response


def _chunk_ref(chunk) -> dict:
    return {
        "chunk_id": getattr(chunk, "chunk_id", None),
        "document_id": (getattr(chunk, "document_id", None)
                        or (getattr(chunk, "metadata", None) or {}).get("document_id")),
    }


def authorized_retrieve_balanced(user: User, query: str, sellers: list[str],
                                candidate_limit: int, request_id: str,
                                include_organization: bool = False) -> AccessResponse:
    """Seller-only partitions; no caller-supplied value can grant a seller.

    Candidate quotas sum to the answering pool (and never exceed the existing
    raw candidate ceiling). Organization material cannot consume seller slots.
    """
    identity = _resolve(user)
    _engine.get_authorized_scope(identity)  # validate identity before partitioning
    authorized_sellers = _engine.effective_seller_scope(identity)
    if (not sellers or len(set(sellers)) != len(sellers)
            or any(seller not in authorized_sellers for seller in sellers)):
        raise ScopeDeniedError("balanced sellers are outside the authorized scope")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if (type(candidate_limit) is not int
            or not 1 <= candidate_limit <= MAX_CANDIDATES):
        raise ValueError("invalid balanced candidate limit")
    groups = [*sellers, *(["Organization-wide"] if include_organization else [])]
    quotas = {seller: candidate_limit // len(groups) + (index < candidate_limit % len(groups))
              for index, seller in enumerate(groups)}
    partitions = {
        seller: ({"must": [_engine.get_authorized_scope(identity, None if seller == "Organization-wide" else [seller]),
                           {"seller_id": {"value": seller}}]}, limit)
        for seller, limit in quotas.items()}
    retrieved, revision = retrieve_partitioned(query, partitions)
    raw, allowed, denied, scope_dropped = [], [], [], []
    for seller, limit in quotas.items():
        candidates = retrieved.get(seller, [])[:limit]
        raw.extend(candidates)
        checked, rejected = _engine.authorize_chunks(identity, candidates, request_id)
        denied.extend(rejected)
        for chunk in checked:
            if chunk.metadata.get("seller_id") != seller:
                scope_dropped.append(chunk)
            else:
                allowed.append(chunk)
    # RRF scores use the same query and rank constant in every partition.
    # Keep the candidate trace ranked across partitions, rather than grouped
    # by seller, so fixed-K evaluation does not merely inspect the first shard.
    raw.sort(key=lambda chunk: chunk.score, reverse=True)
    allowed.sort(key=lambda chunk: chunk.score, reverse=True)
    response = AccessResponse(
        identity.user_id, query, request_id, chunks=allowed,
        withheld_count=len(denied) + len(scope_dropped), refused=not allowed,
        stage_trace={
            "index_revision": revision, "policy_version": "permission-engine-v1",
            "retrieval_path": "seller_balanced", "seller_candidate_quotas": quotas,
            "raw_retrieval_candidates": [_chunk_ref(chunk) for chunk in raw],
            "postfilter_allowed_candidates": [_chunk_ref(chunk) for chunk in allowed],
            "postfilter_denied_candidates": [_chunk_ref(chunk) for chunk in denied],
            "partition_scope_dropped": [_chunk_ref(chunk) for chunk in scope_dropped],
        })
    log_request(identity.user_id, request_id, len(raw), len(allowed),
                response.withheld_count, response.refused)
    return response
