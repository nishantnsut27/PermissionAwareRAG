"""Token-bounded context packing with provenance derived from included spans."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from functools import lru_cache

from . import config

SEPARATOR = "\n\n----------------------------\n\n"
OMISSION = "\n[... middle omitted ...]\n"


@lru_cache(maxsize=1)
def _encoding():
    # The chosen hosted model has no tiktoken mapping in the checked-in
    # runtime. A UTF-8 byte is a conservative upper bound on model tokens and
    # works fully offline; never let counting trigger an implicit download.
    return _Utf8ByteBound()


class _Utf8ByteBound:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def count_tokens(text: str) -> int:
    """Return a conservative token upper bound (UTF-8 bytes, offline)."""
    return len(_encoding().encode(text))


@dataclass
class Source:
    document_id: str
    document_type: str | None
    label: str
    seller_id: str | None
    department: str | None
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    relevance: float | None = None
    spans: list[dict] = field(default_factory=list, repr=False)

    def citation(self) -> str:
        page = ""
        if self.pages:
            page = (f" - Page {self.pages[0]}" if len(self.pages) == 1
                    else f" - Pages {', '.join(str(p) for p in self.pages)}")
        return f"{self.label}{page}"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "label": self.label,
            "seller_id": self.seller_id,
            "department": self.department,
            "pages": self.pages,
            "chunk_ids": self.chunk_ids,
            "evidence_ids": self.evidence_ids,
            "relevance": (None if self.relevance is None
                          else round(self.relevance, 4)),
            "citation": self.citation(),
        }


def _label(document_id: str, document_type: str | None) -> str:
    return f"{document_type} {document_id}" if document_type else document_id


def _pages(chunk) -> list[int]:
    values = getattr(chunk, "metadata", {}).get("page_numbers") or []
    return sorted({page for page in values
                   if isinstance(page, int) and not isinstance(page, bool) and page > 0})


def _included_pages(chunk, ranges: list[list[int]], trimmed: int = 0) -> list[int]:
    """Map included character ranges to chunk-relative page spans.

    Legacy multi-page chunks have no reliable mapping for an excerpt. In that
    case omit page attribution rather than citing text the model did not see.
    """
    metadata = getattr(chunk, "metadata", {}) or {}
    spans = metadata.get("page_spans")
    if isinstance(spans, list) and spans:
        valid = [span for span in spans if isinstance(span, dict)
                 and type(span.get("page")) is int and span["page"] > 0
                 and type(span.get("start")) is int
                 and type(span.get("end")) is int
                 and 0 <= span["start"] < span["end"] <= len(chunk.text)]
        if len(valid) != len(spans):
            return []
        return sorted({span["page"] for span in valid
                       if any(span["start"] < b + trimmed
                              and span["end"] > a + trimmed for a, b in ranges)})
    pages = _pages(chunk)
    full = ranges == [[0, len((chunk.text or "").strip())]]
    return pages if full or len(pages) == 1 else []



def _evidence_id(chunk_id: str, ranges: list[list[int]], text: str) -> str:
    raw = f"{chunk_id}|{ranges}|{text}".encode("utf-8")
    return "EV-" + hashlib.sha256(raw).hexdigest()[:12]


def build(scored_chunks: list[tuple[object, float | None]],
          cfg: config.AnsweringConfig | None = None, *,
          omissions: list | None = None) -> tuple[str, list[Source]]:
    """Pack complete chunks in priority order, backfilling every unused slot.

    top_k limits distinct documents. Up to twelve chunks share the full token
    and character budgets; no head/tail excerpts or fixed seller partitions can
    silently remove a checklist's middle or strand unused capacity.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    omissions = omissions if omissions is not None else []
    blocks, seen = [], set()
    sources_by_doc = {}
    chunk_count = 0
    for chunk, score in scored_chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        doc = getattr(chunk, "document_id", None) or metadata.get("document_id")
        text = getattr(chunk, "text", "") or ""
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        reason = None
        if not doc or not chunk_id or not text.strip():
            reason = "empty_passage_or_provenance"
        elif chunk_id in seen:
            reason = "duplicate_chunk"
        elif chunk_count >= cfg.max_context_chunks:
            reason = "chunk_limit"
        elif doc not in sources_by_doc and len(sources_by_doc) >= cfg.top_k:
            reason = "document_limit"
        if reason:
            omissions.append({"document_id": doc, "chunk_id": chunk_id, "reason": reason})
            continue

        ranges = [[0, len(text)]]
        pages = _included_pages(chunk, ranges)
        evidence_id = _evidence_id(chunk_id, ranges, text)
        label = _label(doc, metadata.get("document_type"))
        page_text = ", ".join(map(str, pages)) if pages else "unspecified"
        header = f"Evidence [{evidence_id}] | {label} | Pages {page_text}\n"
        block = header + text
        candidate = SEPARATOR.join([*blocks, block])
        if len(candidate) > cfg.max_context_chars or count_tokens(candidate) > cfg.max_context_tokens:
            omissions.append({"document_id": doc, "chunk_id": chunk_id,
                              "reason": "complete_chunk_exceeds_remaining_budget",
                              "characters": len(block), "token_bound": count_tokens(block)})
            continue

        blocks.append(block)
        seen.add(chunk_id)
        chunk_count += 1
        source = sources_by_doc.get(doc)
        if source is None:
            source = Source(doc, metadata.get("document_type"), label,
                            metadata.get("seller_id"), metadata.get("department"), relevance=score)
            sources_by_doc[doc] = source
        elif score is not None and (source.relevance is None or score > source.relevance):
            source.relevance = score
        source.pages = sorted(set(source.pages + pages))
        source.chunk_ids.append(chunk_id)
        source.evidence_ids.append(evidence_id)
        source.spans.append({"evidence_id": evidence_id, "document_id": doc,
                             "chunk_ids": [chunk_id], "pages": pages,
                             "text": text, "offsets": ranges})
    return SEPARATOR.join(blocks), list(sources_by_doc.values())


def build_balanced(scored_chunks: list[tuple[object, float | None]],
                   sellers: list[str], cfg: config.AnsweringConfig, *,
                   omissions: list | None = None) -> tuple[str, list[Source]]:
    """Reserve one passage per seller, then redistribute all unused capacity."""
    reserved = []
    for seller in sellers:
        pair = next((p for p in scored_chunks if p[0].metadata.get("seller_id") == seller), None)
        if pair:
            reserved.append(pair)
    selected = {id(pair[0]) for pair in reserved}
    ordered = reserved + [pair for pair in scored_chunks if id(pair[0]) not in selected]
    return build(ordered, cfg, omissions=omissions)
