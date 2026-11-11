"""Golden dataset and independently reviewed expected authorization facts."""
from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src.users import USERS, User, get_user

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
V11_DATASET_FILE = PACKAGE_ROOT / "golden_dataset.json"
GOLDEN_DATASET_FILE = PACKAGE_ROOT / "golden_dataset_v12.json"
RESULTS_FILE = PACKAGE_ROOT / "results.json"
DOCUMENTS_FILE = (PROJECT_ROOT / "ingestion" / "processed" / "documents"
                  / "documents.jsonl")
ACCESS_MATRIX_FILE = (PROJECT_ROOT / "access_control" / "fixtures"
                      / "expected_access_matrix.json")

CATEGORIES = (
    "authorized_factual",
    "multi_document",
    "cross_seller",
    "follow_up",
    "unauthorized_access",
    "mixed_access",
    "policy_governance",
    "ambiguous",
    "prompt_injection",
)

BEHAVIOURS = ("answer", "partial_answer", "refuse")

METADATA_FIELDS = ("document_id", "document_type", "seller_id", "department",
                   "classification", "created_date", "scenario_id")


class DatasetError(ValueError):
    """The golden dataset is malformed or inconsistent with the permission model."""


@dataclass(frozen=True)
class EvalCase:
    eval_id: str
    user_id: str
    question: str
    category: str
    expected_behavior: str
    expected_answer: str
    expected_facts: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    forbidden_sources: tuple[str, ...] = ()
    conversation_from: str | None = None
    notes: str = ""
    seller_filter: tuple[str, ...] | None = None
    tags: tuple[str, ...] = ()
    source_provenance: tuple[dict, ...] = ()

    @property
    def user(self) -> User:
        return get_user(self.user_id)

    @property
    def user_name(self) -> str:
        return self.user.name


@dataclass
class GoldenDataset:
    dataset_id: str
    version: str
    description: str
    cases: list[EvalCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def by_id(self, eval_id: str) -> EvalCase | None:
        return next((c for c in self.cases if c.eval_id == eval_id), None)

    def category_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in CATEGORIES}
        for case in self.cases:
            counts[case.category] = counts.get(case.category, 0) + 1
        return {k: v for k, v in counts.items() if v}

    def user_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.user_name] = counts.get(case.user_name, 0) + 1
        return counts

    def summary(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "total_cases": len(self.cases),
            "categories": self.category_counts(),
            "users": self.user_counts(),
            "security_cases": sum(
                1 for c in self.cases
                if c.category in ("unauthorized_access", "prompt_injection",
                                  "mixed_access", "cross_seller")),
            "refusal_cases": sum(1 for c in self.cases
                                 if c.expected_behavior == "refuse"),
            "capabilities": {tag: sum(tag in c.tags for c in self.cases)
                             for tag in sorted({t for c in self.cases for t in c.tags})},
            "expected_source_documents": len({s for c in self.cases for s in c.expected_sources}),
        }


class PermissionCatalog:
    """Per-user expected decisions from the reviewed static access fixture."""

    def __init__(self, documents: list[dict] | None = None) -> None:
        load_current_catalog = documents is None
        self.documents = documents if documents is not None else load_documents()
        self.by_document_id = {d["document_id"]: d for d in self.documents}
        matrix = json.loads(ACCESS_MATRIX_FILE.read_text(encoding="utf-8"))
        self.matrix_users = matrix["users"]
        self.matrix_rows = {row["document_id"]: row for row in matrix["documents"]}
        self.matrix = {doc_id: frozenset(row["allowed_user_ids"])
                       for doc_id, row in self.matrix_rows.items()}
        self.malformed_payloads = matrix["malformed_payloads"]
        if load_current_catalog:
            if (len(self.by_document_id) != matrix.get("document_count")
                    or set(self.by_document_id) != set(self.matrix_rows)):
                raise DatasetError("Reviewed authorization fixture does not match current document IDs")
            for document_id, document in self.by_document_id.items():
                expected = self.matrix_rows[document_id]
                for key in METADATA_FIELDS:
                    if document.get(key) != expected.get(key):
                        raise DatasetError(
                            f"Reviewed authorization fixture metadata differs for {document_id}:{key}")
            for user_id, profile in self.matrix_users.items():
                user = get_user(user_id)
                if (user.role != profile["role"]
                        or user.department != profile["department"]
                        or user.clearance_level != profile["clearance"]
                        or (user.has_organization_wide_scope()
                            != profile.get("organization_wide", False))
                        or (not user.has_organization_wide_scope()
                            and sorted(user.seller_scope)
                            != sorted(profile["sellers"]))):
                    raise DatasetError(
                        f"Reviewed authorization fixture identity differs for {user_id}")
        self._authorized = {
            user_id: frozenset(document_id for document_id, users
                               in self.matrix.items()
                               if user_id in users and document_id in self.by_document_id)
            for user_id in self.matrix_users
        }
        fact_policy_path = PACKAGE_ROOT / "fixtures" / "fact_visibility.json"
        self.fact_visibility = json.loads(fact_policy_path.read_text(encoding="utf-8"))
        self.fact_fixture = json.loads((PACKAGE_ROOT / "fixtures" /
                                      "restricted_facts.json").read_text(encoding="utf-8"))

    def restricted_facts(self, user_id: str) -> list[dict]:
        return [row for row in self.fact_fixture["facts"]
                if user_id not in row["allowed_user_ids"]]

    def authorized_documents(self, user_id: str) -> frozenset[str]:
        if user_id not in self.matrix_users:
            return frozenset()
        return self._authorized[user_id]

    def denied_documents(self, user_id: str) -> frozenset[str]:
        return frozenset(self.by_document_id) - self.authorized_documents(user_id)

    def seller_scope(self, user_id: str) -> list[str]:
        profile = self.matrix_users.get(user_id) or {}
        return sorted(profile.get("sellers", []))

    def denial_reason(self, user_id: str, document_id: str) -> str | None:
        document = self.by_document_id.get(document_id)
        if document is None:
            return "unknown_document"
        if user_id in self.matrix.get(document_id, frozenset()):
            return None
        return "not_allowed_by_reviewed_matrix"

    def leakage_policy_status(self, eval_id: str) -> str:
        return self.fact_visibility.get("case_status", {}).get(eval_id, "reviewed")

    def identity(self, user_id: str) -> dict:
        user = get_user(user_id)
        return {
            "user_id": user.user_id,
            "name": user.name,
            "role": user.role,
            "department": user.department,
            "clearance_level": user.clearance_level,
            "seller_scope": self.seller_scope(user_id),
            "authorized_documents": len(self.authorized_documents(user_id)),
            "denied_documents": len(self.denied_documents(user_id)),
        }


def load_documents(path: Path | None = None) -> list[dict]:
    """Phase 2 document catalog, reduced to the fields the engine reads."""
    path = path or DOCUMENTS_FILE
    if not path.is_file():
        raise DatasetError(
            f"Document catalog not found: {path}. Run the Phase 2 ingestion "
            "pipeline first.")
    documents = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            documents.append({key: raw.get(key) for key in METADATA_FIELDS})
    return documents


def _require(raw: dict, key: str, eval_id: str) -> object:
    if key not in raw:
        raise DatasetError(f"{eval_id}: missing required field '{key}'")
    return raw[key]


def _tuple(raw: dict, key: str) -> tuple[str, ...]:
    value = raw.get(key) or []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DatasetError(f"{raw.get('eval_id')}: '{key}' must be a list")
    return tuple(value)


def load(path: Path | None = None) -> GoldenDataset:
    path = path or GOLDEN_DATASET_FILE
    if not path.is_file():
        raise DatasetError(f"Golden dataset not found: {path}")
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise DatasetError("Golden dataset must be an object with a 'cases' list")

    cases = []
    for entry in raw["cases"]:
        eval_id = str(entry.get("eval_id", "<missing eval_id>"))
        cases.append(EvalCase(
            eval_id=eval_id,
            user_id=str(_require(entry, "user_id", eval_id)),
            question=str(_require(entry, "question", eval_id)),
            category=str(_require(entry, "category", eval_id)),
            expected_behavior=str(_require(entry, "expected_behavior", eval_id)),
            expected_answer=str(entry.get("expected_answer", "")),
            expected_facts=_tuple(entry, "expected_facts"),
            expected_sources=_tuple(entry, "expected_sources"),
            forbidden_sources=_tuple(entry, "forbidden_sources"),
            conversation_from=entry.get("conversation_from"),
            notes=str(entry.get("notes", "")),
            seller_filter=_tuple(entry, "seller_filter") if "seller_filter" in entry else None,
            tags=_tuple(entry, "tags"),
            source_provenance=tuple(entry.get("source_provenance", [])),
        ))
    return GoldenDataset(
        dataset_id=str(raw.get("dataset_id", "golden")),
        version=str(raw.get("version", "1.0")),
        description=str(raw.get("description", "")),
        cases=cases,
    )


def validate(dataset: GoldenDataset,
             catalog: PermissionCatalog | None = None) -> list[str]:
    """Return every problem found. An empty list means the spec is coherent."""
    catalog = catalog or PermissionCatalog()
    problems: list[str] = []
    seen: set[str] = set()
    chunks = None

    for case in dataset.cases:
        prefix = f"{case.eval_id}"
        if case.eval_id in seen:
            problems.append(f"{prefix}: duplicate eval_id")
        seen.add(case.eval_id)

        if case.user_id not in USERS:
            problems.append(f"{prefix}: unknown user_id {case.user_id!r}")
            continue
        if case.category not in CATEGORIES:
            problems.append(f"{prefix}: unknown category {case.category!r}")
        if case.expected_behavior not in BEHAVIOURS:
            problems.append(
                f"{prefix}: unknown expected_behavior "
                f"{case.expected_behavior!r}")
        if not case.question.strip():
            problems.append(f"{prefix}: empty question")

        authorized = catalog.authorized_documents(case.user_id)
        denied = catalog.denied_documents(case.user_id)
        if case.seller_filter is not None:
            if not case.seller_filter or any(s not in {"S001", "S002", "S003", "S004"} for s in case.seller_filter):
                problems.append(f"{prefix}: invalid seller_filter")
            excluded = [d for d in case.expected_sources if d in catalog.by_document_id
                        and catalog.by_document_id[d]["seller_id"] not in (*case.seller_filter, "Organization-wide")]
            if excluded:
                problems.append(f"{prefix}: expected sources excluded by seller_filter: {excluded}")
        if case.eval_id in {f"E{i:03}" for i in range(61, 91)}:
            if not case.tags or not case.source_provenance:
                problems.append(f"{prefix}: new cases require tags and source provenance")
            if chunks is None:
                chunks = {r["chunk_id"]: r for r in (json.loads(line) for line in
                    (PROJECT_ROOT / "ingestion/processed/chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines())}
            covered = set()
            for proof in case.source_provenance:
                if not isinstance(proof, dict):
                    problems.append(f"{prefix}: malformed provenance")
                    continue
                chunk = chunks.get(proof.get("chunk_id"), {})
                if (not proof.get("quote") or proof.get("quote") not in chunk.get("text", "")
                        or proof.get("document_id") != chunk.get("document_id")
                        or proof.get("text_sha256") != hashlib.sha256(chunk.get("text", "").encode()).hexdigest()):
                    problems.append(f"{prefix}: source evidence no longer matches the corpus")
                covered.add(proof.get("document_id"))
            if set(case.expected_sources) - covered:
                problems.append(f"{prefix}: missing expected-source provenance")

        unknown = [d for d in case.expected_sources + case.forbidden_sources
                   if d not in catalog.by_document_id]
        if unknown:
            problems.append(f"{prefix}: unknown document ids {sorted(unknown)}")

        # A test that expects a document the engine would deny is a broken
        # test, not a system failure; catching it here keeps the two apart.
        not_allowed = [d for d in case.expected_sources
                       if d in catalog.by_document_id and d not in authorized]
        if not_allowed:
            problems.append(
                f"{prefix}: expected_sources not authorized for "
                f"{case.user_id}: {sorted(not_allowed)}")

        not_denied = [d for d in case.forbidden_sources
                      if d in catalog.by_document_id and d not in denied]
        if not_denied:
            problems.append(
                f"{prefix}: forbidden_sources are actually authorized for "
                f"{case.user_id}: {sorted(not_denied)}")

        if case.expected_behavior == "refuse" and case.expected_sources:
            problems.append(
                f"{prefix}: expected_behavior 'refuse' cannot have "
                "expected_sources")
        if case.expected_behavior != "refuse" and not case.expected_sources:
            problems.append(
                f"{prefix}: expected_behavior {case.expected_behavior!r} "
                "requires at least one expected source")

        if case.conversation_from:
            parent = dataset.by_id(case.conversation_from)
            if parent is None:
                problems.append(
                    f"{prefix}: conversation_from references unknown case "
                    f"{case.conversation_from!r}")
            elif parent.eval_id == case.eval_id or parent.eval_id not in seen:
                problems.append(
                    f"{prefix}: conversation_from must reference an earlier case")
            elif parent.user_id != case.user_id:
                problems.append(
                    f"{prefix}: conversation_from case {parent.eval_id} "
                    "belongs to a different user")
    return problems
