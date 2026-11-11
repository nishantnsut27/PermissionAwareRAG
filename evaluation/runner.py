"""Explicit, paced evaluation runs. --plan and --validate-only are offline.

Checkpoints are atomic and resumable only for matching inputs. Exhausted quota
stops the batch at its checkpoint. Existing results are never replaced implicitly.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answering.src import config
from answering.src.conversation import parse_history
from indexing.src import config as index_config
from indexing.src import retrieval as index_retrieval
from . import dataset as ds
from . import evaluator, metrics

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 6
CASE_DELAY_S = float(os.getenv("EVAL_CASE_DELAY_S", "90") or 90)
RETRY_WAIT_S = float(os.getenv("EVAL_RETRY_WAIT_S", "90") or 90)
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "0") or 0)
MAX_CASE_ATTEMPTS = 4
PRE_RETRY_RUNNER_SHA256 = "a19971eb83cc69d764a977b534acd0473f58eddcc40d689cc33f8bbe0b9a5c2e"
PRE_JINA_ROTATION_SHA256 = "04595f66f26de09d0c733db99902cee093e29c78a27349b5defc5e29a6faca85"
JINA_REQUEST_POLICY = "round_robin_stop_on_error_v1"
GROQ_REQUEST_POLICY = "round_robin_provider_cooldown_v1"
PRE_TARGETED_RETRIES_SHA256 = "245f0f480edf5f620ab8415d2b80ddc42213ba55cccd826c1eb73034e3dd0aca"


def conversation_history(case, dataset, rows):
    """Use actual completed prerequisite answers, oldest first, with UI limits."""
    chain = []
    parent_id = case.conversation_from
    while parent_id:
        parent = rows.get(parent_id)
        if (not parent or parent.get("outcome") != "answer"
                or not parent.get("answered") or not parent.get("answer")):
            raise ds.DatasetError(f"Dependency {parent_id} produced no completed answer")
        chain.append(parent)
        parent_id = dataset.by_id(parent_id).conversation_from
    turns = [turn for row in reversed(chain) for turn in (
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]})]
    return [asdict(t) for t in parse_history(turns[-config.DEFAULT_ANSWERING.max_history_turns:])]


def resolve_index_snapshot() -> dict:
    return index_retrieval.revision_snapshot()


def verify_index_snapshot(snapshot: dict) -> None:
    if resolve_index_snapshot() != snapshot:
        raise ds.DatasetError("Active index revision/artifacts or revocations changed during evaluation")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_quota_error(message: str | None) -> bool:
    return bool(re.search(r"\bquota\b|\bquota_(?:exceeded|exhausted)\b|\b429\b|rate[- ]limit|noavailablekeyerror",
                          message or "", re.IGNORECASE))


def case_finished(row: dict) -> bool:
    """A scored outcome; failed outcomes may be retried on the next invocation."""
    return row.get("status") in (evaluator.PASS, evaluator.FAIL) and not row.get("error")


def attempt_count(row: dict) -> int:
    return len(row.get("attempts") or [])


def attempt_limits(previous: dict | None) -> dict[str, int]:
    """Explicit, report-specific fifth attempts; other cases retain four."""
    if not previous:
        return {}
    try:
        manifest = json.loads((ds.PACKAGE_ROOT / "fixtures/groq_rotation_checkpoints.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(previous.get("configuration", {}).get("case_attempt_limits", {}))
    return dict(manifest.get("attempt_extensions", {}).get(previous.get("run_id"), {}))


def attempt_limit(case_id: str, limits: dict | None = None) -> int:
    value = (limits or {}).get(case_id, MAX_CASE_ATTEMPTS)
    return value if type(value) is int and value > 0 else MAX_CASE_ATTEMPTS


def targeted_retry_plan(previous: dict) -> dict:
    """Freeze one extra attempt for errors and FAIL rows with <=2 attempts.

    Repeating the command consumes only the remaining frozen allowances. It
    never grants another extension after an interruption or provider failure.
    """
    if previous.get("targeted_retry") is not None:
        return copy.deepcopy(previous["targeted_retry"])
    rows = previous["results"]
    errors = [r for r in rows if r.get("status") == evaluator.ERROR or r.get("error")]
    failures = [r for r in rows if r.get("status") == evaluator.FAIL
                and not r.get("error") and attempt_count(r) <= 2]
    wanted = {r["eval_id"] for r in errors + failures}
    return {"policy": "errors_and_failures_at_most_two_attempts_v1", "created_at": _now(),
            "error_ids": [r["eval_id"] for r in errors],
            "failure_ids": [r["eval_id"] for r in failures],
            "attempt_caps": {r["eval_id"]: attempt_count(r) + 1
                             for r in rows if r["eval_id"] in wanted}}


def retry_candidate(row: dict) -> bool:
    return bool(row) and (row.get("status") != evaluator.PASS or bool(row.get("error")))


def execution_order(cases, rows, limits=None):
    """A fixed queue per invocation: fresh cases, then earlier failures once.

    Descendants of a retried case follow it so they cannot consume stale
    prerequisite answers. Cases failing in this invocation wait for a restart.
    """
    tail = {c.eval_id for c in cases if retry_candidate(rows.get(c.eval_id, {}))
            and attempt_count(rows[c.eval_id]) > 0
            and attempt_count(rows[c.eval_id]) < attempt_limit(c.eval_id, limits)}
    for case in cases:  # Dataset validation guarantees parents precede children.
        if case.conversation_from in tail:
            tail.add(case.eval_id)
    return [c for c in cases if c.eval_id not in tail] + [c for c in cases if c.eval_id in tail]


def print_case_result(case, row, limits=None, retry_quality=True):
    limit = attempt_limit(case.eval_id, limits)
    print(f"{case.eval_id} {case.category}: {row['status']} "
          f"(attempt {attempt_count(row)}/{limit})", flush=True)
    reasons = list(row.get("failure_reasons") or [])
    if row.get("error"):
        reasons.append(row["error"])
    if retry_candidate(row) and not reasons:
        reasons.append("Case did not pass; inspect its saved scores and evidence.")
    for reason in dict.fromkeys(reasons):
        print(f"  Why: {reason}", flush=True)
    if retry_candidate(row):
        if not retry_quality and case_finished(row):
            print("  Quality failure preserved; targeted runs do not retry it.", flush=True)
            return
        print("  Attempt limit reached; retained as failed/unresolved."
              if attempt_count(row) >= limit else
              "  Queued at the end for the next explicit rerun; no immediate retry.", flush=True)


def select_cases(dataset: ds.GoldenDataset, limit=None, only=None) -> list[ds.EvalCase]:
    if limit is not None and limit <= 0:
        raise ds.DatasetError("--limit must be positive")
    cases = dataset.cases
    if only is not None:
        wanted = {value.strip().upper() for value in only if value.strip()}
        unknown = wanted - {case.eval_id.upper() for case in cases}
        if unknown or not wanted:
            raise ds.DatasetError(f"Unknown or empty case selection: {sorted(unknown)}")
        cases = [case for case in cases if case.eval_id.upper() in wanted]
    if limit is not None:
        cases = cases[:limit]
    wanted = {case.eval_id for case in cases}
    for case in list(cases):
        parent = case.conversation_from
        visited = {case.eval_id}
        while parent:
            if parent in visited or dataset.by_id(parent) is None:
                raise ds.DatasetError("Invalid conversation dependency")
            visited.add(parent)
            wanted.add(parent)
            parent = dataset.by_id(parent).conversation_from
    selected = [case for case in dataset.cases if case.eval_id in wanted]
    if not selected:
        raise ds.DatasetError("No cases selected")
    return selected


def input_manifest(dataset: ds.GoldenDataset, catalog: ds.PermissionCatalog,
                run_config: dict | None = None, *, _source_overrides=None,
                _source_excludes=()) -> dict:
    """Signature of run controls and the resolved immutable index artifacts."""
    snapshot = (run_config or {}).get("index_snapshot") or resolve_index_snapshot()
    settings = {
        "schema": SCHEMA_VERSION, "dataset": asdict(dataset),
        "scoring_policy": evaluator.SCORING_POLICY,
        "run_configuration": run_config or {},
        "documents": catalog.documents, "answering": asdict(config.DEFAULT_ANSWERING),
        "retrieval": asdict(index_config.DEFAULT_RETRIEVAL),
        "model": config.GROQ_MODEL, "temperature": config.GROQ_TEMPERATURE,
        "max_tokens": config.GROQ_MAX_TOKENS,
        "reasoning_effort": config.GROQ_REASONING_EFFORT,
        "rerank_model": config.JINA_RERANK_MODEL, "rerank_enabled": config.RERANK_ENABLED,
        "embedding_model": index_config.JINA_EMBEDDING_MODEL,
        "collection": index_config.QDRANT_COLLECTION_NAME,
        "answering_config": asdict(config.DEFAULT_ANSWERING),
        "retrieval_config": asdict(index_config.DEFAULT_RETRIEVAL),
        "embedding_endpoint": index_config.JINA_API_URL,
        "rerank_endpoint": config.JINA_RERANK_URL,
        "generation_endpoint": config.GROQ_API_URL,
        "judge_tokens": evaluator.judge.JUDGE_MAX_TOKENS,
        "thresholds": {"recall": evaluator.MIN_RECALL, "groundedness": evaluator.MIN_GROUNDEDNESS,
                       "correctness": evaluator.MIN_CORRECTNESS, "relevancy": evaluator.MIN_RELEVANCY},
        "qdrant_endpoint": index_config.QDRANT_URL,
        "case_top_k": evaluator.CASE_TOP_K,
        "index_snapshot": snapshot,
    }
    hashes = {}
    files = []
    for folder in ("evaluation", "access_control/src", "answering/src", "indexing/src"):
        files.extend((ds.PROJECT_ROOT / folder).glob("*.py"))
    files.extend([index_config.PHASE2_CHUNKS_FILE, index_config.VOCAB_FILE])
    files.extend([ds.ACCESS_MATRIX_FILE,
                  ds.GOLDEN_DATASET_FILE,
                  ds.V11_DATASET_FILE,
                  ds.PACKAGE_ROOT / "fixtures" / "fact_visibility.json",
                  ds.PACKAGE_ROOT / "fixtures" / "restricted_facts.json",
                  ds.PACKAGE_ROOT / "fixtures" / "remediation_v12_review.json",
                  ds.PROJECT_ROOT / "scripts" / "run_evaluation.py"])
    files.extend((ds.PROJECT_ROOT / "frontend/src").glob("*"))
    for path in sorted(set(files)):
        if path.is_dir():
            continue
        relative = Path(os.path.relpath(path, ds.PROJECT_ROOT))
        if relative.as_posix() in _source_excludes:
            continue
        override = (_source_overrides or {}).get(relative.as_posix())
        content = override if override is not None else path.read_bytes() if path.is_file() else b"<missing>"
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return {"settings": settings, "files": hashes}


def fingerprint(dataset, catalog, run_config=None, **kwargs):
    manifest = input_manifest(dataset, catalog, run_config, **kwargs)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def changed_components(previous, current):
    old = previous.get("input_manifest") or {}
    changes = []
    for section in ("files", "settings"):
        before, after = old.get(section, {}), current.get(section, {})
        changes.extend(f"{section}.{key}" for key in sorted(set(before) | set(after))
                       if before.get(key) != after.get(key))
    return changes or ["checkpoint metadata or selection"]


def compatible_scheduler_checkpoint(previous, dataset, catalog, current_config):
    """Allow only the requested scheduler upgrade; scoring inputs stay exact.

    Recompute the old fingerprint with its archived, checksum-pinned scheduler.
    Every other source file, model setting, threshold, corpus and index still
    participates in the match. Old attempt fingerprints are never rewritten.
    """
    old_config = previous.get("configuration")
    expected = {k: v for k, v in current_config.items() if k not in {"fingerprint", "max_case_attempts"}}
    if old_config != expected:
        return False
    archive = ds.PACKAGE_ROOT / "fixtures" / "runner_before_deferred_retries.py.txt"
    try:
        source = archive.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(source).hexdigest() != PRE_RETRY_RUNNER_SHA256:
        return False
    return previous.get("fingerprint") == fingerprint(
        dataset, catalog, old_config, _source_overrides={"evaluation/runner.py": source})


def compatible_jina_checkpoint(previous, dataset, catalog, current_config):
    """Allow the recorded transport-policy upgrade, never arbitrary code drift.

    Both the old sources and the reviewed new source hashes are retained. Only
    this known change may bridge fingerprints; all other evaluated inputs must
    still match. Credential values never enter either signature.
    """
    old_config = previous.get('configuration')
    if (current_config.get('max_case_attempts') != 4
            or current_config.get('jina_request_policy') != JINA_REQUEST_POLICY):
        return False
    expected = {k: v for k, v in current_config.items()
                if k not in {'fingerprint', 'jina_request_policy'}}
    expected['max_case_attempts'] = 3
    if old_config != expected:
        return False
    try:
        archive = (ds.PACKAGE_ROOT / 'fixtures/before_jina_rotation.json').read_bytes()
        if hashlib.sha256(archive).hexdigest() != PRE_JINA_ROTATION_SHA256:
            return False
        before = json.loads(archive)
        target = json.loads((ds.PACKAGE_ROOT / 'fixtures/jina_rotation_target.json').read_bytes())
        if set(target) != set(before['sources']) | set(before['added']):
            return False
        for name, digest in target.items():
            source = (ds.PROJECT_ROOT / name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(source).hexdigest() != digest:
                return False
        overrides = {name: base64.b64decode(source, validate=True)
                     for name, source in before['sources'].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return previous.get('fingerprint') == fingerprint(dataset, catalog, old_config,
        _source_overrides=overrides, _source_excludes=before['added'])


def compatible_groq_checkpoint(previous, current_config):
    """Bridge only the reviewed checkpoint/input pair for this transport update."""
    if current_config.get("groq_request_policy") != GROQ_REQUEST_POLICY:
        return False
    old_config = previous.get("configuration") or {}
    expected = {key: value for key, value in current_config.items() if key != "fingerprint"}
    for key in ("groq_request_policy", "case_attempt_limits"):
        if key not in old_config:
            expected.pop(key, None)
    if old_config != expected:
        return False
    try:
        allowed = json.loads((ds.PACKAGE_ROOT / "fixtures/groq_rotation_checkpoints.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(allowed, dict) and {"from_fingerprint": previous.get("fingerprint"),
            "to_fingerprint": current_config.get("fingerprint")} in allowed.get("checkpoints", [])


def compatible_targeted_checkpoint(previous, dataset, catalog, current_config):
    """Verify the archived scheduler while keeping every scoring input exact."""
    controls = {k: v for k, v in current_config.items() if k != "fingerprint"}
    if previous.get("configuration") != controls:
        return False
    try:
        source = (ds.PACKAGE_ROOT / "fixtures/runner_before_targeted_retries.py.txt").read_bytes()
    except OSError:
        return False
    if hashlib.sha256(source).hexdigest() != PRE_TARGETED_RETRIES_SHA256:
        return False
    return previous.get("fingerprint") == fingerprint(
        dataset, catalog, controls, _source_overrides={"evaluation/runner.py": source})


def _matrix_row(result: dict) -> dict:
    retrieval = result.get("retrieval") or {}
    generation = result.get("generation") or {}
    security = result.get("security") or {}
    passed = security.get("passed")
    return {
        **{key: result.get(key, "") for key in
           ("eval_id", "user", "category", "expected_behavior", "actual_behavior", "status",
            "outcome", "finish_reason")},
        "recall_at_k": retrieval.get("recall_at_k"),
        "precision_at_k": retrieval.get("precision_at_k"), "mrr": retrieval.get("mrr"),
        "groundedness": generation.get("groundedness"),
        "relevancy": generation.get("answer_relevancy"),
        "correctness": generation.get("answer_correctness"),
        "security": "PASS" if passed is True else "FAIL" if passed is False else None,
    }


def save(document: dict, path: Path | None = None) -> Path:
    path = Path(path or ds.RESULTS_FILE).resolve()
    if path == ds.GOLDEN_DATASET_FILE.resolve():
        raise ValueError("refusing to write evaluation results over the golden dataset")
    if path.suffix.lower() != ".json":
        raise ValueError("evaluation output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_results(path: Path | None = None) -> dict | None:
    path = Path(path or ds.RESULTS_FILE)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise ValueError("invalid results document")
        return document
    except (ValueError, OSError):
        logger.warning("Stored evaluation results could not be read: %s", path)
        return None


def save_checkpoint(document: dict, path: Path) -> None:
    """Checkpoint plus a human-readable local response log; no data is published."""
    save(document, path)
    directory = ds.PROJECT_ROOT / ".cache" / "evaluation-responses"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{document['run_id']}.jsonl"
    entries = [{"run_id": document["run_id"], "fingerprint": document["fingerprint"],
                "report": str(path), "started_at": document["started_at"]}]
    entries.extend({"eval_id": row["eval_id"], **attempt} for row in document["results"]
                   for attempt in row.get("attempts", []))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
def _execute_case(case, catalog, history, retry_wait_s, max_retries,
                  run_config: dict, prior_attempts: list[dict] | None = None,
                  started_at: str | None = None, attempt_cap: int | None = None):
    attempts = list(prior_attempts or [])
    limit = attempt_cap if attempt_cap is not None else attempt_limit(case.eval_id, run_config.get("case_attempt_limits"))
    if len(attempts) >= limit:
        raise ds.DatasetError(f"{case.eval_id}: maximum {limit} attempts reached")
    started_at = started_at or _now()
    with index_retrieval.pin_revision(run_config["index_snapshot"]):
        current = evaluator.run_case(case, catalog, history).to_dict()
    attempts.append({"attempt": len(attempts) + 1, "started_at": started_at,
                     "finished_at": _now(), "fingerprint": run_config["fingerprint"],
                     "configuration": dict(run_config), "result": copy.deepcopy(current)})
    if attempt_cap is not None:
        attempts[-1]["targeted_attempt_cap"] = attempt_cap
    current["attempts"] = attempts
    return current


def run(dataset=None, catalog=None, limit=None, only=None, delay_s=None,
        retry_wait_s=None, max_retries=None, resume=False, checkpoint=True,
        verbose=True, output: Path | None = None, retry_failed: bool = False,
        initialize_only: bool = False, targeted_retry: bool = False,
        retry_quality: bool = True) -> dict:
    if targeted_retry and not resume:
        raise ds.DatasetError("Targeted retries require an existing report and resume=True")
    dataset = ds.load() if dataset is None else dataset
    catalog = catalog or ds.PermissionCatalog()
    delay_s = CASE_DELAY_S if delay_s is None else delay_s
    retry_wait_s = RETRY_WAIT_S if retry_wait_s is None else retry_wait_s
    max_retries = MAX_RETRIES if max_retries is None else max_retries
    if (any(not math.isfinite(v) or v < 0 for v in (delay_s, retry_wait_s))
            or max_retries < 0):
        raise ds.DatasetError("Delays and retry counts must be finite and non-negative")
    if max_retries != 0:
        raise ds.DatasetError(f"Immediate automatic retries are disabled. Restart to retry failed cases, up to {MAX_CASE_ATTEMPTS} total attempts.")
    problems = ds.validate(dataset, catalog)
    if problems:
        raise ds.DatasetError("\n".join(problems))
    cases = select_cases(dataset, limit, only)
    selected_ids = [case.eval_id for case in cases]
    output = Path(output or ds.RESULTS_FILE)
    previous = load_results(output) if resume else None
    limits = attempt_limits(previous)
    run_configuration = {
        "retry_quality": retry_quality,
        "scoring_policy": evaluator.SCORING_POLICY,
        "request_timeout_s": config.API_REQUEST_TIMEOUT_S,
        "verification_timeout_s": config.VERIFICATION_TIMEOUT_S,
        "verification_max_tokens": config.VERIFICATION_MAX_TOKENS,
        "tokens_per_minute": config.API_TOKENS_PER_MINUTE,
        "delay_s": delay_s,
        "retry_wait_s": retry_wait_s,
        "max_retries": max_retries,
        "max_case_attempts": MAX_CASE_ATTEMPTS,
        "jina_request_policy": JINA_REQUEST_POLICY,
        "groq_request_policy": GROQ_REQUEST_POLICY,
        "case_attempt_limits": limits,
        "case_top_k": evaluator.CASE_TOP_K,
        "model": config.GROQ_MODEL,
        "temperature": config.GROQ_TEMPERATURE,
        "max_tokens": config.GROQ_MAX_TOKENS,
        "reasoning_effort": config.GROQ_REASONING_EFFORT,
        "rerank_model": config.JINA_RERANK_MODEL,
        "rerank_enabled": config.RERANK_ENABLED,
        "embedding_model": index_config.JINA_EMBEDDING_MODEL,
        "collection": index_config.QDRANT_COLLECTION_NAME,
        "embedding_endpoint": index_config.JINA_API_URL,
        "rerank_endpoint": config.JINA_RERANK_URL,
        "generation_endpoint": config.GROQ_API_URL,
        "qdrant_endpoint": index_config.QDRANT_URL,
        "answering_config": asdict(config.DEFAULT_ANSWERING),
        "retrieval_config": asdict(index_config.DEFAULT_RETRIEVAL),
        "judge_tokens": evaluator.judge.JUDGE_MAX_TOKENS,
        "thresholds": {"recall": evaluator.MIN_RECALL, "groundedness": evaluator.MIN_GROUNDEDNESS,
                       "correctness": evaluator.MIN_CORRECTNESS, "relevancy": evaluator.MIN_RELEVANCY},
        "index_snapshot": resolve_index_snapshot(),
    }
    manifest = input_manifest(dataset, catalog, run_configuration)
    signature = fingerprint(dataset, catalog, run_configuration)
    run_configuration["fingerprint"] = signature
    output = Path(output or ds.RESULTS_FILE)
    if checkpoint and (output.suffix.lower() != ".json"
                       or output.resolve() in {ds.GOLDEN_DATASET_FILE.resolve(), ds.V11_DATASET_FILE.resolve()}):
        raise ds.DatasetError("Choose a .json results path separate from the golden dataset")
    rows: dict[str, dict] = {}
    previous_rows: dict[str, dict] = {}
    attempt_history: dict[str, list[dict]] = {}
    previous_duration = 0
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")
    started_at = _now()
    scheduler_migrations = []
    policy_migrations = []
    if resume:
        fingerprint_mismatch = (previous is not None
                                and previous.get("fingerprint") != signature)
        if (previous is None or previous.get("schema_version") != SCHEMA_VERSION
                or fingerprint_mismatch or previous.get("selected_ids") != selected_ids):
            detail = ", ".join(changed_components(previous or {}, manifest))
            raise ds.DatasetError("Cannot resume: mismatched checkpoint. Changed components: "
                                  + detail + ". Use a new --output path.")
        rows = {row["eval_id"]: row for row in previous["results"]}
        if len(rows) != len(previous["results"]) or set(rows) - set(selected_ids):
            raise ds.DatasetError("Checkpoint contains duplicate or unexpected cases")
        previous_rows = {key: dict(value) for key, value in rows.items()}
        attempt_history = {key: list(row.get("attempts", []))
                           for key, row in rows.items()}
        run_id, started_at = previous["run_id"], previous["started_at"]
        previous_duration = previous.get("duration_s", 0)
        scheduler_migrations = list(previous.get("scheduler_migrations", [])) + scheduler_migrations
        policy_migrations = list(previous.get("policy_migrations", [])) + policy_migrations
    elif checkpoint and output.exists():
        raise ds.DatasetError("Output already exists. Archive it, choose a new --output "
                              "path, or use --resume for a matching run.")

    retry_batch = targeted_retry_plan(previous) if targeted_retry else (previous or {}).get("targeted_retry")
    if targeted_retry:
        limits = {**limits, **retry_batch["attempt_caps"]}

    dependents: dict[str, list[str]] = {}
    case_map = {c.eval_id: c for c in cases}
    for case in cases:
        if case.conversation_from:
            dependents.setdefault(case.conversation_from, []).append(case.eval_id)

    def invalidate(case_id: str) -> None:
        row = rows.get(case_id)
        if row is not None:
            attempt_history[case_id] = list(row.get("attempts", []))
            case = case_map[case_id]
            blocked = evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question,
                case.category, case.expected_behavior, conversation_from=case.conversation_from,
                tags=list(case.tags), dependency_blocked=True,
                error="Prerequisite answer changed or is unavailable; this case needs a new attempt.")
            blocked.attempts = attempt_history[case_id]
            rows[case_id] = blocked.to_dict()
        for child_id in dependents.get(case_id, []):
            invalidate(child_id)

    for case in cases:
        parent = rows.get(case.conversation_from)
        if case.conversation_from and (
                not parent or parent.get("outcome") != "answer" or not parent.get("answered") or not parent.get("answer")):
            invalidate(case.eval_id)
    retry_ids = {c.eval_id for c in cases if retry_candidate(rows.get(c.eval_id, {}))
                 and attempt_count(rows[c.eval_id]) < attempt_limit(c.eval_id, limits)
                 and (retry_quality or rows[c.eval_id].get("status") == evaluator.ERROR)}
    queue = execution_order(cases, rows, limits)
    if targeted_retry:
        retry_ids &= retry_batch["attempt_caps"].keys()
        queue = [c for c in queue if c.eval_id in retry_batch["attempt_caps"]]
    started = time.monotonic()
    stop_reason = None
    stop_detail = None
    progress = {"phase": "starting", "case_id": None, "updated_at": _now()}

    def document():
        results = [rows[key] for key in selected_ids if key in rows]
        summary = metrics.aggregate(results)
        complete = (len(results) == len(cases) and summary["errors"] == 0
                    and not any(r.get("error") for r in results))
        return {
            "schema_version": SCHEMA_VERSION, "fingerprint": signature,
            "input_manifest": manifest, "scoring_policy": evaluator.SCORING_POLICY,
            "status_semantics": "answer_security", "run_id": run_id, "started_at": started_at, "generated_at": _now(),
            "partial": not complete or len(cases) != len(dataset),
            "selection_complete": complete, "selected_ids": selected_ids,
            "case_specs": [{"eval_id": c.eval_id, "category": c.category,
                            "expected_behavior": c.expected_behavior, "tags": list(c.tags)} for c in cases],
            "stop_reason": stop_reason, "dataset": dataset.summary(),
            "stop_detail": stop_detail,
            "response_log": f".cache/evaluation-responses/{run_id}.jsonl",
            "scheduler_migrations": scheduler_migrations,
            "policy_migrations": policy_migrations,
            "targeted_retry": retry_batch,
            "execution_order": [c.eval_id for c in queue],
            "retry_queue": [c.eval_id for c in execution_order(cases, rows, limits)
                            if retry_candidate(rows.get(c.eval_id, {}))
                            and (retry_quality or rows[c.eval_id].get("status") == evaluator.ERROR)
                            and (not targeted_retry or c.eval_id in retry_batch["attempt_caps"])
                            and 0 < attempt_count(rows[c.eval_id]) < attempt_limit(c.eval_id, limits)],
            "exhausted_cases": [c.eval_id for c in cases if retry_candidate(rows.get(c.eval_id, {}))
                                and attempt_count(rows[c.eval_id]) >= attempt_limit(c.eval_id, limits)],
            "first_attempt_summary": metrics.aggregate([
                (r.get("attempts") or [{}])[0].get("result", r) for r in results]),
            "score_view": "latest_attempt; original scores retained in first_attempt_summary and attempts",
            "progress": dict(progress),
            "configuration": {key: value for key, value in run_configuration.items()
                              if key != "fingerprint"},
            "model": config.GROQ_MODEL, "case_delay_s": delay_s,
            "duration_s": round(previous_duration + time.monotonic() - started, 1),
            "summary": summary, "matrix": [_matrix_row(row) for row in results],
            "measurements": metrics.measurements(results, cases),
            "cohorts": {"original_60": metrics.measurements(
                [r for r in results if r["eval_id"] in {f"E{i:03}" for i in range(1, 61)}],
                [c for c in cases if c.eval_id in {f"E{i:03}" for i in range(1, 61)}]),
                "expanded": metrics.measurements(results, cases)},
            "results": results,
        }

    if initialize_only:
        progress.update(phase="starting")
        initial = document()
        if checkpoint:
            save_checkpoint(initial, output)
        return initial

    executed = 0
    try:
        # Publish an empty checkpoint so a new run appears before its first call.
        if checkpoint:
            save_checkpoint(document(), output)
        for case in queue:
            current_row = rows.get(case.eval_id, {})
            if attempt_count(current_row) >= attempt_limit(case.eval_id, limits):
                if verbose and retry_candidate(current_row):
                    print_case_result(case, current_row, limits, retry_quality)
                continue
            if case_finished(current_row) and case.eval_id not in retry_ids:
                continue
            history = None
            if case.conversation_from:
                parent = rows.get(case.conversation_from)
                try:
                    history = conversation_history(case, dataset, rows)
                except ds.DatasetError as exc:
                    blocked = evaluator.CaseResult(
                        case.eval_id, case.user_id, case.user_name, case.question,
                        case.category, case.expected_behavior,
                        expected_sources=list(case.expected_sources),
                        expected_answer=case.expected_answer,
                        expected_facts=list(case.expected_facts),
                        conversation_from=case.conversation_from,
                        tags=list(case.tags), dependency_blocked=True,
                        error=str(exc))
                    blocked.attempts = attempt_history.get(case.eval_id, [])
                    rows[case.eval_id] = blocked.to_dict()
                    if verbose:
                        print_case_result(case, rows[case.eval_id], limits, retry_quality)
                    if checkpoint:
                        save_checkpoint(document(), output)
                    continue
            # Pace the first resumed request too: a quota wall may be recent.
            if delay_s and (executed or resume):
                progress.update(phase="waiting", case_id=case.eval_id, updated_at=_now())
                if checkpoint:
                    save_checkpoint(document(), output)
                time.sleep(delay_s)
            progress.update(phase="evaluating", case_id=case.eval_id, updated_at=_now())
            try:
                verify_index_snapshot(run_configuration["index_snapshot"])
                controls = {k: v for k, v in run_configuration.items() if k != "fingerprint"}
                if fingerprint(dataset, catalog, controls) != signature:
                    stop_reason = "evaluation_inputs_changed"
                    break
            except ds.DatasetError:
                stop_reason = "index_revision_changed"
                break
            if checkpoint:
                save_checkpoint(document(), output)
            old_answer = (previous_rows.get(case.eval_id) or {}).get("answer")
            prior_attempts = list(rows.get(case.eval_id, {}).get("attempts", []))
            attempt_started = _now()
            # Reserve before provider work: a killed terminal cannot evade the
            # attempt limit or erase earlier attempts.
            reserved = evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question,
                case.category, case.expected_behavior, tags=list(case.tags),
                conversation_from=case.conversation_from, interrupted=True,
                error="Attempt started but did not finish; restart to retry it at the end.").to_dict()
            reserved["attempts"] = prior_attempts + [{"attempt": len(prior_attempts) + 1,
                "started_at": attempt_started, "finished_at": None,
                "fingerprint": signature, "configuration": dict(run_configuration),
                "result": copy.deepcopy(reserved)}]
            if targeted_retry:
                reserved["attempts"][-1]["targeted_attempt_cap"] = limits[case.eval_id]
            rows[case.eval_id] = reserved
            if checkpoint:
                save_checkpoint(document(), output)
            rows[case.eval_id] = _execute_case(
                case, catalog, history, retry_wait_s, max_retries,
                run_configuration, prior_attempts, started_at=attempt_started,
                attempt_cap=limits[case.eval_id] if targeted_retry else None)
            if old_answer is not None and (rows[case.eval_id].get("answer") != old_answer
                    or not rows[case.eval_id].get("answered")
                    or rows[case.eval_id].get("outcome") != "answer"):
                for child_id in dependents.get(case.eval_id, []):
                    invalidate(child_id)
            try:
                verify_index_snapshot(run_configuration["index_snapshot"])
            except ds.DatasetError as exc:
                rows[case.eval_id]["status"] = evaluator.ERROR
                rows[case.eval_id]["error"] = str(exc)
                for attempt in rows[case.eval_id].get("attempts", [])[-1:]:
                    attempt["result"]["status"] = evaluator.ERROR
                    attempt["result"]["error"] = str(exc)
                stop_reason = "index_revision_changed"
            executed += 1
            row = rows[case.eval_id]
            if verbose:
                print_case_result(case, row, limits, retry_quality)
            if not stop_reason and row.get("interrupted"):
                stop_reason = "interrupted"
            elif not stop_reason and _is_quota_error(row.get("error")):
                stop_reason = "quota_exhausted"
            elif not stop_reason and row.get("error"):
                stop_reason = "execution_error"
            if checkpoint:
                save_checkpoint(document(), output)
            if stop_reason:
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as exc:
        stop_reason = "execution_error"
        stop_detail = f"{type(exc).__name__}: {exc}"
        active = rows.get(progress.get("case_id"))
        if active and active.get("attempts") and active["attempts"][-1].get("finished_at") is None:
            active.update(status=evaluator.ERROR, error=stop_detail, interrupted=False)
            active["attempts"][-1].update(finished_at=_now(), result=copy.deepcopy(
                {key: value for key, value in active.items() if key != "attempts"}))
            if verbose:
                print_case_result(case_map[active["eval_id"]], active, limits, retry_quality)
        logger.exception("Evaluation paused; previous case results are preserved")
    progress.update(phase="stopped" if stop_reason else "finished", case_id=None,
                    updated_at=_now())
    final = document()
    if checkpoint:
        save_checkpoint(final, output)
    if verbose and final["retry_queue"]:
        print("Next rerun, after fresh cases: " + ", ".join(final["retry_queue"]), flush=True)
    if verbose and final["exhausted_cases"]:
        print("Case attempt limit reached: " + ", ".join(final["exhausted_cases"]), flush=True)
    if verbose and stop_reason:
        print(f"Paused: {stop_reason}" + (f". Why: {stop_detail}" if stop_detail else ""), flush=True)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="comma-separated ids; prerequisite turns are included")
    parser.add_argument("--delay", type=float, default=CASE_DELAY_S)
    parser.add_argument("--retry-wait", type=float, default=RETRY_WAIT_S)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help=f"compatibility flag; resume already queues failures once, at the end, up to {MAX_CASE_ATTEMPTS} attempts")
    parser.add_argument("--initialize-only", action="store_true", help="save a frozen empty checkpoint without generating or judging")
    parser.add_argument("--output", type=Path, default=ds.RESULTS_FILE)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--plan", action="store_true", help="preview without API calls")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    try:
        dataset, catalog = ds.load(), ds.PermissionCatalog()
        problems = ds.validate(dataset, catalog)
        if problems:
            raise ds.DatasetError("\n".join(problems))
        cases = select_cases(dataset, args.limit, args.only.split(",") if args.only is not None else None)
        if (not math.isfinite(args.delay) or not math.isfinite(args.retry_wait)
                or min(args.delay, args.retry_wait, args.max_retries) < 0):
            raise ds.DatasetError("Delays and retry counts must be finite and non-negative")
        if args.validate_only or args.plan:
            print(json.dumps({"dataset": dataset.summary(),
                              "selected_ids": [c.eval_id for c in cases],
                              "selected_cases": len(cases), "network_calls": 0,
                              "minimum_pacing_minutes": round(max(0, len(cases) - 1) * args.delay / 60, 1)}, indent=2))
            print("Offline validation complete. No queries or model calls were made.")
            return 0
        config.validate_env()
        result = run(dataset, catalog, args.limit,
                     args.only.split(",") if args.only is not None else None,
                     args.delay, args.retry_wait, args.max_retries, args.resume,
                     not args.no_save, output=args.output,
                     retry_failed=args.retry_failed, initialize_only=args.initialize_only)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": result["run_id"], "partial": result["partial"],
                      "stop_reason": result["stop_reason"], "summary": result["summary"]}, indent=2))
    if not args.no_save:
        print(f"Results: {args.output.resolve()}")
    if result["stop_reason"] == "interrupted":
        return 130
    if result["summary"]["errors"] or not result["selection_complete"]:
        return 2
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
