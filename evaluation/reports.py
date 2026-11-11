"""Read saved checkpoints for the dashboard without executing the pipeline."""
from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from pathlib import Path

from . import dataset as ds, metrics


class ReportNotFound(ValueError):
    pass


def resolve(report: str | None = None) -> Path:
    """Accept only the default report or one file immediately under runs/."""
    if report is None:
        available = list_reports()
        report = available[0]["report_id"] if available else "results.json"
    if report == "results.json":
        path = ds.RESULTS_FILE
        root = ds.PACKAGE_ROOT.resolve()
    elif re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9_.-]*\.json", report):
        root = (ds.PACKAGE_ROOT / "runs").resolve()
        path = root / report.removeprefix("runs/")
    else:
        raise ReportNotFound("Unknown evaluation report.")
    resolved = path.resolve()
    if resolved.parent != root or path.is_symlink():
        raise ReportNotFound("Unknown evaluation report.")
    return resolved


def load(report: str | None = None) -> dict | None:
    document = None
    try:
        candidate = json.loads(resolve(report).read_text(encoding="utf-8"),
                               parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        rows = candidate.get("results") if isinstance(candidate, dict) else None
        if (not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("eval_id"), str) for r in rows)
                or len({r["eval_id"] for r in rows}) != len(rows)):
            raise ValueError("Invalid case rows")
        # Validate the aggregate projection as well; malformed nested rows
        # must not hide a newer valid report by crashing the listing.
        public_summary(candidate)
        document = candidate
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    if document is None and report is not None:
        raise ReportNotFound("The selected report is missing or unreadable. Refresh the run list.")
    return document


def summary(document: dict) -> dict:
    result = {key: document.get(key) for key in (
        "run_id", "generated_at", "started_at", "duration_s", "model", "partial",
        "summary", "schema_version", "selection_complete", "stop_reason", "progress")}
    rows = document.get("results") or []
    selected_ids = document.get("selected_ids") or [r["eval_id"] for r in rows]
    recorded_ids = {r["eval_id"] for r in rows}
    result.update({
        "selected_cases": len(selected_ids),
        "pending_ids": [key for key in selected_ids if key not in recorded_ids],
        "report_dataset": document.get("dataset"),
    })
    return result


def list_reports() -> list[dict]:
    paths = [ds.RESULTS_FILE, *sorted((ds.PACKAGE_ROOT / "runs").glob("*.json"))]
    reports = []
    for path in paths:
        report_id = path.relative_to(ds.PACKAGE_ROOT).as_posix()
        try:
            document = load(report_id)
        except (ReportNotFound, OSError):
            continue
        reports.append({"report_id": report_id, **summary(document), **descriptor(document)})
    return sorted(reports, key=lambda row: (_timestamp(row.get("started_at") or row.get("generated_at")), row["report_id"]), reverse=True)


def _timestamp(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError, AttributeError):
        return 0


def descriptor(document):
    version = document.get("schema_version", 0)
    legacy = type(version) is not int or version < 5
    dataset = document.get("dataset") or {}
    selected = document.get("selected_ids") or [r["eval_id"] for r in document["results"]]
    return {"run_id": document.get("run_id"), "started_at": document.get("started_at"),
            "report_kind": "latest_by_case" if document.get("report_kind") == "latest_by_case" else "run",
            "source_run_count": len(document.get("snapshot_sources") or []) if document.get("report_kind") == "latest_by_case" else 1,
            "scoring_policy": ("answer_security_v6_retrieval_diagnostic" if type(version) is int and version >= 6
                               else "historical_combined"),
            "status_semantics": "answer_security" if type(version) is int and version >= 6 else "historical_combined",
            "generated_at": document.get("generated_at"), "dataset_version": dataset.get("version"),
            "selected_cases": len(selected), "total_cases": len(document["results"]),
            "status": "Legacy" if legacy else "Complete" if document.get("selection_complete") is True else "Incomplete"}


def public_summary(document):
    """Allowlist only aggregate fields. Never copy stored summary dictionaries."""
    rows = document["results"]
    lookup = {r["eval_id"]: r for r in rows}
    ids = document.get("selected_ids") or list(lookup)
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError("Invalid selection")
    specs = document.get("case_specs") or [lookup.get(i, {"eval_id": i}) for i in ids]
    if not isinstance(specs, list) or any(not isinstance(c, dict) for c in specs):
        raise ValueError("Invalid case descriptors")
    original = {f"E{i:03}" for i in range(1, 61)}
    progress = document.get("progress") or {}
    phase = progress.get("phase")
    stop = document.get("stop_reason")
    return {**descriptor(document), "schema_version": document.get("schema_version"),
            "selection_complete": document.get("selection_complete") is True,
            "partial": document.get("partial", True), "duration_s": document.get("duration_s"),
            "model": document.get("model"), "dataset_id": (document.get("dataset") or {}).get("dataset_id"),
            "stop_reason": stop if stop in {"quota_exhausted", "interrupted", "index_revision_changed", "evaluation_inputs_changed", "execution_error"} else None,
            "phase": phase if phase in {"starting", "waiting", "evaluating", "stopped", "finished"} else None,
            "measurements": metrics.measurements(rows, specs),
            "cohorts": {"original_60": metrics.measurements([r for r in rows if r["eval_id"] in original], [c for c in specs if c.get("eval_id") in original]),
                        "expanded": metrics.measurements(rows, specs)}}


def public_payload(report=None):
    available = list_reports()
    selected = report if report is not None else available[0]["report_id"] if available else None
    document = load(selected) if selected else None
    fields = ("report_id", "run_id", "started_at", "generated_at", "status", "dataset_version", "selected_cases", "total_cases", "scoring_policy", "status_semantics", "report_kind", "source_run_count")
    result = {"has_results": document is not None,
              "available_reports": [{k: r.get(k) for k in fields} for r in available]}
    if document is not None:
        result.update(report_id=selected, **public_summary(document))
    return result
