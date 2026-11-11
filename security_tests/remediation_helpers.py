"""Explicit provider doubles for legacy pipeline/security contract tests.

These are not semantic quality tests. Real planner/verifier schemas and failure
paths are exercised separately by remediation_v12.
"""
import json
import re
from answering.src import answer_scope


def plan_response(messages, intent="other", **kwargs):
    data = json.loads(messages[1]["content"])
    question = data["question"]
    prior = data.get("previous_user_questions", [])
    personal = intent == "portfolio" or bool(answer_scope._OWN.search(answer_scope._normalized(question)))
    org = any(word in question.lower() for word in ("organization", "policy", "entire company", "整个公司", "organización"))
    named = answer_scope.mentioned_sellers(question)
    indices = [len(prior) - 1] if prior and re.search(r"which|those|that|them|what control|ellos", question, re.I) else []
    raw = {"scope": "organization" if org else "seller" if personal or named else "general",
           "seller_mode": "personal" if personal else "named" if named else "unspecified",
           "source_types": [], "time_intent": "unspecified", "history_indices": indices,
           "reason": "Explicit offline provider fixture"}
    raw.update(subject="policy" if org else "profile" if personal else "other",
               source_coverage="history" if 'history' in question.lower() else "focused")
    return json.dumps(raw), {"finish_reason": "stop", "total_tokens": 0}


def review_passthrough(question, context, sources, draft, plan, trace):
    trace.update(status="verified", usage={"total_tokens": 0}, evidence=[], corrections=[], latency_ms=0)
    return draft, {"total_tokens": 0}
