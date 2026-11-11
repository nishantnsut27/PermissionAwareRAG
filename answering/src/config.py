"""Phase 5 configuration: Groq answering, Jina reranking, retrieval budgets."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from . import groq_keys, jina_keys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


GROQ_KEY_COOLDOWN_S = _float("GROQ_KEY_COOLDOWN_S", 60.0)

GROQ_API_URL = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = _float("GROQ_TEMPERATURE", 0.0)
GROQ_MAX_TOKENS = _int("GROQ_MAX_TOKENS", 1200)
# gpt-oss models spend part of max_tokens on hidden reasoning; "low" keeps the
# visible answer within budget.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
GROQ_TIMEOUT_S = _int("GROQ_TIMEOUT_S", 60)
GROQ_MAX_RETRIES = _int("GROQ_MAX_RETRIES", 2)

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")
JINA_RERANK_MODEL = os.getenv(
    "JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
RERANK_TIMEOUT_S = _int("RERANK_TIMEOUT_S", 30)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() != "false"

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_ALLOWED_ORIGINS = tuple(value.strip() for value in
                            os.getenv("API_ALLOWED_ORIGINS", "").split(",") if value.strip())
API_PORT = _int("API_PORT", 8000)
API_DEMO_MODE = os.getenv("API_DEMO_MODE", "false").strip().lower() == "true"
# Public persona switching follows the already explicit demo switch unless an
# operator overrides it.
API_PUBLIC_DEMO_MODE = os.getenv(
    "API_PUBLIC_DEMO_MODE", "true" if API_DEMO_MODE else "false"
).strip().lower() == "true"
API_MAX_CONCURRENT_REQUESTS = _int("API_MAX_CONCURRENT_REQUESTS", 8)
API_BODY_TIMEOUT_S = _float("API_BODY_TIMEOUT_S", 5.0)
API_HEADER_TIMEOUT_S = _float("API_HEADER_TIMEOUT_S", 5.0)
API_REQUEST_TIMEOUT_S = _float("API_REQUEST_TIMEOUT_S", 90.0)
API_REQUESTS_PER_MINUTE = _int("API_REQUESTS_PER_MINUTE", 30)
API_TOKENS_PER_MINUTE = _int("API_TOKENS_PER_MINUTE", 60000)
VERIFICATION_TIMEOUT_S = 30
VERIFICATION_MAX_TOKENS = 2400

# This flag can turn detailed report routes off. It never grants access.
EVALUATION_REPORTS_ENABLED = os.getenv(
    "EVALUATION_REPORTS_ENABLED", "false").strip().lower() == "true"


@dataclass(frozen=True)
class AnsweringConfig:
    top_k: int = _int("ANSWER_TOP_K", 5)
    candidate_pool: int = _int("ANSWER_CANDIDATE_POOL", 40)
    max_context_chunks: int = 12
    multi_seller_retrieval: str = os.getenv("MULTI_SELLER_RETRIEVAL", "global").strip().lower()
    # Provisional threshold. No held-out quality calibration has been run.
    rerank_threshold: float = _float("RERANK_SCORE_THRESHOLD", 0.20)
    max_history_turns: int = _int("MAX_HISTORY_TURNS", 6)
    max_history_chars: int = _int("MAX_HISTORY_CHARS", 1500)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 12000)
    max_context_tokens: int = _int("MAX_CONTEXT_TOKENS", 12000)
    max_query_chars: int = _int("MAX_QUERY_CHARS", 1000)


DEFAULT_ANSWERING = AnsweringConfig()


def validate_settings() -> None:
    """Reject malformed or unsafe resource limits at process startup."""
    for origin in API_ALLOWED_ORIGINS:
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.path
                or parsed.query or parsed.fragment or "*" in origin):
            raise RuntimeError("API_ALLOWED_ORIGINS must contain exact HTTP(S) origins without paths")
    numeric = {
        "API_MAX_CONCURRENT_REQUESTS": (API_MAX_CONCURRENT_REQUESTS, 1, 256),
        "API_PORT": (API_PORT, 1, 65535),
        "API_REQUESTS_PER_MINUTE": (API_REQUESTS_PER_MINUTE, 1, 10000),
        "API_TOKENS_PER_MINUTE": (API_TOKENS_PER_MINUTE, 100, 10000000),
        "API_BODY_TIMEOUT_S": (API_BODY_TIMEOUT_S, 0.1, 120),
        "API_HEADER_TIMEOUT_S": (API_HEADER_TIMEOUT_S, 0.1, 120),
        "API_REQUEST_TIMEOUT_S": (API_REQUEST_TIMEOUT_S, 1, 900),
        "GROQ_KEY_COOLDOWN_S": (GROQ_KEY_COOLDOWN_S, 0, 86400),
        "ANSWER_TOP_K": (DEFAULT_ANSWERING.top_k, 1, 20),
        "ANSWER_CANDIDATE_POOL": (DEFAULT_ANSWERING.candidate_pool, 1, 40),
        "MAX_HISTORY_TURNS": (DEFAULT_ANSWERING.max_history_turns, 0, 100),
        "MAX_HISTORY_CHARS": (DEFAULT_ANSWERING.max_history_chars, 0, 100000),
        "MAX_CONTEXT_CHARS": (DEFAULT_ANSWERING.max_context_chars, 1, 1000000),
        "MAX_QUERY_CHARS": (DEFAULT_ANSWERING.max_query_chars, 1, 10000),
        "MAX_CONTEXT_TOKENS": (DEFAULT_ANSWERING.max_context_tokens, 1, 100000),
        "GROQ_TIMEOUT_S": (GROQ_TIMEOUT_S, 1, 900),
        "GROQ_MAX_RETRIES": (GROQ_MAX_RETRIES, 0, 10),
        "RERANK_TIMEOUT_S": (RERANK_TIMEOUT_S, 1, 300),
    }
    for name, (value, minimum, maximum) in numeric.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not minimum <= value <= maximum):
            raise RuntimeError(f"Invalid {name}: expected a value in [{minimum}, {maximum}]")
    for name in ("API_MAX_CONCURRENT_REQUESTS", "API_REQUESTS_PER_MINUTE",
                 "API_TOKENS_PER_MINUTE", "ANSWER_TOP_K", "ANSWER_CANDIDATE_POOL",
                 "API_PORT", "MAX_HISTORY_TURNS", "MAX_HISTORY_CHARS",
                 "MAX_QUERY_CHARS", "MAX_CONTEXT_TOKENS", "MAX_CONTEXT_CHARS",
                 "GROQ_TIMEOUT_S", "GROQ_MAX_RETRIES", "RERANK_TIMEOUT_S"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                int(raw)
            except ValueError as exc:
                raise RuntimeError(f"Invalid integer setting {name}") from exc
    for name in ("API_HEADER_TIMEOUT_S", "API_BODY_TIMEOUT_S", "API_REQUEST_TIMEOUT_S",
                 "GROQ_KEY_COOLDOWN_S", "RERANK_SCORE_THRESHOLD"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                float(raw)
            except ValueError as exc:
                raise RuntimeError(f"Invalid float setting {name}") from exc
    if not 0 <= DEFAULT_ANSWERING.rerank_threshold <= 1:
        raise RuntimeError("Invalid RERANK_SCORE_THRESHOLD: expected [0, 1]")
    if DEFAULT_ANSWERING.multi_seller_retrieval not in {"global", "seller_balanced"}:
        raise RuntimeError("Invalid MULTI_SELLER_RETRIEVAL: expected global or seller_balanced")

REFUSAL_ANSWER = (
    "I couldn't find sufficient authorized information to answer this request."
)

REQUIRED_ENV_VARS = ("QDRANT_URL", "QDRANT_API_KEY")


def validate_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if not jina_keys.load_keys():
        missing.append("JINA_API_KEY_1 (or JINA_API_KEY)")
    if not groq_keys.pool().size:
        missing.append("GROQ_API_KEY_1")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Check .env at the repository root.")
