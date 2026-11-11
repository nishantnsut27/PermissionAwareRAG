"""JSON + SSE HTTP API over the Phase 5 service.

Deliberately stdlib-only: the pipeline lives in `service.py` and is framework
agnostic, so swapping in FastAPI later is a transport change, not a rewrite.

The backend validates a sample password at `/demo/login`, then binds requests
to a short-lived server session.

`POST /query`        buffered JSON answer
`POST /query/stream` Server-Sent Events: meta -> delta* -> done
`GET  /evaluation`        stored evaluation summary
`GET  /evaluation/matrix` stored evaluation matrix and per-case results
`GET /demo/users`     demo persona directory
`GET /ping`           plain-text liveness response for uptime checks
`POST /demo/login`    demo password validation and session issuance
`/demo/query/*`       session-bound demo requests

The evaluation endpoints are read-only. A run is a paced, quota-heavy batch
job executed from the command line (`python -m evaluation.runner`); serving it
from an HTTP request would let a page load burn the API budget.
"""
from __future__ import annotations

import json
import hmac
import ipaddress
import logging
import secrets
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import config, identity, service
from .deadline import (DeadlineExceeded, SocketDeadline, check_deadline,
                       remaining_timeout, request_deadline)

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024
_budget_lock = threading.Lock()
_principal_budgets: dict[str, deque[list[float | int]]] = {}
MAX_BUDGET_PRINCIPALS = 10000
DEMO_SESSION_TTL_S = 60 * 60
MAX_DEMO_SESSIONS = 10000
_demo_session_lock = threading.Lock()
_demo_sessions: dict[str, tuple[str, float]] = {}
_demo_login_budget_lock = threading.Lock()
_demo_login_budgets: dict[str, deque[float]] = {}


def _demo_password(user) -> str:
    first_name = str(user.name or "").strip().split()
    return f"{first_name[0].casefold()}@login" if first_name else ""


def _charge_demo_login(client_ip: str) -> bool:
    """Limit login attempts per peer without trusting forwarded headers."""
    now = time.monotonic()
    with _demo_login_budget_lock:
        budget = _demo_login_budgets.get(client_ip)
        if budget is None:
            if len(_demo_login_budgets) >= MAX_BUDGET_PRINCIPALS:
                for peer, entries in list(_demo_login_budgets.items()):
                    while entries and entries[0] <= now - 60:
                        entries.popleft()
                    if not entries:
                        del _demo_login_budgets[peer]
                if len(_demo_login_budgets) >= MAX_BUDGET_PRINCIPALS:
                    return False
            budget = _demo_login_budgets.setdefault(client_ip, deque())
        while budget and budget[0] <= now - 60:
            budget.popleft()
        if len(budget) >= config.API_REQUESTS_PER_MINUTE:
            return False
        budget.append(now)
        return True


def _issue_demo_session(user_id: str) -> tuple[str, int] | None:
    now = time.time()
    with _demo_session_lock:
        expired = [token for token, (_, expires) in _demo_sessions.items()
                   if expires <= now]
        for token in expired:
            del _demo_sessions[token]
        if len(_demo_sessions) >= MAX_DEMO_SESSIONS:
            return None
        token = secrets.token_urlsafe(32)
        _demo_sessions[token] = (user_id, now + DEMO_SESSION_TTL_S)
    return token, DEMO_SESSION_TTL_S


def _resolve_demo_session(token: str) -> str | None:
    now = time.time()
    with _demo_session_lock:
        session = _demo_sessions.get(token)
        if session is None:
            return None
        user_id, expires = session
        if expires <= now:
            del _demo_sessions[token]
            return None
        return user_id


def _reserved_tokens(payload: dict) -> int:
    query = payload.get("query") if isinstance(payload.get("query"), str) else ""
    conversation = payload.get("conversation")
    history_chars = (sum(len(row.get("content", "")) for row in conversation
                         if isinstance(row, dict)
                         and isinstance(row.get("content"), str))
                     if isinstance(conversation, list) else 0)
    input_chars = min(len(query), config.DEFAULT_ANSWERING.max_query_chars)
    input_chars += min(history_chars, config.DEFAULT_ANSWERING.max_history_chars)
    # Reserve all three calls, including the complete context and draft read
    # by verification. Explicit operator allowances remain authoritative.
    answer_prompt = (config.DEFAULT_ANSWERING.max_context_tokens
                     + (len(service.prompts.SYSTEM_PROMPT) + input_chars + 2048 + 2) // 3)
    intent_prompt = (len(service.query_plan.PLAN_SYSTEM) + input_chars + 512 + 2) // 3
    verification_prompt = (config.DEFAULT_ANSWERING.max_context_tokens
                           + config.GROQ_MAX_TOKENS
                           + service.verification.SCHEMA_RESERVATION_TOKENS
                           + (len(service.verification.SYSTEM_PROMPT) + input_chars + 2048 + 2) // 3)
    return (answer_prompt + intent_prompt + verification_prompt + config.GROQ_MAX_TOKENS
            + service.query_plan.PLAN_MAX_TOKENS + config.VERIFICATION_MAX_TOKENS)


def _charge_budget(principal, payload: dict):
    # The subject is resolved from a server-issued demo session; the JSON
    # user_id is deliberately not consulted here.
    key = principal.subject
    tokens = _reserved_tokens(payload)
    now = time.monotonic()
    with _budget_lock:
        if key not in _principal_budgets and len(_principal_budgets) >= MAX_BUDGET_PRINCIPALS:
            oldest = next(iter(_principal_budgets), None)
            if oldest is not None:
                del _principal_budgets[oldest]
        budget = _principal_budgets.setdefault(key, deque())
        while budget and budget[0][0] <= now - 60:
            budget.popleft()
        if (len(budget) >= config.API_REQUESTS_PER_MINUTE
                or sum(item[1] for item in budget) + tokens > config.API_TOKENS_PER_MINUTE):
            return False
        reservation = [now, tokens]
        budget.append(reservation)
        return reservation


def _reconcile_budget(principal, reservation, usage: dict | None) -> None:
    if not reservation or not isinstance(usage, dict) or not usage.get("reported"):
        return
    actual = usage.get("total_tokens")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        return
    key = principal.subject
    with _budget_lock:
        if any(item is reservation for item in _principal_budgets.get(key, ())):
            reservation[1] = actual


def _validate_payload(payload: dict, demo: bool) -> bool:
    allowed = {"query", "conversation", "seller_filter", "top_k"}
    if not demo:
        allowed.add("user_id")
    if set(payload) - allowed:
        return False
    if demo and "user_id" in payload and not isinstance(payload["user_id"], str):
        return False
    return True


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound concurrent handlers and return an explicit overload response."""
    daemon_threads = True

    def __init__(self, address, handler):
        self._admission = threading.BoundedSemaphore(config.API_MAX_CONCURRENT_REQUESTS)
        self._rejections = threading.BoundedSemaphore(2)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._admission.acquire(blocking=False):
            if not self._rejections.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                threading.Thread(target=self.reject_request,
                                 args=(request,), daemon=True).start()
            except Exception:
                self._rejections.release()
                self.shutdown_request(request)
                raise
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._admission.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._admission.release()

    def reject_request(self, request):
        """Read only bounded headers to attach CORS to overload responses.

        Dedicated slots keep slow peers out of both the accept loop and the
        main worker pool. No handler, authentication, body or provider runs.
        """
        deadline = time.monotonic() + 1.0
        try:
            header = bytearray()
            while b"\r\n\r\n" not in header and len(header) < 64 * 1024:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                request.settimeout(remaining)
                # Peek first so bytes after the header are never consumed.
                pending = request.recv(min(4096, 64 * 1024 - len(header)), socket.MSG_PEEK)
                if not pending:
                    return
                end = (header + pending).find(b"\r\n\r\n")
                size = len(pending) if end < 0 else end + 4 - len(header)
                header.extend(request.recv(size))
            if not header.endswith(b"\r\n\r\n"):
                return
            origins = [line.split(b":", 1)[1].strip().decode("latin-1")
                       for line in header.split(b"\r\n")[1:]
                       if line.lower().startswith(b"origin:")]
            origin = origins[0] if len(origins) == 1 else None
            body = b'{"error":"overloaded","message":"The service is busy. Try again shortly."}'
            headers = ["HTTP/1.1 503 Service Unavailable", "Connection: close",
                       "Content-Type: application/json; charset=utf-8", "Cache-Control: no-store",
                       "Retry-After: 1", "Vary: Origin", f"Content-Length: {len(body)}"]
            if origin and origin in config.API_ALLOWED_ORIGINS:
                headers += [f"Access-Control-Allow-Origin: {origin}",
                            "Access-Control-Expose-Headers: Retry-After"]
            remaining = deadline - time.monotonic()
            if remaining > 0:
                request.settimeout(remaining)
                request.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("latin-1") + body)
        except OSError:
            pass
        finally:
            self.shutdown_request(request)
            self._rejections.release()


def _evaluation():
    """Import Phase 7 lazily so Phase 5 does not depend on it at startup."""
    from evaluation import dataset, reports
    return dataset, reports


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraKnowledge/6.0"
    protocol_version = "HTTP/1.1"

    def handle_one_request(self):
        # This starts before BaseHTTPRequestHandler reads even the request
        # line. Shutdown interrupts buffered reads and writes, including peers
        # that keep sending bytes below an inactivity timeout.
        with request_deadline(config.API_REQUEST_TIMEOUT_S):
            total_guard = SocketDeadline(config.API_REQUEST_TIMEOUT_S)
            total_guard.register(self.connection)
            self._header_guard = SocketDeadline(
                min(config.API_HEADER_TIMEOUT_S, config.API_REQUEST_TIMEOUT_S),
                read_only=True)
            self._header_guard.register(self.connection)
            self.connection.settimeout(remaining_timeout(config.API_HEADER_TIMEOUT_S))
            try:
                super().handle_one_request()
            except (OSError, DeadlineExceeded):
                self.close_connection = True
            finally:
                self._header_guard.cancel()
                total_guard.cancel()

    def parse_request(self):
        parsed = super().parse_request()
        self._header_guard.cancel()
        if self._header_guard.expired:
            self.close_connection = True
            return False
        check_deadline()
        self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))
        if parsed and self.headers.get("Origin") and self.headers["Origin"] not in config.API_ALLOWED_ORIGINS:
            self._send(403, {"error": "origin_not_allowed", "message": "This website cannot access the demo API."})
            return False
        return parsed

    def end_headers(self):
        origin = self.headers.get("Origin") if hasattr(self, "headers") else None
        if origin and origin in config.API_ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _open_event_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # Close-delimited: the body length is unknown up front, and disabling
        # proxy buffering is what makes the delivery genuinely incremental.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _ping(self, *, head: bool = False) -> None:
        """A two-byte liveness probe; never touches retrieval or providers."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", "2")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        if not head:
            self.wfile.write(b"ok")
        self.close_connection = True

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path.rstrip("/") == "/ping":
            self._ping(head=True)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

    def _emit(self, event: dict) -> None:
        frame = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin")
        method = self.headers.get("Access-Control-Request-Method", "")
        headers = {value.strip().lower() for value in
                   self.headers.get("Access-Control-Request-Headers", "").split(",") if value.strip()}
        if (origin not in config.API_ALLOWED_ORIGINS or method not in {"GET", "POST", "HEAD"}
                or not headers <= {"authorization", "content-type", "accept"}):
            self._send(403, {"error": "preflight_not_allowed"})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        route = parsed.path.rstrip("/")
        report = parse_qs(parsed.query).get("report", [None])[0]
        if route == "/ping":
            self._ping()
        elif route == "/health":
            self._send(200, {"status": "ok"})
        elif route == "/live":
            self._send(200, {"status": "live"})
        elif route == "/ready":
            self._readiness()
        elif route == "/demo/users":
            if self._demo_request():
                self._send(200, {"users": service.list_users()})
        elif route == "/demo/me":
            principal = self._authenticate()
            if principal is not None:
                profile = next(row for row in service.list_users()
                               if row["user_id"] == principal.user.user_id)
                self._send(200, {"user": profile})
        elif route == "/users":
            principal = self._authenticate()
            if principal is not None:
                self._send(200, {"users": service.list_users()})
        elif route == "/evaluation":
            self._evaluation_summary(report)
        elif route == "/evaluation/matrix":
            self._evaluation_matrix(report)
        else:
            self._send(404, {"error": "not_found"})

    def _evaluation_summary(self, report: str | None = None) -> None:
        try:
            _, reports = _evaluation()
            payload = {**reports.public_payload(report),
                       "reports_enabled": config.EVALUATION_REPORTS_ENABLED}
        except ValueError as exc:
            self._send(404, {"error": "report_not_found", "message": str(exc)})
            return
        except Exception:
            logger.exception("evaluation summary failed")
            self._send(500, {"error": "internal_error",
                             "message": "Evaluation data is unavailable."})
            return
        self._send(200, payload)

    def _readiness(self) -> None:
        problems = []
        try:
            config.validate_settings()
        except RuntimeError as exc:
            problems.append(str(exc))
        try:
            config.validate_env()
        except RuntimeError as exc:
            problems.append(str(exc))
        if not config.API_DEMO_MODE:
            problems.append("demo login is disabled")
        try:
            from indexing.src import config as index_config
            from indexing.src.retrieval import active_revision
            collection, _ = active_revision()
            pointer = json.loads((index_config.REVISIONS_DIR / "active.json")
                                 .read_text(encoding="utf-8"))
            if pointer.get("collection_name") != collection:
                problems.append("local active pointer differs from the Qdrant alias")
        except Exception:
            problems.append("no valid local active index manifest")
        if problems:
            self._send(503, {"status": "not_ready", "problems": problems})
        else:
            self._send(200, {"status": "ready"})

    def _evaluation_matrix(self, report: str | None = None) -> None:
        if not config.EVALUATION_REPORTS_ENABLED:
            self._send(403, {"error": "reports_disabled", "message":
                            "Detailed reports are disabled. See guide.md for local operator access."})
            return
        principal = self._authenticate()
        if principal is None:
            return
        if not principal.is_operator:
            self._send(403, {"error": "forbidden",
                             "message": "Operator permission is required."})
            return
        try:
            _, reports = _evaluation()
            document = reports.load(report)
        except ValueError as exc:
            self._send(404, {"error": "report_not_found", "message": str(exc)})
            return
        except Exception:
            logger.exception("evaluation matrix failed")
            self._send(500, {"error": "internal_error",
                             "message": "Evaluation data is unavailable."})
            return
        if document is None:
            self._send(404, {"error": "no_results",
                             "message": "No evaluation has been run yet."})
            return
        self._send(200, {**reports.summary(document),
                         "report_id": report or "results.json",
                         "matrix": document.get("matrix", []),
                         "results": document.get("results", [])})

    def _read_json_body(self) -> dict | None:
        self.connection.settimeout(remaining_timeout(config.API_BODY_TIMEOUT_S))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "invalid_request",
                             "message": "Invalid Content-Length."})
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"error": "invalid_request",
                             "message": "Request body is missing or too large."})
            return None
        guard = SocketDeadline(remaining_timeout(config.API_BODY_TIMEOUT_S), read_only=True)
        guard.register(self.connection)
        try:
            raw = self.rfile.read(length)
            if guard.expired:
                raise socket.timeout("absolute body deadline exceeded")
            check_deadline()
            if len(raw) != length:
                raise ValueError("incomplete body")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            return payload
        except (socket.timeout, OSError):
            self._send(408, {"error": "request_timeout",
                             "message": "Request body read deadline exceeded."})
            return None
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "invalid_request",
                             "message": "Request body must be a JSON object."})
            return None
        finally:
            guard.cancel()
            self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))

    def _reject(self, exc: Exception) -> None:
        if isinstance(exc, DeadlineExceeded):
            self._send(504, {"outcome": "error", "completed": False,
                             "answered": False, "error": "request_timeout",
                             "message": "The request deadline was exceeded."})
        elif isinstance(exc, service.AnsweringError):
            logger.warning("request rejected: %s: %s", type(exc).__name__, exc)
            self._send(exc.status, {"outcome": "error", "completed": False,
                                    "answered": False,
                                    "error": type(exc).__name__,
                                    "message": exc.client_message})
        elif isinstance(exc, ValueError):
            logger.warning("bad request: %s", exc)
            self._send(400, {"outcome": "error", "completed": False,
                             "answered": False, "error": "invalid_request",
                             "message": "The request could not be processed."})
        else:
            logger.exception("unhandled error while answering")
            self._send(500, {"outcome": "error", "completed": False,
                             "answered": False, "error": "internal_error",
                             "message": "The request could not be completed."})

    def _authenticate(self):
        if not self._demo_request():
            self._discard_rejected_body()
            return None
        return self._demo_principal()

    def _discard_rejected_body(self) -> None:
        if self.command != "POST":
            return
        self.connection.settimeout(remaining_timeout(config.API_BODY_TIMEOUT_S))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= MAX_BODY_BYTES:
            guard = SocketDeadline(remaining_timeout(config.API_BODY_TIMEOUT_S), read_only=True)
            guard.register(self.connection)
            try:
                self.rfile.read(length)
            except (socket.timeout, OSError):
                return
            finally:
                guard.cancel()
                self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))

    def _demo_request(self) -> bool:
        if not config.API_DEMO_MODE:
            self._send(404, {"error": "not_found"})
            return False
        if config.API_PUBLIC_DEMO_MODE:
            return True
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                raise ValueError("non-loopback peer")
        except ValueError:
            self._send(403, {"error": "demo_local_only",
                             "message": "Demo routes are available locally only."})
            return False
        return True

    def _demo_principal(self):
        authorization = self.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        user_id = (_resolve_demo_session(token.strip())
                   if scheme.casefold() == "bearer" and token.strip() else None)
        if user_id is None:
            self._discard_rejected_body()
            self._send(401, {"error": "demo_login_required",
                             "message": "Select an account and sign in to the demo."})
            return None
        try:
            user = service.get_demo_user(user_id)
        except service.AnsweringError:
            self._discard_rejected_body()
            self._send(401, {"error": "demo_session_invalid",
                             "message": "The demo session is invalid. Sign in again."})
            return None
        subject = ("demo:public" if config.API_PUBLIC_DEMO_MODE
                   else f"demo:{user_id}")
        return identity.Principal(user, subject)

    def _handle_demo_login(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if set(payload) != {"user_id", "password"}:
            self._send(400, {"error": "invalid_request",
                             "message": "A demo account and password are required."})
            return
        try:
            user = service.get_demo_user(payload.get("user_id"))
        except service.AnsweringError:
            user = None
        supplied = payload.get("password")
        supplied = supplied.strip().casefold() if isinstance(supplied, str) else ""
        expected = _demo_password(user) if user else ""
        if (not expected or not hmac.compare_digest(
                supplied.encode("utf-8"), expected.encode("utf-8"))):
            self._send(401, {"error": "invalid_demo_credentials",
                             "message": "The account or demo password is incorrect."})
            return
        session = _issue_demo_session(user.user_id)
        if session is None:
            self._send(503, {"error": "demo_session_capacity",
                             "message": "The demo is busy. Please try again later."})
            return
        token, expires_in = session
        profile = next((row for row in service.list_users()
                        if row["user_id"] == user.user_id), None)
        self._send(200, {"access_token": token, "token_type": "Bearer",
                         "expires_in": expires_in, "user": profile})

    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        demo = route in {"/demo/query", "/demo/query/stream"}
        if route == "/demo/login":
            if not self._demo_request():
                return
            if not _charge_demo_login(str(self.client_address[0])):
                self._discard_rejected_body()
                self._send(429, {"error": "rate_limited", "message":
                                 "Too many demo login attempts. Please try again later."})
                return
            self._handle_demo_login()
        elif route in {"/query", "/demo/query"}:
            if demo and not self._demo_request():
                return
            principal = self._demo_principal() if demo else self._authenticate()
            if principal is not None:
                self._handle_query(principal, demo)
        elif route in {"/query/stream", "/demo/query/stream"}:
            if demo and not self._demo_request():
                return
            principal = self._demo_principal() if demo else self._authenticate()
            if principal is not None:
                self._handle_query_stream(principal, demo)
        elif route == "/evaluation/run":
            # The UI and API never start quota-consuming evaluation jobs.
            self.close_connection = True
            self._send(405, {"error": "method_not_allowed", "message":
                            "Run evaluations explicitly from the CLI; see guide.md."})
        else:
            self._send(404, {"error": "not_found"})

    def _handle_query(self, principal=None, demo: bool = False) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if not _validate_payload(payload, demo):
            self._send(400, {"error": "invalid_request",
                             "message": "Request fields do not match the supported schema."})
            return
        budget_principal = principal
        reservation = _charge_budget(budget_principal, payload)
        if not reservation:
            self._send(429, {"error": "rate_limited", "message":
                             "The per-principal request or token budget is exhausted."})
            return
        try:
            with request_deadline(config.API_REQUEST_TIMEOUT_S):
                response = service.answer_query(
                    principal=budget_principal,
                    query=payload.get("query"),
                    conversation_history=payload.get("conversation"),
                    seller_filter=payload.get("seller_filter"),
                    top_k=payload.get("top_k"),
                )
        except Exception as exc:
            _reconcile_budget(budget_principal, reservation,
                              getattr(exc, "security_trace", {}).get("model_usage"))
            self._reject(exc)
            return
        _reconcile_budget(budget_principal, reservation,
                          response.security_trace.get(
                              "model_usage", {"reported": True, "total_tokens": 0}))
        self._send(200, response.to_dict())

    def _handle_query_stream(self, principal=None, demo: bool = False) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if not _validate_payload(payload, demo):
            self._send(400, {"error": "invalid_request",
                             "message": "Request fields do not match the supported schema."})
            return
        service_principal = principal
        reservation = _charge_budget(service_principal, payload)
        if not reservation:
            self._send(429, {"error": "rate_limited", "message":
                             "The per-principal request or token budget is exhausted."})
            return
        deadline_context = request_deadline(config.API_REQUEST_TIMEOUT_S)
        deadline_context.__enter__()
        events = service.answer_query_stream(
            principal=service_principal,
            query=payload.get("query"),
            conversation_history=payload.get("conversation"),
            seller_filter=payload.get("seller_filter"),
            top_k=payload.get("top_k"),
        )
        # Pull the first event before committing to 200: validation errors are
        # raised on first advance and still deserve a proper status code.
        try:
            first = next(events)
        except StopIteration:
            deadline_context.__exit__(None, None, None)
            self._send(500, {"error": "internal_error",
                             "message": "The request could not be completed."})
            return
        except Exception as exc:
            deadline_context.__exit__(type(exc), exc, exc.__traceback__)
            self._reject(exc)
            return

        self._open_event_stream()
        try:
            if first.get("type") == "done":
                _reconcile_budget(service_principal, reservation,
                                  first.pop("_token_usage",
                                            {"reported": True, "total_tokens": 0}))
            self._emit(first)
            for event in events:
                if event.get("type") in {"done", "error"}:
                    _reconcile_budget(service_principal, reservation,
                                      event.pop("_token_usage",
                                                None))
                self._emit(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("client disconnected during stream")
            events.close()
        except Exception:
            logger.exception("error while streaming")
            try:
                self._emit({"type": "error", "outcome": "error",
                            "completed": False, "answered": False,
                            "error": "internal_error",
                            "message": "The request could not be completed."})
            except OSError:
                pass
        finally:
            deadline_context.__exit__(None, None, None)
            self.close_connection = True


def serve(host: str | None = None, port: int | None = None) -> None:
    config.validate_settings()
    host = host or config.API_HOST
    port = port or config.API_PORT
    if config.API_PUBLIC_DEMO_MODE and not config.API_DEMO_MODE:
        raise RuntimeError("Public demo mode requires API_DEMO_MODE=true.")
    # Public demo mode is an explicit opt-in for synthetic-data deployments.
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if config.API_DEMO_MODE and not config.API_PUBLIC_DEMO_MODE and not loopback:
        raise RuntimeError("Demo mode may bind only to a loopback address.")
    if not config.API_DEMO_MODE and not loopback:
        raise RuntimeError(
            "Demo login is disabled; API must bind to loopback.")
    httpd = BoundedHTTPServer((host, port), Handler)
    logger.info("Phase 5 API listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
