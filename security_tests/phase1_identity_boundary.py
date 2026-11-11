from __future__ import annotations

import threading
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import requests

from answering.src import api, config, identity, service
from access_control.src.users import get_user


@contextmanager
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self):
        budgets = patch.dict(api._principal_budgets, {}, clear=True)
        budgets.start()
        self.addCleanup(budgets.stop)
        self.demo = patch.object(config, "API_DEMO_MODE", False)
        self.demo.start()

    def tearDown(self):
        self.demo.stop()

    def auth(self, token):
        if token == "Bearer normal":
            return identity.Principal(get_user("U-001"), "normal-sub")
        if token == "Bearer admin":
            return identity.Principal(get_user("U-005"), "admin-sub")
        raise RuntimeError("invalid test credential")

    def test_anonymous_query_and_directory_fail_closed(self):
        with api_server() as base:
            response = requests.post(base + "/query", json={"query": "hello"})
            users = requests.get(base + "/users")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "not_found")
        self.assertEqual(users.status_code, 404)

    def test_normal_principal_wins_over_body_admin_id(self):
        result = service.AnswerResponse("id", "U-001", "q", "a", True)
        with patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(service, "answer_query", return_value=result) as answer:
            with api_server() as base:
                response = requests.post(base + "/query", json={
                    "user_id": "U-005", "query": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["principal"].user.user_id, "U-001")

    def test_json_and_sse_use_same_verified_principal(self):
        result = service.AnswerResponse("id", "U-001", "q", "a", True)
        events = [{"type": "meta", "user_id": "U-001"},
                  {"type": "done", "user_id": "U-001"}]
        with patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(service, "answer_query", return_value=result) as answer, \
             patch.object(service, "answer_query_stream", return_value=iter(events)) as stream:
            with api_server() as base:
                json_response = requests.post(base + "/query", json={"user_id": "U-005", "query": "q"},
                                              headers={"Authorization": "Bearer normal"})
                sse_response = requests.post(base + "/query/stream", json={"user_id": "U-005", "query": "q"},
                                             headers={"Authorization": "Bearer normal", "Accept": "text/event-stream"})
        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(sse_response.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["principal"].user.user_id, "U-001")
        self.assertEqual(stream.call_args.kwargs["principal"].user.user_id, "U-001")
        self.assertIn('"user_id": "U-001"', sse_response.text)

    def test_invalid_credential_is_rejected_for_sse_and_users(self):
        with (patch.object(config, "API_DEMO_MODE", True),
              patch.object(config, "API_PUBLIC_DEMO_MODE", True)):
            with api_server() as base:
                query = requests.post(base + "/query/stream", json={"query": "q"},
                                      headers={"Authorization": "Bearer invalid"})
                users = requests.get(base + "/users", headers={"Authorization": "Bearer invalid"})
        self.assertEqual(query.status_code, 401)
        self.assertEqual(users.status_code, 401)

    def test_report_matrix_requires_operator_even_when_feature_enabled(self):
        document = {"results": [{"security_trace": {
            "raw_retrieval_candidates": [{"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {
            "load": staticmethod(lambda _: document),
            "summary": staticmethod(lambda _: {"run_id": "r"}),
        })
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("S004-SECRET", response.text)

    def test_operator_can_read_detail_only_when_feature_enabled(self):
        document = {"summary": {}, "matrix": [{"answer": "sensitive"}],
                    "results": [{"answer": "sensitive", "evidence": ["private"],
                                 "security_trace": {"raw_retrieval_candidates": [
                                     {"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {"load": staticmethod(lambda _: document),
                                       "summary": staticmethod(lambda _: {"run_id": "r"})})
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer admin")), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], document["results"])

    def test_report_flag_alone_never_authorizes(self):
        document = {"results": [{"security_trace": {
            "raw_retrieval_candidates": [{"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {
            "load": staticmethod(lambda _: document),
            "summary": staticmethod(lambda _: {"run_id": "r"}),
        })
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("S004-SECRET", response.text)

    def test_public_report_summary_contains_no_case_details(self):
        document = {"run_id": "run-1", "generated_at": "2026-10-31",
                    "summary": {"total_cases": 7},
                    "results": [{"answer": "private answer", "evidence": ["secret"]}]}
        reports = type("Reports", (), {
            "public_payload": staticmethod(lambda _: {"has_results": True, "run_id": "run-1", "total_cases": 7}),
            "load": staticmethod(lambda _: document),
            "list_reports": staticmethod(lambda: [
                {"report_id": "runs/run-1.json", "generated_at": "2026-10-31",
                 "results": document["results"]}]),
        })
        with patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("results", response.json())
        self.assertNotIn("answer", response.text)
        self.assertNotIn("evidence", response.text)

    def test_api_refuses_remote_bind_without_demo_login(self):
        with patch.object(config, "API_DEMO_MODE", False), \
             patch.object(config, "API_PUBLIC_DEMO_MODE", False):
            with self.assertRaises(RuntimeError):
                api.serve("0.0.0.0", 8000)

    def test_demo_mode_always_refuses_remote_bind(self):
        with patch.object(config, "API_DEMO_MODE", True), \
             patch.object(config, "API_PUBLIC_DEMO_MODE", False):
            with self.assertRaises(RuntimeError):
                api.serve("0.0.0.0", 8000)

    def test_remote_demo_requests_require_explicit_public_mode(self):
        handler = type("FakeHandler", (), {})()
        handler.client_address = ("203.0.113.8", 12345)
        handler._send = lambda status, payload: setattr(handler, "sent", (status, payload))
        with patch.object(config, "API_DEMO_MODE", True), \
             patch.object(config, "API_PUBLIC_DEMO_MODE", False):
            self.assertFalse(api.Handler._demo_request(handler))
            self.assertEqual(handler.sent[0], 403)
        with patch.object(config, "API_DEMO_MODE", True), \
             patch.object(config, "API_PUBLIC_DEMO_MODE", True):
            self.assertTrue(api.Handler._demo_request(handler))

    def test_public_demo_directory_is_served_by_backend(self):
        users = [{"user_id": "U-001", "name": "Sample Account", "role": "Account Manager"}]
        with (patch.object(config, "API_DEMO_MODE", True),
              patch.object(config, "API_PUBLIC_DEMO_MODE", True),
              patch.object(service, "list_users", return_value=users)):
            with api_server() as base:
                response = requests.get(base + "/demo/users")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["users"], users)

    def test_demo_login_issues_a_session_and_queries_use_its_server_identity(self):
        result = service.AnswerResponse("id", "U-001", "q", "answer", True)
        with (patch.object(config, "API_DEMO_MODE", True),
              patch.object(config, "API_PUBLIC_DEMO_MODE", True),
              patch.object(service, "answer_query", return_value=result) as answer):
            with api_server() as base:
                unicode_password = requests.post(base + "/demo/login", json={
                    "user_id": "U-001", "password": "áditya@login"})
                invalid = requests.post(base + "/demo/login", json={
                    "user_id": "U-001", "password": "wrong"})
                login = requests.post(base + "/demo/login", json={
                    "user_id": "U-001", "password": "aditya@login"})
                self.assertEqual(unicode_password.status_code, 401)
                self.assertEqual(invalid.status_code, 401)
                self.assertEqual(login.status_code, 200)
                token = login.json()["access_token"]
                switched = requests.post(base + "/demo/query", json={
                    "query": "q", "user_id": "U-005"},
                    headers={"Authorization": f"Bearer {token}"})
                response = requests.post(base + "/demo/query", json={"query": "q"},
                    headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(switched.status_code, 400)
        self.assertEqual(response.status_code, 200)
        principal = answer.call_args.kwargs["principal"]
        self.assertEqual(principal.user.user_id, "U-001")
        self.assertEqual(principal.subject, "demo:public")

    def test_demo_login_rate_limit_is_per_peer(self):
        api._demo_login_budgets.clear()
        with patch.object(config, "API_REQUESTS_PER_MINUTE", 1):
            self.assertTrue(api._charge_demo_login("203.0.113.8"))
            self.assertFalse(api._charge_demo_login("203.0.113.8"))
            self.assertTrue(api._charge_demo_login("203.0.113.9"))

    def test_full_demo_session_store_keeps_existing_sessions(self):
        api._demo_sessions.clear()
        api._demo_sessions["existing"] = ("U-001", api.time.time() + 60)
        try:
            with patch.object(api, "MAX_DEMO_SESSIONS", 1):
                self.assertIsNone(api._issue_demo_session("U-002"))
            self.assertEqual(api._demo_sessions["existing"][0], "U-001")
        finally:
            api._demo_sessions.clear()

    def test_public_demo_mode_can_bind_remotely_only_with_demo_mode(self):
        fake_server = type("FakeServer", (), {
            "serve_forever": lambda self: (_ for _ in ()).throw(KeyboardInterrupt()),
            "server_close": lambda self: None,
        })()
        with (patch.object(config, "validate_settings"),
              patch.object(config, "API_DEMO_MODE", True),
              patch.object(config, "API_PUBLIC_DEMO_MODE", True),
              patch.object(api, "BoundedHTTPServer", return_value=fake_server) as server):
            api.serve("0.0.0.0", 8000)
        server.assert_called_once_with(("0.0.0.0", 8000), api.Handler)

        with (patch.object(config, "API_DEMO_MODE", False),
              patch.object(config, "API_PUBLIC_DEMO_MODE", True)):
            with self.assertRaises(RuntimeError):
                api.serve("0.0.0.0", 8000)


class DirectServiceBoundaryTests(unittest.TestCase):
    def test_direct_service_string_identity_is_rejected_before_pipeline(self):
        with self.assertRaises(service.InvalidUserError):
            service.answer_query("U-005", "hello")

    def test_direct_service_bare_user_is_rejected_before_pipeline(self):
        with self.assertRaises(service.InvalidUserError):
            service.answer_query(get_user("U-005"), "hello")

    def test_direct_stream_string_identity_is_rejected_before_pipeline(self):
        events = service.answer_query_stream("U-005", "hello")
        with self.assertRaises(service.InvalidUserError):
            next(events)


if __name__ == "__main__":
    unittest.main()
