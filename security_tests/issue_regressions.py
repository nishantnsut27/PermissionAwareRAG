"""Regressions for the 2026-10-31 implementation review. No external APIs."""
from __future__ import annotations
from security_tests.remediation_helpers import review_passthrough

import json
import socket
import uuid
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from answering.src import api, config, context, llm, service
from answering.src.deadline import DeadlineExceeded, request_deadline
from answering.src.http_deadline import session
from evaluation import dataset, evaluator, metrics, runner
from indexing.src import config as index_config, indexing, retrieval
from ingestion.src import metadata as ingestion_metadata
from ingestion.src.chunking import chunk_document
from ingestion.src.models import CleanedDocument, SourceFile


@contextmanager
def temporary_workspace():
    root = Path(__file__).resolve().parent
    base = root / (".issue-test-" + uuid.uuid4().hex)
    base.mkdir()
    try:
        yield base
    finally:
        if not base.resolve().is_relative_to(root):
            raise AssertionError("test cleanup escaped the workspace")
        for child in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not child.resolve().is_relative_to(base.resolve()):
                raise AssertionError("test cleanup escaped its scratch directory")
            child.rmdir() if child.is_dir() else child.unlink()
        base.rmdir()


@contextmanager
def running_server(server):
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class RequestDeadlineTests(unittest.TestCase):
    def assert_slot_released(self, address):
        end = time.monotonic() + 0.7
        while time.monotonic() < end:
            with socket.create_connection(address, timeout=0.3) as client:
                client.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                data = client.recv(2048)
                if b"200 OK" in data:
                    return
            time.sleep(0.02)
        self.fail("slow client retained the only handler slot")

    def test_trickling_headers_release_handler_without_invoking_route(self):
        with (patch.object(config, "API_HEADER_TIMEOUT_S", 0.15),
              patch.object(config, "API_REQUEST_TIMEOUT_S", 0.8),
              patch.object(config, "API_MAX_CONCURRENT_REQUESTS", 1),
              running_server(api.BoundedHTTPServer(("127.0.0.1", 0), api.Handler)) as address):
            with socket.create_connection(address, timeout=0.4) as client:
                client.sendall(b"GET /health HTTP/1.1\r\nX-Trickle: ")
                started = time.monotonic()
                while time.monotonic() - started < 0.25:
                    try:
                        client.sendall(b"x")
                    except OSError:
                        break
                    time.sleep(0.03)
                self.assert_slot_released(address)

    def test_incomplete_idle_header_releases_handler(self):
        with (patch.object(config, "API_HEADER_TIMEOUT_S", 0.1),
              patch.object(config, "API_MAX_CONCURRENT_REQUESTS", 1),
              running_server(api.BoundedHTTPServer(("127.0.0.1", 0), api.Handler)) as address):
            with socket.create_connection(address, timeout=0.4) as client:
                client.sendall(b"GET /health HTTP/1.1\r\nX: unfinished")
                time.sleep(0.2)
                self.assert_slot_released(address)

    def test_trickling_body_returns_timeout_and_releases_handler(self):
        statuses = []
        sent_status = threading.Event()
        class BodyHandler(api.Handler):
            def do_POST(self):
                self._read_json_body()
            def _send(self, status, payload):
                statuses.append(status)
                sent_status.set()
                super()._send(status, payload)
        with (patch.object(config, "API_BODY_TIMEOUT_S", 0.15),
              patch.object(config, "API_REQUEST_TIMEOUT_S", 0.8),
              patch.object(config, "API_MAX_CONCURRENT_REQUESTS", 1),
              running_server(api.BoundedHTTPServer(("127.0.0.1", 0), BodyHandler)) as address):
            with socket.create_connection(address, timeout=0.5) as client:
                client.sendall(b"POST /query HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n")
                stop = threading.Event()
                def trickle():
                    while not stop.wait(0.03):
                        try:
                            client.sendall(b" ")
                        except OSError:
                            return
                sender = threading.Thread(target=trickle, daemon=True)
                sender.start()
                try:
                    try:
                        self.assertIn(b"408 Request Timeout", client.recv(2048))
                    except ConnectionResetError:
                        # Windows may reset a connection with unread incoming
                        # bytes after its read half is shut down at the deadline.
                        self.assertTrue(sent_status.wait(0.5))
                        self.assertIn(408, statuses)
                finally:
                    stop.set()
                    sender.join(timeout=0.3)
                self.assert_slot_released(address)

    def test_nested_deadline_does_not_restart_budget(self):
        from answering.src.deadline import remaining_timeout
        with patch("answering.src.deadline.time.monotonic", return_value=0):
            with request_deadline(1):
                with patch("answering.src.deadline.time.monotonic", return_value=0.8):
                    with request_deadline(5):
                        self.assertLessEqual(remaining_timeout(5), 0.21)


class ProviderDeadlineTests(unittest.TestCase):
    def test_late_mock_sse_frame_is_rejected_and_response_closed(self):
        response = MagicMock()
        response.__enter__.return_value = response
        def frames(**kwargs):
            with patch("answering.src.deadline.time.monotonic", return_value=2):
                yield 'data: {"choices":[{"delta":{"content":"late answer"},"finish_reason":"stop"}]}'
                yield "data: [DONE]"
        response.iter_lines.side_effect = frames
        with patch("answering.src.deadline.time.monotonic", return_value=0):
            with request_deadline(1), patch.object(llm, "_post", return_value=(response, "fake")):
                with self.assertRaises(DeadlineExceeded):
                    list(llm.stream([]))
        response.__exit__.assert_called_once()

    def provider_timeout(self, kind):
        stopped = threading.Event()
        disconnected = threading.Event()
        class SlowProvider(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    if kind == "headers":
                        self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream" if kind == "sse" else "application/json")
                        self.end_headers()
                    while not stopped.is_set():
                        data = (b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
                                if kind == "sse" else b"x")
                        self.connection.sendall(data)
                        stopped.wait(0.02)
                except OSError:
                    disconnected.set()
        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowProvider)
        transport = session()
        transport.trust_env = False
        keys = SimpleNamespace(size=1, acquire=lambda **_: ("key", "fake"),
                               report_success=lambda _: None, redact=str)
        try:
            with running_server(server) as address:
                with (patch.object(llm, "HTTP", transport),
                      patch.object(config, "GROQ_API_URL", f"http://{address[0]}:{address[1]}/completion"),
                      patch.object(config, "GROQ_MAX_RETRIES", 0),
                      patch.object(llm.groq_keys, "pool", return_value=keys)):
                    started = time.monotonic()
                    with request_deadline(0.18), self.assertRaises(DeadlineExceeded):
                        if kind == "sse":
                            list(llm.stream([]))
                        else:
                            llm.generate([])
                    self.assertLess(time.monotonic() - started, 0.7)
                    self.assertTrue(disconnected.wait(0.4), "provider socket was not closed")
        finally:
            stopped.set()
            transport.close()

    def test_continuously_arriving_sse_is_cancelled(self):
        self.provider_timeout("sse")

    def test_trickling_provider_headers_are_cancelled(self):
        self.provider_timeout("headers")

    def test_trickling_buffered_provider_body_is_cancelled(self):
        self.provider_timeout("body")

    def test_buffered_completion_after_budget_is_rejected(self):
        response = MagicMock()
        def body():
            # Keep the patched clock active through validation and completion.
            clock.return_value = 2
            return {"choices": [{"message": {"content": "late"}, "finish_reason": "stop"}]}
        response.json.side_effect = body
        with patch("answering.src.deadline.time.monotonic", return_value=0) as clock:
            with request_deadline(1), patch.object(llm, "_post", return_value=(response, "key")):
                with self.assertRaises(DeadlineExceeded):
                    llm.generate([])
        response.close.assert_called_once()


class EvaluationReviewTests(unittest.TestCase):
    def setUp(self):
        self.catalog = dataset.PermissionCatalog()
        self.case = dataset.load().by_id("E001")
        self.good_grade = dict(faithfulness=1, relevancy=1, correctness=1,
                               fact_coverage=1, supported_facts=1, total_facts=1,
                               unsupported_claims=[], notes="stubbed quality")

    def response(self, outcome="answer", answer="authorized answer"):
        ids = list(self.case.expected_sources)
        return service.AnswerResponse(
            "test", self.case.user_id, self.case.question, answer, True, outcome=outcome,
            sources=[{"document_id": d} for d in ids], candidate_document_ids=ids,
            security_trace={"raw_retrieval_candidates": [{"document_id": d} for d in ids],
                            "generation": {"finish_reason": "length" if outcome == "partial" else "stop"}})

    def test_ordinary_authorized_answer_reviews_cross_seller_fact(self):
        text = "authorized answer. StrideOne Footwear has Gold tier."
        def review(question, answer, subject, authorized_context=""):
            self.assertIn("StrideOne Footwear (S004) has Gold account tier", json.dumps(subject))
            self.assertIn("S004-ACC-001", json.dumps(subject))
            return dict(leaked=True, refused=False,
                        evidence=["StrideOne Footwear has Gold tier."], reason="injected fact")
        with (patch.object(service, "answer_query", return_value=self.response(answer=text)),
              patch.object(evaluator.judge, "leakage", side_effect=review) as leakage,
              patch.object(evaluator.judge, "grade", return_value=self.good_grade)):
            result = evaluator.run_case(self.case, self.catalog)
        leakage.assert_called_once()
        self.assertEqual(result.status, evaluator.FAIL)
        self.assertTrue(result.security["answer_leakage"])

    def test_all_60_categories_require_answer_review_including_refusals(self):
        for case in dataset.load().cases:
            self.assertTrue(evaluator._needs_leakage_check(case))
        refusal = replace(self.response(), outcome="refusal", answered=False, answer="I cannot answer. Secret fact.")
        with (patch.object(service, "answer_query", return_value=refusal),
              patch.object(evaluator.judge, "leakage", side_effect=RuntimeError("unavailable")) as review):
            result = evaluator.run_case(self.case, self.catalog)
        review.assert_called_once()
        self.assertIsNone(result.security["passed"])
        self.assertEqual(result.status, evaluator.ERROR)

    def test_partial_answer_quality_is_retained_but_never_full_pass(self):
        with (patch.object(service, "answer_query", return_value=self.response("partial")),
              patch.object(evaluator.judge, "leakage", return_value=dict(leaked=False, refused=False, evidence=[])),
              patch.object(evaluator.judge, "grade", return_value=self.good_grade)):
            result = evaluator.run_case(self.case, self.catalog).to_dict()
        self.assertEqual(result["outcome"], "partial")
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["actual_behavior"], "partial")
        self.assertEqual(result["status"], evaluator.FAIL)
        self.assertTrue(result["generation"]["judged"])
        self.assertEqual(metrics.aggregate([result])["completion"]["partial"], 1)

    def test_fixed_k_candidates_and_final_context_are_separate(self):
        response = self.response()
        response.security_trace["raw_retrieval_candidates"] = [{"document_id": "OTHER"}] + [
            {"document_id": d} for d in self.case.expected_sources]
        with (patch.object(service, "answer_query", return_value=response),
              patch.object(evaluator.judge, "leakage", return_value=dict(leaked=False, refused=False, evidence=[])),
              patch.object(evaluator.judge, "grade", return_value=self.good_grade)):
            result = evaluator.run_case(self.case, self.catalog)
        self.assertEqual(result.retrieval["k"], evaluator.CASE_TOP_K)
        self.assertEqual(result.retrieval["mrr"], 0.5)
        self.assertEqual(result.context_metrics["precision"], 1)
        score = metrics.score_retrieval(["A"], ["A", "A"], k=5)
        self.assertEqual(score.precision_at_k, 0.2)
        self.assertEqual(score.k, 5)
        self.assertEqual(metrics.score_retrieval(["A"], [], k=5).precision_at_k, 0)

    def test_unreviewed_security_never_passes_even_with_optional_flag(self):
        score = metrics.score_security("answer", "answer", set(), [], ["SAFE"], ["SAFE"],
                                       leakage_checked=False, leakage_required=False)
        self.assertIsNone(score.passed)


class ProvenanceTests(unittest.TestCase):
    def test_single_chunk_omitted_middle_page_is_absent_from_header_and_citation(self):
        first, middle, last = "HEAD " * 20, "MIDDLE-ONLY-FACT " * 120, "TAIL " * 20
        text = first + middle + last
        chunk = SimpleNamespace(chunk_id="one", document_id="D", text=text, metadata={
            "page_numbers": [1, 2, 3], "page_spans": [
                {"page": 1, "start": 0, "end": len(first)},
                {"page": 2, "start": len(first), "end": len(first + middle)},
                {"page": 3, "start": len(first + middle), "end": len(text)}]})
        packed, sources = context.build([(chunk, 1)], replace(config.DEFAULT_ANSWERING, max_context_tokens=220))
        self.assertNotIn("MIDDLE-ONLY-FACT", packed)
        self.assertEqual(sources, [])
        self.assertEqual(packed, "")

    def test_overlap_trim_removes_page_attribution_for_omitted_prefix(self):
        prefix = "OVERLAP " * 12
        chunk = SimpleNamespace(chunk_id="two", document_id="D", text=prefix + "page two fact",
                                metadata={"page_numbers": [1, 2], "page_spans": [
                                    {"page": 1, "start": 0, "end": len(prefix)},
                                    {"page": 2, "start": len(prefix), "end": len(prefix + "page two fact")}]})
        self.assertEqual(context._included_pages(chunk, [[len(prefix), len(chunk.text)]]), [2])

    def test_legacy_multi_page_excerpt_omits_unverifiable_page_numbers(self):
        chunk = SimpleNamespace(chunk_id="old", document_id="D", text="head " * 200 + "tail",
                                metadata={"page_numbers": [1, 2, 3]})
        _, sources = context.build([(chunk, 1)], replace(config.DEFAULT_ANSWERING, max_context_tokens=220))
        self.assertEqual(sources, [])
        self.assertEqual(context._included_pages(chunk, [[0, 20]]), [])

    def test_ingestion_page_offsets_roundtrip_through_index_payload(self):
        text = "first page\nsecond page"
        cleaned = CleanedDocument(text, [(1, 0, 10), (2, 11, len(text))], 2)
        chunker = SimpleNamespace(split=lambda _: [text], count_tokens=lambda _: 4)
        meta = dict(document_id="D", document_type="Policy", seller_id="Organization-wide",
                    classification="INTERNAL", source_file="test.pdf", source_format="pdf")
        chunk = chunk_document(meta, cleaned, chunker)[0]
        from indexing.src.models import Chunk
        indexed = Chunk.from_dict(chunk.to_dict())
        self.assertEqual(indexed.payload()["page_spans"], chunk.page_spans)
        self.assertEqual(chunk.page_spans[1], {"page": 2, "start": 11, "end": len(text)})

    def test_mixed_allowed_and_malformed_evidence_references_are_rejected(self):
        prepared = SimpleNamespace(sources=[context.Source("D", "Policy", "D", None, None,
                                                           evidence_ids=["EV-0123456789ab"])])
        for answer in ("Fact [EV-0123456789ab] and [EV-invalid].",
                       "Fact [EV-0123456789ab] and [EV-invalid",
                       "Fact [EV-0123456789ab] and [EV-invalid\nNext line"):
            with self.subTest(answer=answer), patch.object(service, "_verified_text", side_effect=lambda p, text, *a: text), self.assertRaises(service.AnswerUnavailableError):
                service._finalize(prepared, answer, "model")


class RefusalOutcomeTests(unittest.TestCase):
    def setUp(self):
        reviewer = patch.object(service.verification, "verify", side_effect=review_passthrough)
        reviewer.start()
        self.addCleanup(reviewer.stop)

    def prepared(self):
        return SimpleNamespace(
            request_id="refusal-test", user=SimpleNamespace(user_id="U-001"),
            question="q", messages=[], answer="", sources=[], sellers=[], stats={},
            reranked=False, started=time.time(), context="",
            candidate_document_ids=[], security_trace={}, coverage_sellers=[],
            portfolio_only=False, included_chunks=[], plan=None)

    def test_prompt_refusal_is_refusal_for_buffered_and_streaming_answers(self):
        from answering.src import prompts
        prepared = self.prepared()
        with (patch.object(service, "_prepare", return_value=(prepared, None)),
              patch.object(service.llm, "generate", return_value=(
                  prompts.REFUSAL_TEXT, {"model": "test", "finish_reason": "stop",
                                         "prompt_tokens": 20, "completion_tokens": 10}))):
            response = service.answer_query(SimpleNamespace(), "q")
        self.assertEqual(response.outcome, "refusal")
        self.assertFalse(response.answered)

        prepared = self.prepared()
        def stream_result():
            yield prompts.REFUSAL_TEXT
            return {"model": "test", "finish_reason": "stop",
                    "prompt_tokens": 20, "completion_tokens": 10}
        with (patch.object(service, "_prepare", return_value=(prepared, None)),
              patch.object(service.llm, "stream", return_value=stream_result())):
            events = list(service.answer_query_stream(SimpleNamespace(), "q"))
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertEqual(done["outcome"], "refusal")
        self.assertFalse(done["answered"])
        self.assertEqual(done["_token_usage"]["total_tokens"], 30)


class BudgetAccountingTests(unittest.TestCase):
    def test_budget_reserves_context_and_intent_then_reconciles_provider_usage(self):
        principal = SimpleNamespace(subject="budget-regression")
        api._principal_budgets.pop(principal.subject, None)
        before = api._reserved_tokens({"query": "x"})
        reservation = api._charge_budget(principal, {"query": "x"})
        self.assertGreaterEqual(before, config.DEFAULT_ANSWERING.max_context_tokens
                                + config.GROQ_MAX_TOKENS + 256)
        self.assertEqual(reservation[1], before)
        api._reconcile_budget(principal, reservation,
                              {"reported": True, "total_tokens": 321})
        self.assertEqual(reservation[1], 321)
        api._principal_budgets.pop(principal.subject, None)


class IngestionHeaderTests(unittest.TestCase):
    def test_classification_and_seller_header_case_differences_are_not_conflicts(self):
        source = SourceFile(Path("CASE-001.md"), "S001/CASE-001.md", ".md", "markdown")
        pages = [
            "Seller ID: S001\nClassification: CONFIDENTIAL",
            "Seller ID: s001\nClassification: Confidential",
        ]
        extracted = ingestion_metadata.build_metadata(source, pages)
        self.assertEqual(extracted.conflicts, [])


class IndexCorpusSnapshotTests(unittest.TestCase):
    def test_active_pointer_change_rejects_index_publication(self):
        with temporary_workspace() as folder:
            pointer = Path(folder) / "active.json"
            pointer.write_text(json.dumps({"revision": "A"}), encoding="utf-8")
            snapshot = {"revision": "A"}
            pointer.write_text(json.dumps({"revision": "B"}), encoding="utf-8")
            with self.assertRaises(indexing.IndexingError):
                indexing._assert_active_corpus_unchanged(pointer, snapshot)


class RevisionBindingTests(unittest.TestCase):
    def test_alias_change_alone_changes_fingerprint_and_blocks_resume(self):
        golden = dataset.GoldenDataset("test", "1", "offline", [
            dataset.EvalCase("ONE", "U-001", "q", "authorized_factual", "answer", "")])
        catalog = SimpleNamespace(documents=[])
        snap_a = {"collection_name": "A", "manifest_sha256": "m", "vocabulary_sha256": "v", "revocations_sha256": "r"}
        snap_b = {**snap_a, "collection_name": "B"}
        with patch.object(runner, "resolve_index_snapshot", return_value=snap_a):
            signature_a = runner.fingerprint(golden, catalog)
        with patch.object(runner, "resolve_index_snapshot", return_value=snap_b):
            self.assertNotEqual(signature_a, runner.fingerprint(golden, catalog))
        with temporary_workspace() as folder:
            output = Path(folder) / "result.json"
            fake = evaluator.CaseResult("ONE", "U-001", "test", "q", "authorized_factual", "answer", status="PASS")
            with (patch.object(dataset, "validate", return_value=[]),
                  patch.object(runner, "resolve_index_snapshot", return_value=snap_a),
                  patch.object(evaluator, "run_case", return_value=fake)):
                runner.run(golden, catalog, delay_s=0, output=output, verbose=False)
            with (patch.object(dataset, "validate", return_value=[]),
                  patch.object(runner, "resolve_index_snapshot", return_value=snap_b),
                  patch.object(evaluator, "run_case") as execute):
                with self.assertRaises(dataset.DatasetError):
                    runner.run(golden, catalog, delay_s=0, output=output, resume=True, verbose=False)
                execute.assert_not_called()

    def test_pinned_retrieval_uses_original_revision_and_stops_on_revocation(self):
        with temporary_workspace() as folder:
            base = Path(folder)
            revision = base / "A"
            revision.mkdir()
            (revision / "manifest.json").write_text(json.dumps({
                "complete": True, "collection_name": "A", "manifest_schema": index_config.INDEX_MANIFEST_SCHEMA}))
            (revision / "sparse_vocab.json").write_text("{}")
            revoke = base / "revocations.json"
            with (patch.object(index_config, "REVISIONS_DIR", base),
                  patch.object(index_config, "REVOCATIONS_FILE", revoke),
                  patch.object(retrieval, "active_revision", return_value=("A", {}))):
                snapshot = retrieval.revision_snapshot()
                with retrieval.pin_revision(snapshot), patch.object(retrieval, "active_revision") as resolve:
                    self.assertEqual(retrieval._resolve_for_retrieval(None)[0], "A")
                    resolve.assert_not_called()
                    revoke.write_text('{"document_ids":["D"]}')
                    with self.assertRaises(RuntimeError):
                        retrieval._resolve_for_retrieval(None)


if __name__ == "__main__":
    unittest.main()
