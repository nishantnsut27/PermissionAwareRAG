"""Offline fault-injection checks for Phase 4 reliability boundaries."""
from __future__ import annotations

import json
import hashlib
import io
import socket
import threading
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indexing.src import config as index_config
from indexing.src import revisions
from ingestion.src import cleaning, config as ingestion_config, metadata, validation
from ingestion.src.models import SourceFile, ExtractedMetadata
from answering.src import api, config as answer_config, conversation, service
from answering.src.deadline import (DeadlineExceeded, record_retry,
                                    remaining_timeout, request_deadline,
                                    retry_snapshot)
from indexing.src import embeddings
from indexing.src import indexing as index_pipeline
from indexing.src import retrieval as index_retrieval
from indexing.src.models import Chunk as IndexChunk
from indexing.src.qdrant_store import upsert_points


class FakeClient:
    def __init__(self, ids):
        self.ids = list(ids)
        self.aliases = [SimpleNamespace(alias_name=revisions.logical_alias(),
                                         collection_name="old-revision")]
        self.fail_switch = False

    def count(self, **kwargs):
        return SimpleNamespace(count=len(self.ids))

    def scroll(self, **kwargs):
        return ([SimpleNamespace(id=item, payload={}) for item in self.ids], None)

    def get_aliases(self):
        return SimpleNamespace(aliases=self.aliases)

    def update_collection_aliases(self, change_aliases_operations):
        if self.fail_switch:
            raise RuntimeError("injected alias fault")
        op = change_aliases_operations[-1]
        self.aliases = [SimpleNamespace(
            alias_name=op.create_alias.alias_name,
            collection_name=op.create_alias.collection_name)]


def manifest_for(collection, point_ids):
    return {"complete": True, "collection_name": collection,
            "manifest_schema": index_config.INDEX_MANIFEST_SCHEMA,
            "point_ids": sorted(point_ids),
            "point_payload_hashes": {
                point_id: hashlib.sha256(b"{}").hexdigest()
                for point_id in point_ids}}


class RevisionLifecycleTests(unittest.TestCase):
    def test_incomplete_upload_is_never_activated(self):
        client = FakeClient([])
        with self.assertRaises(index_pipeline.IndexingError):
            index_pipeline.finalize_revision(client, "partial", {
                "complete": False, "collection_name": "partial"}, set())
        self.assertEqual(client.aliases[0].collection_name, "old-revision")

    def test_upload_failure_keeps_old_alias(self):
        class FailingUpload(FakeClient):
            def upsert(self, **kwargs):
                raise RuntimeError("injected upload fault")
        client = FailingUpload([])
        chunk = IndexChunk(chunk_id="c1", text="sample", document_id="d1")
        indexed, failed, errors = upsert_points(
            client, [(chunk, [0.1, 0.2], {"indices": [0], "values": [1.0]})],
            collection_name="partial")
        self.assertEqual((indexed, failed), (0, 1))
        self.assertTrue(errors)
        with self.assertRaises(index_pipeline.IndexingError):
            index_pipeline.finalize_revision(client, "partial", {
                "complete": False, "collection_name": "partial"}, set())
        self.assertEqual(client.aliases[0].collection_name, "old-revision")
    def test_exact_point_mismatch_blocks_publication_and_old_alias_survives(self):
        with patch.object(revisions, "_atomic_json") as write_manifest:
            client = FakeClient(["one", "unexpected"])
            manifest = manifest_for("next", {"one"})
            with self.assertRaises(revisions.RevisionError):
                revisions.publish_alias(client, "next", manifest, {"one"},
                                        manifest["point_payload_hashes"])
            self.assertEqual(client.aliases[0].collection_name, "old-revision")
            write_manifest.assert_not_called()

    def test_atomic_alias_fault_leaves_previous_revision_active(self):
        with patch.object(revisions, "_atomic_json") as write_manifest:
            client = FakeClient(["one"])
            client.fail_switch = True
            with self.assertRaisesRegex(RuntimeError, "injected"):
                manifest = manifest_for("next", {"one"})
                revisions.publish_alias(client, "next", manifest, {"one"},
                                        manifest["point_payload_hashes"])
            self.assertEqual(client.aliases[0].collection_name, "old-revision")
            # Manifest persistence is complete before the atomic alias operation.
            write_manifest.assert_called_once()

    def test_point_count_and_ids_are_both_checked(self):
        client = FakeClient(["one", "one"])
        with self.assertRaises(revisions.RevisionError):
            revisions.validate_collection(client, "x", {"one", "two"})

    def test_point_payload_hash_mismatch_blocks_activation(self):
        client = FakeClient(["one"])
        with self.assertRaisesRegex(revisions.RevisionError, "payload/ACL"):
            revisions.validate_collection(client, "x", {"one"}, {"one": "wrong"})

    def test_rollback_requires_same_corpus_and_acl_fingerprint(self):
        def read_text(path, encoding=None):
            revision = path.parent.name
            fingerprint = "old" if revision == "target" else "current"
            document = manifest_for(revision, {"one"})
            document["corpus_fingerprint"] = fingerprint
            if path.name == "active.json":
                document["collection_name"] = "current"
            return json.dumps(document)
        with patch.object(index_config, "REVISIONS_DIR", Path("fake-revisions")), \
             patch.object(Path, "read_text", read_text):
            client = FakeClient(["one"])
            with self.assertRaisesRegex(revisions.RevisionError, "stale"):
                revisions.rollback(client, "target")
            self.assertEqual(client.aliases[0].collection_name, "old-revision")


class IngestionSafetyTests(unittest.TestCase):
    def test_repeated_body_heading_is_retained_but_positional_header_removed(self):
        pages = [
            ["Nexora report", "Seller S001", "", "Pricing", "Body", "Pricing", "p1"],
            ["Nexora report", "Seller S001", "", "Pricing", "More", "Pricing", "p2"],
            ["Nexora report", "Seller S001", "", "Pricing", "Other", "Pricing", "p3"],
        ]
        cleaned = cleaning.strip_repeated_lines(pages)
        self.assertNotIn("Nexora report", cleaned[0])
        self.assertEqual(cleaned[0].count("Pricing"), 2)

    def test_unknown_path_scope_is_not_organization_wide(self):
        source = SourceFile(Path("unknown.pdf"), "misc/unknown.pdf", ".pdf", "pdf")
        extracted = metadata.build_metadata(source, [
            "Document ID UNKNOWN-1 Document Type Organization Reference "
            "Classification CONFIDENTIAL"])
        self.assertIsNone(extracted.values.get("seller_id"))
        issues = validation.validate(source, extracted)
        self.assertIn("SELLER_ID_MISSING", {issue.code for issue in issues})

    def test_metadata_conflicts_and_unknown_type_fail_closed(self):
        source = SourceFile(Path("x.pdf"), "misc/x.pdf", ".pdf", "pdf")
        md = ExtractedMetadata({"document_id": "x", "document_type": "unknown",
                                "seller_id": "S001", "classification": "INTERNAL"})
        self.assertTrue(validation.has_errors(validation.validate(source, md)))


class RequestAndProviderTests(unittest.TestCase):
    def _handler(self, body=b"", length=None):
        handler = object.__new__(api.Handler)
        handler.headers = {"Content-Length": str(len(body) if length is None else length)}
        handler.rfile = io.BytesIO(body)
        handler.connection = SimpleNamespace(settimeout=MagicMock())
        return handler

    def test_malformed_and_oversized_bodies_are_rejected(self):
        for handler, status in ((self._handler(b"{"), 400),
                                (self._handler(b"x", api.MAX_BODY_BYTES + 1), 400)):
            with patch.object(handler, "_send") as send:
                self.assertIsNone(handler._read_json_body())
            self.assertEqual(send.call_args.args[0], status)

    def test_slow_request_body_gets_408(self):
        handler = self._handler(b"{}")
        handler.rfile = SimpleNamespace(read=MagicMock(side_effect=socket.timeout()))
        with patch.object(handler, "_send") as send:
            self.assertIsNone(handler._read_json_body())
        self.assertEqual(send.call_args.args[0], 408)

    def test_admission_overload_returns_503_without_spawning_worker(self):
        semaphore = threading.BoundedSemaphore(1)
        semaphore.acquire()
        rejections = threading.BoundedSemaphore(2)
        rejections.acquire()
        rejections.acquire()
        server = SimpleNamespace(_admission=semaphore, _rejections=rejections, shutdown_request=MagicMock())
        request = SimpleNamespace(sendall=MagicMock())
        api.BoundedHTTPServer.process_request(server, request, ("127.0.0.1", 1))
        request.sendall.assert_not_called()
        server.shutdown_request.assert_called_once_with(request)

    def test_client_disconnect_closes_stream_generator(self):
        closed = []
        def events():
            try:
                yield {"type": "meta"}
                yield {"type": "delta", "text": "x"}
            finally:
                closed.append(True)
        handler = object.__new__(api.Handler)
        handler.close_connection = False
        principal = SimpleNamespace(subject="subject")
        with patch.object(handler, "_read_json_body", return_value={"query": "q"}), \
             patch.object(api, "_charge_budget", return_value=True), \
             patch.object(api.service, "answer_query_stream", return_value=events()), \
             patch.object(handler, "_open_event_stream"), \
             patch.object(handler, "_emit", side_effect=BrokenPipeError()):
            handler._handle_query_stream(principal)
        self.assertEqual(closed, [True])
        self.assertTrue(handler.close_connection)

    def test_service_stream_cancellation_closes_provider_iterator(self):
        closed = []
        def provider_stream(_messages):
            try:
                yield "partial"
                raise service.llm.LLMError("cancelled upstream")
            finally:
                closed.append(True)
        user = service.USERS["U-001"]
        prepared = SimpleNamespace(
            request_id="request", user=user, sources=[], sellers=[], stats={},
            reranked=False, messages=[], security_trace={})
        with patch.object(service, "_prepare", return_value=(prepared, None)), \
             patch.object(service.llm, "stream", side_effect=provider_stream):
            events = service.answer_query_stream(
                service.identity.Principal(user, "verified"), "question")
            self.assertEqual(next(events)["type"], "meta")
            self.assertEqual(next(events)["type"], "error")
            events.close()
        self.assertEqual(closed, [True])

    def test_malformed_history_and_oversized_query_rejected(self):
        with self.assertRaises(ValueError):
            conversation.parse_history([None])
        with self.assertRaises(service.InvalidQueryError):
            service._clean_query("x" * (answer_config.DEFAULT_ANSWERING.max_query_chars + 1),
                                 answer_config.DEFAULT_ANSWERING)
        principal = service.identity.Principal(
            service.USERS["U-001"], "verified-subject")
        with self.assertRaises(ValueError):
            service._prepare(principal, "q", [], None, True, "req",
                             answer_config.DEFAULT_ANSWERING)
        with self.assertRaises(ValueError):
            service._prepare(principal, "q", [], ["S999"], None, "req",
                             answer_config.DEFAULT_ANSWERING)

    def test_unknown_api_fields_are_rejected_but_spoof_field_is_ignored(self):
        self.assertFalse(api._validate_payload({"query": "q", "admin": True}, False))
        self.assertTrue(api._validate_payload({"query": "q", "user_id": "U-005"}, False))

    def test_budgets_key_on_verified_subject_not_body_user_id(self):
        api._principal_budgets.clear()
        principal_a = SimpleNamespace(subject="api:regular-user")
        payload = {"query": "x", "user_id": "U-005"}
        self.assertTrue(api._charge_budget(principal_a, payload))
        self.assertEqual(list(api._principal_budgets), ["api:regular-user"])
        self.assertNotIn("U-005", api._principal_budgets)

    def test_deadline_is_propagated_as_remaining_timeout(self):
        with request_deadline(1):
            self.assertGreater(remaining_timeout(5), 0)
            self.assertLessEqual(remaining_timeout(5), 1)
            record_retry("embedding")
            self.assertEqual(retry_snapshot(), {"embedding": 1})
        with patch("answering.src.deadline.time.monotonic", return_value=0):
            with request_deadline(1), patch(
                    "answering.src.deadline.time.monotonic", return_value=2):
                with self.assertRaises(DeadlineExceeded):
                    remaining_timeout(5)

    def test_malformed_provider_response_is_not_retried(self):
        response = MagicMock()
        response.json.return_value = {"choices": []}
        with patch.object(answer_config, "GROQ_MAX_RETRIES", 3), \
             patch.object(service.llm, "_post", return_value=(response, "key")) as post:
            with self.assertRaises(service.llm.LLMError):
                service.llm.generate([])
        self.assertEqual(post.call_count, 1)

    def test_permanent_provider_auth_error_is_not_retried(self):
        response = SimpleNamespace(status_code=401, text="unauthorized")
        with patch.object(embeddings.jina_keys, "pool", return_value=embeddings.jina_keys.JinaKeyPool([('JINA_API_KEY', 'fake')])), \
             patch.object(embeddings.HTTP, "post", return_value=response) as post:
            result, failed = embeddings.embed_texts(["offline fixture"], max_retries=4)
        self.assertEqual(result, [None])
        self.assertEqual(failed, [0])
        self.assertEqual(post.call_count, 1)

    def test_transient_provider_429_uses_bounded_retries(self):
        response = SimpleNamespace(status_code=429, text="rate limited")
        with patch.object(embeddings.jina_keys, "pool", return_value=embeddings.jina_keys.JinaKeyPool([('JINA_API_KEY', 'fake')])), \
             patch.object(embeddings.HTTP, "post", return_value=response) as post, \
             patch("indexing.src.embeddings.time.sleep") as sleep:
            _, failed = embeddings.embed_texts(["offline fixture"], max_retries=2)
        self.assertEqual(failed, [0])
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_transient_5xx_and_timeout_retries_are_bounded(self):
        response = SimpleNamespace(status_code=503, text="unavailable")
        with patch.object(embeddings.jina_keys, "pool", return_value=embeddings.jina_keys.JinaKeyPool([('JINA_API_KEY', 'fake')])), \
             patch.object(embeddings.HTTP, "post", return_value=response) as post, \
             patch("indexing.src.embeddings.time.sleep"):
            _, failed = embeddings.embed_texts(["offline fixture"], max_retries=2)
        self.assertEqual(failed, [0])
        self.assertEqual(post.call_count, 3)
        with patch.object(embeddings.jina_keys, "pool", return_value=embeddings.jina_keys.JinaKeyPool([('JINA_API_KEY', 'fake')])), \
             patch.object(embeddings.HTTP, "post",
                          side_effect=embeddings.requests.Timeout("offline")) as post, \
             patch("indexing.src.embeddings.time.sleep"):
            _, failed = embeddings.embed_texts(["offline fixture"], max_retries=1)
        self.assertEqual(failed, [0])
        self.assertEqual(post.call_count, 2)

    def test_emergency_revocation_is_merged_and_malformed_file_denies(self):
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text",
                          return_value=json.dumps({"document_ids": ["D-1"]})):
            result = index_retrieval._effective_filter({"seller_id": {"any": ["S001"]}})
        self.assertEqual(result["must_not"], [{"document_id": {"any": ["D-1"]}}])
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "read_text", return_value="bad json"):
            with self.assertRaises(RuntimeError):
                index_retrieval._effective_filter({"seller_id": {"any": ["S001"]}})

    def test_invalid_numeric_configuration_fails_at_startup(self):
        with patch.object(answer_config, "API_MAX_CONCURRENT_REQUESTS", 0):
            with self.assertRaises(RuntimeError):
                answer_config.validate_settings()


if __name__ == "__main__":
    unittest.main()
