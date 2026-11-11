from __future__ import annotations
from security_tests.remediation_helpers import plan_response, review_passthrough

import json
import math
import requests
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from answering.src import config as answer_config
from answering.src import context, llm, reranker, service
from answering.src.identity import Principal
from access_control.src.users import get_user
from indexing.src import config as index_config
from indexing.src import retrieval as index_retrieval
from indexing.src.sparse_search import (
    SPARSE_SCHEMA_VERSION, SPARSE_VARIANT, SparseCompatibilityError,
    SparseEncoder, tokenize,
)


def chunk(chunk_id="c1", text="passage", document_id="D1", pages=None):
    return SimpleNamespace(
        chunk_id=chunk_id, text=text, document_id=document_id,
        metadata={"document_id": document_id, "document_type": "Account Overview",
                  "seller_id": "S001", "department": "Business",
                  "classification": "CONFIDENTIAL",
                  "page_numbers": pages or []})


class SparseRankingTests(unittest.TestCase):
    def test_document_and_query_weights_match_numeric_bm25_fixture(self):
        encoder = SparseEncoder().fit(["alpha alpha beta", "alpha gamma"])
        doc = encoder.encode_document("alpha alpha beta")
        query = encoder.encode_query("alpha beta beta")
        values = dict(zip(doc.indices, doc.values))
        terms = encoder.vocab
        idf_alpha = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
        idf_beta = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
        norm_alpha = 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2.5))
        norm_beta = 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2.5))
        self.assertAlmostEqual(values[terms["alpha"]], idf_alpha * norm_alpha)
        self.assertAlmostEqual(values[terms["beta"]], idf_beta * norm_beta)
        query_values = dict(zip(query.indices, query.values))
        self.assertEqual(query_values, {terms["alpha"]: 1.0,
                                        terms["beta"]: 2.0})
        score = sum(values[index] * weight
                    for index, weight in zip(query.indices, query.values))
        self.assertAlmostEqual(score, idf_alpha * norm_alpha +
                               2 * idf_beta * norm_beta)
        self.assertEqual(SPARSE_VARIANT,
                         "bm25-rsj-positive-idf-linear-query-tf-v2")

    def test_unicode_identifier_hyphens_normalize_to_ascii(self):
        self.assertEqual(tokenize("INC‑2026−0618 inc-2026-0618"),
                         ["inc-2026-0618", "inc-2026-0618"])
        self.assertEqual(tokenize("INC_2026_0618"), ["inc-2026-0618"])

    def test_version_configuration_and_index_report_must_match(self):
        vocab_file = Path(__file__).with_name(".phase3_vocab.json")
        report_file = Path(__file__).with_name(".phase3_index.json")
        try:
            with patch.object(index_config, "VOCAB_FILE", vocab_file), \
                 patch.object(index_config, "INDEXING_REPORT_FILE", report_file):
                encoder = SparseEncoder().fit(["alpha beta", "alpha gamma"])
                encoder.save()
                loaded = SparseEncoder.load()
                self.assertEqual(loaded.fingerprint, encoder.fingerprint)
                manifest = {
                    "complete": True,
                    "collection_name": "revision-1",
                    "sparse_config": {
                        "variant": SPARSE_VARIANT,
                        "vocabulary_fingerprint": encoder.fingerprint,
                    },
                }
                with patch.object(index_retrieval, "_encoders", {}), \
                     patch.object(SparseEncoder, "load", return_value=encoder):
                    self.assertEqual(index_retrieval._sparse_encoder(
                        "revision-1", manifest).fingerprint, encoder.fingerprint)

                with patch.object(index_config, "DEFAULT_RETRIEVAL",
                                  replace(index_config.DEFAULT_RETRIEVAL, rrf_k=61)):
                    with self.assertRaises(SparseCompatibilityError):
                        SparseEncoder.load()

                bad_manifest = {**manifest, "sparse_config": {"variant": SPARSE_VARIANT,
                                "vocabulary_fingerprint": "wrong"}}
                with patch.object(index_retrieval, "_encoders", {}), \
                     patch.object(SparseEncoder, "load", return_value=encoder):
                    with self.assertRaises(SparseCompatibilityError):
                        index_retrieval._sparse_encoder("revision-1", bad_manifest)

                vocab_file.write_text(json.dumps({"vocab": {}, "idf": {},
                                                  "avgdl": 0}), encoding="utf-8")
                with self.assertRaises(SparseCompatibilityError):
                    SparseEncoder.load()
        finally:
            vocab_file.unlink(missing_ok=True)
            report_file.unlink(missing_ok=True)

    def test_non_finite_statistics_are_rejected(self):
        encoder = SparseEncoder()
        encoder.vocab = {"alpha": 0}
        encoder.idf = {"alpha": float("nan")}
        with self.assertRaises(SparseCompatibilityError):
            encoder._validate()

    def test_sparse_query_filter_pins_matching_point_revision(self):
        encoder = SparseEncoder().fit(["alpha beta", "alpha gamma"])
        client = Mock()
        client.query_points.return_value = SimpleNamespace(points=[])
        with patch.object(index_retrieval, "connect", return_value=client), \
             patch.object(index_retrieval, "_sparse_encoder", return_value=encoder):
            index_retrieval.sparse_search(
                "alpha", 5,
                {"must": [{"seller_id": {"any": ["S001"]}}]},
                collection_name="revision-1",
                manifest={"complete": True, "collection_name": "revision-1"})
        compiled = client.query_points.call_args.kwargs["query_filter"].model_dump(
            exclude_none=True)
        encoded = json.dumps(compiled)
        self.assertIn("_sparse_revision", encoded)
        self.assertIn(encoder.fingerprint, encoded)
        self.assertIn("seller_id", encoded)


class RerankerEvidenceTests(unittest.TestCase):
    def test_windows_include_tail_only_evidence(self):
        text = "HEAD-FACT " + ("x" * 2600) + " TAIL-ONLY-FACT"
        windows = reranker.evidence_windows(text)
        self.assertTrue(all(len(window) <= reranker.MAX_DOC_CHARS
                            for window in windows))
        self.assertIn("TAIL-ONLY-FACT", windows[-1])

    def test_duplicate_indices_collapse_to_one_original_chunk(self):
        passage = chunk(text="z" * 3500)
        response = Mock(status_code=200)
        response.json.return_value = {"results": [
            {"index": 0, "relevance_score": 0.4},
            {"index": 1, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
        ]}
        with patch.object(answer_config, "RERANK_ENABLED", True), \
             patch.object(reranker.jina_keys, "pool", return_value=reranker.jina_keys.JinaKeyPool([('JINA_API_KEY', 'offline')])), \
             patch.object(reranker.HTTP, "post", return_value=response):
            result = reranker.rerank("tail fact", [passage])
        self.assertTrue(result.applied)
        self.assertEqual(result.items, [(passage, 0.9)])

    def test_invalid_reranker_values_fall_back_without_scores(self):
        for score in (float("nan"), -0.1, 1.1, "0.5"):
            response = Mock(status_code=200)
            response.json.return_value = {"results": [
                {"index": 0, "relevance_score": score}]}
            with self.subTest(score=score), \
                 patch.object(answer_config, "RERANK_ENABLED", True), \
                 patch.object(reranker.jina_keys, "pool", return_value=reranker.jina_keys.JinaKeyPool([('JINA_API_KEY', 'offline')])), \
                 patch.object(reranker.HTTP, "post", return_value=response):
                result = reranker.rerank("q", [chunk()])
            self.assertFalse(result.applied)
            self.assertEqual(result.items, [(result.items[0][0], None)])

    def test_followup_resolved_query_is_identical_for_retrieval_and_reranking(self):
        user = get_user("U-001")
        passage = chunk(pages=[1])
        access = SimpleNamespace(chunks=[passage], withheld_count=0,
                                 stage_trace={"raw_retrieval_candidates": []})
        seen = {}
        def retrieve(_user, query, **_kwargs):
            seen["retrieval"] = query
            return access
        def rank(query, _chunks, **_kwargs):
            seen["reranker"] = query
            return reranker.RerankOutcome([(passage, 0.8)], True)
        cfg = replace(answer_config.DEFAULT_ANSWERING,
                      max_context_tokens=500, max_context_chars=5000)
        with patch.object(service.conversation, "retrieval_query",
                          return_value="resolved question with prior topic") as resolver, \
             patch.object(service.llm, "generate", side_effect=plan_response), \
             patch.object(service, "authorized_retrieve", side_effect=retrieve), \
             patch.object(service.reranker, "rerank", side_effect=rank):
            service._prepare(Principal(user, "test"), "Which one?",
                             [{"role": "user", "content": "Prior seller topic"}],
                             None, None, "r", cfg)
        resolver.assert_called_once()
        self.assertEqual(seen["retrieval"], seen["reranker"])
        self.assertEqual(seen["retrieval"], "resolved question with prior topic")


class ContextAndCitationTests(unittest.TestCase):
    def setUp(self):
        reviewer = patch.object(service.verification, "verify", side_effect=review_passthrough)
        reviewer.start()
        self.addCleanup(reviewer.stop)

    def test_omitted_chunk_pages_do_not_appear_in_citations(self):
        cfg = replace(answer_config.DEFAULT_ANSWERING,
                      max_context_tokens=100, max_context_chars=5000)
        first = chunk("c1", "A short included evidence span.", "D1", [1])
        second = chunk("c2", "B " * 300, "D2", [2, 3])
        text, sources = context.build([(first, 0.9), (second, 0.8)], cfg)
        self.assertLessEqual(context.count_tokens(text), cfg.max_context_tokens)
        self.assertEqual([source.pages for source in sources], [[1]])
        self.assertEqual([source.chunk_ids for source in sources], [["c1"]])

    def test_oversized_complete_chunk_is_omitted_inside_strict_budget(self):
        cfg = replace(answer_config.DEFAULT_ANSWERING,
                      max_context_tokens=180, max_context_chars=5000)
        passage = chunk(text="HEAD-ONLY-FACT " + ("filler " * 300) + " TAIL-ONLY-FACT",
                        pages=[1, 2])
        split = len(passage.text) // 2
        passage.metadata["page_spans"] = [
            {"page": 1, "start": 0, "end": split},
            {"page": 2, "start": split, "end": len(passage.text)}]
        text, sources = context.build([(passage, 0.8)], cfg)
        self.assertLessEqual(context.count_tokens(text), cfg.max_context_tokens)
        self.assertEqual(text, "")
        self.assertEqual(sources, [])

    def test_duplicate_overlapping_spans_keep_provenance(self):
        cfg = replace(answer_config.DEFAULT_ANSWERING,
                      max_context_tokens=500, max_context_chars=5000)
        text_value = "Same complete text with enough content to be a real passage."
        text, sources = context.build([
            (chunk("c1", text_value, "D1", [4]), 0.9),
            (chunk("c2", text_value, "D1", [5]), 0.8),
        ], cfg)
        self.assertEqual(text.count(text_value), 2)
        self.assertEqual(sources[0].chunk_ids, ["c1", "c2"])
        self.assertEqual(sources[0].pages, [4, 5])

    def test_complete_overlapping_chunks_preserve_text_and_stable_evidence_ids(self):
        cfg = replace(answer_config.DEFAULT_ANSWERING,
                      max_context_tokens=1000, max_context_chars=5000)
        shared = "Shared passage text " + ("detail " * 8)
        first = chunk("c1", "Beginning fact. " + shared, "D1", [1])
        second = chunk("c2", shared + " Tail-only fact.", "D1", [2])
        first_context, first_sources = context.build([(first, 0.9), (second, 0.8)], cfg)
        _, again_sources = context.build([(first, 0.9), (second, 0.8)], cfg)
        self.assertEqual(first_context.count("Shared passage text"), 2)
        self.assertIn("Tail-only fact.", first_context)
        self.assertEqual(first_sources[0].chunk_ids, ["c1", "c2"])
        self.assertEqual(first_sources[0].pages, [1, 2])
        self.assertEqual(first_sources[0].evidence_ids,
                         again_sources[0].evidence_ids)

    def test_invalid_evidence_ids_fail_and_inventory_maps_to_included_pages(self):
        text, sources = context.build([(chunk("c1", "Included fact.", "D1", [7]), 0.9)])
        user = get_user("U-001")
        prepared = service._Prepared(
            "r", user, "question", [], sources, [], {}, False, 0,
            text, [], {})
        with self.assertRaises(service.AnswerUnavailableError):
            service._finalize(prepared, "A fact [EV-000000000000].", "test")
        response = service._finalize(prepared, "A fact.", "test")
        source = response.to_dict()["sources"][0]
        self.assertEqual(source["pages"], [7])
        self.assertEqual(source["evidence_ids"], sources[0].evidence_ids)
        self.assertIn("Evidence available in context:", response.answer)
        cited = service._finalize(
            prepared, f"A fact [{sources[0].evidence_ids[0]}].", "test")
        self.assertNotIn("[EV-", cited.answer.split("Evidence available in context:", 1)[0])

    def test_source_only_model_output_is_not_an_answer(self):
        _, sources = context.build([(chunk(), 0.9)])
        prepared = service._Prepared("r", get_user("U-001"), "q", [],
                                     sources, [], {}, False, 0, "ctx", [], {})
        with self.assertRaises(service.AnswerUnavailableError):
            service._finalize(prepared, "Sources:\n- Runbook D1", "test")


class CompletionContractTests(unittest.TestCase):
    def setUp(self):
        reviewer = patch.object(service.verification, "verify", side_effect=review_passthrough)
        reviewer.start()
        self.addCleanup(reviewer.stop)

    @staticmethod
    def _response(events):
        response = MagicMock()
        response.encoding = None
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_lines.return_value = iter(events)
        return response

    def test_buffered_generation_requires_terminal_and_distinguishes_length(self):
        response = Mock()
        response.json.return_value = {"model": "m", "choices": [{
            "message": {"content": "partial text"}, "finish_reason": "length"}]}
        with patch.object(llm, "_post", return_value=(response, "k")), \
             patch.object(answer_config, "GROQ_MAX_RETRIES", 0):
            answer, usage = llm.generate([])
        self.assertEqual(answer, "partial text")
        self.assertEqual(usage["outcome"], "partial")
        response.json.return_value = {"choices": [{
            "message": {"content": "answer"}, "finish_reason": None}]}
        with patch.object(llm, "_post", return_value=(response, "k")), \
             patch.object(answer_config, "GROQ_MAX_RETRIES", 0):
            with self.assertRaises(llm.LLMError):
                llm.generate([])

        response.json.side_effect = ValueError("bad json")
        with patch.object(llm, "_post", return_value=(response, "k")), \
             patch.object(answer_config, "GROQ_MAX_RETRIES", 0):
            with self.assertRaises(llm.LLMError):
                llm.generate([])

    def test_empty_malformed_eof_and_disconnect_streams_fail(self):
        cases = {
            "empty": [],
            "malformed": ["data: {not-json}"],
            "premature": ['data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}'],
        }
        for name, events in cases.items():
            with self.subTest(name=name), \
                 patch.object(llm, "_post", return_value=(self._response(events), "k")), \
                 patch.object(answer_config, "GROQ_MAX_RETRIES", 0):
                with self.assertRaises(llm.LLMError):
                    list(llm.stream([]))

        response = self._response([])
        def disconnect_after_delta(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}'
            raise requests.ConnectionError("drop")
        response.iter_lines.side_effect = disconnect_after_delta
        with patch.object(llm, "_post", return_value=(response, "k")), \
             patch.object(answer_config, "GROQ_MAX_RETRIES", 0):
            with self.assertRaises(llm.LLMError):
                list(llm.stream([]))

    def test_stream_requires_done_and_returns_partial_for_length(self):
        events = [
            'data: {"choices":[{"delta":{"content":"text"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
            "data: [DONE]",
        ]
        with patch.object(llm, "_post", return_value=(self._response(events), "k")):
            iterator = llm.stream([])
            self.assertEqual(next(iterator), "text")
            with self.assertRaises(StopIteration) as stopped:
                next(iterator)
        self.assertEqual(stopped.exception.value["outcome"], "partial")

    def test_buffered_and_streaming_service_results_share_terminal_contract(self):
        _, sources = context.build([(chunk("c1", "Evidence text", "D1", [1]), 0.9)])
        user = get_user("U-001")
        prepared = service._Prepared("r", user, "question", [], sources, [], {},
                                     False, 0, "context", [], {})
        with patch.object(service, "_prepare", return_value=(prepared, None)), \
             patch.object(service.llm, "generate", return_value=("A concise answer.", {
                 "model": "m", "finish_reason": "stop"})):
            buffered = service.answer_query(Principal(user, "test"), "question",
                                            request_id="r")

        def streamed():
            yield "A concise "
            yield "answer."
            return {"model": "m", "finish_reason": "stop"}
        with patch.object(service, "_prepare", return_value=(prepared, None)), \
             patch.object(service.llm, "stream", side_effect=lambda _messages: streamed()):
            events = list(service.answer_query_stream(Principal(user, "test"),
                                                       "question", request_id="r"))
        terminal = events[-1]
        self.assertEqual(terminal["type"], "done")
        self.assertEqual(terminal["outcome"], buffered.outcome)
        self.assertEqual(terminal["answer"], buffered.answer)
        self.assertEqual(terminal["sources"], buffered.to_dict()["sources"])

        with self.assertRaises(service.AnswerUnavailableError):
            service._finalize(prepared, "Text cut off.", "m", "length")

    def test_stream_provider_failure_has_explicit_error_outcome(self):
        user = get_user("U-001")
        prepared = service._Prepared("r", user, "q", [], [], [], {}, False,
                                     0, "context", [], {})
        with patch.object(service, "_prepare", return_value=(prepared, None)), \
             patch.object(service.llm, "stream", side_effect=llm.LLMError("offline")):
            events = list(service.answer_query_stream(Principal(user, "test"), "q"))
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["outcome"], "error")
        self.assertFalse(events[-1]["completed"])

if __name__ == "__main__":
    unittest.main()
