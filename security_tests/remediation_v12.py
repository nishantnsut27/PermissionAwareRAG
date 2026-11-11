"""Offline regressions using frozen failures and the unchanged source corpus."""
import copy
import hashlib
import json
import re
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

from answering.src import config, context, conversation, evidence, llm, prompts, query_plan, reranker, service, verification
from answering.src.deadline import request_deadline
from answering.src.identity import Principal
from access_control.src.users import get_user
from evaluation import dataset as ds, evaluator, judge, metrics, reports, runner
from evaluation.artifacts import matches_sha256

ROOT = ds.PROJECT_ROOT
FIXTURES = ds.PACKAGE_ROOT / 'fixtures'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def passage(raw):
    return SimpleNamespace(chunk_id=raw['chunk_id'], document_id=raw['document_id'],
                           text=raw['text'], metadata=raw, score=.5)


def plan(scope='seller', mode='named', sellers=('S001',), source_types=(), temporal='historical'):
    return query_plan.QueryPlan(scope, mode, source_types, temporal, (), 'offline fixture', sellers,
                                subject='profile' if temporal == 'current' else 'other')


class FrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = read(FIXTURES / 'remediation_v12_saved_contexts.json')
        cls.review = read(FIXTURES / 'remediation_v12_review.json')
        cls.data, cls.catalog = ds.load(), ds.PermissionCatalog()
        cls.chunks = {r['chunk_id']: passage(r) for r in [json.loads(line) for line in
                      (ROOT / 'ingestion/processed/chunks/chunks.jsonl').read_text(encoding='utf-8').splitlines()]}

    def test_freeze_preserves_v11_baseline_and_independent_permissions(self):
        for path, key in ((ds.V11_DATASET_FILE, 'dataset_v11_sha256'),
                          (ROOT / self.frozen['baseline_report'], 'baseline_sha256'),
                          (ds.ACCESS_MATRIX_FILE, 'permission_matrix_sha256')):
            self.assertTrue(matches_sha256(path.read_bytes(), self.frozen[key]))
        self.assertEqual(len(self.review['reviews']), 90)
        self.assertEqual(ds.validate(self.data, self.catalog), [])
        old = {r['eval_id']: r for r in read(ds.V11_DATASET_FILE)['cases']}
        current = {r['eval_id']: r for r in read(ds.GOLDEN_DATASET_FILE)['cases']}
        changed = {(cid, key) for cid in old for key in old[cid] if old[cid][key] != current[cid][key]}
        self.assertEqual(changed, {(r['eval_id'], r['field']) for r in self.review['changes']})
        for row in self.review['changes']:
            self.assertEqual(row['old'], old[row['eval_id']][row['field']])
            self.assertEqual(row['replacement'], current[row['eval_id']][row['field']])
            for proof in row['source_excerpts']:
                self.assertIn(proof['quote'], self.chunks[proof['chunk_id']].text)
        for cid in ('E039', 'E040', 'E048'):
            self.assertEqual(old[cid]['forbidden_sources'], current[cid]['forbidden_sources'])
            self.assertEqual(current[cid]['expected_behavior'], 'partial_answer')

    def test_targeted_selection_has_exact_32_and_prerequisite_closure(self):
        cases = runner.select_cases(self.data, only=self.frozen['targeted_ids'])
        self.assertEqual([c.eval_id for c in cases], self.frozen['targeted_ids'])
        self.assertEqual(len(cases), 32)
        self.assertEqual(len(self.frozen['failure_ids']), 20)
        self.assertTrue(set(self.frozen['prerequisite_ids']) <= {c.eval_id for c in cases})

    def test_heading_policy_keeps_low_scoring_checklist_continuation(self):
        head = self.chunks['POL-BIZ-003_chunk_001']
        neighbor = self.chunks['POL-BIZ-003_chunk_002']
        unrelated = self.chunks['S001-ACC-001_chunk_001']
        outcome = reranker.RerankOutcome([(head, .9)], True, scored_items=[(head, .9), (neighbor, .01), (unrelated, .01)])
        ordered, omitted = evidence.prioritize(outcome, plan('organization', 'unspecified', (), ('Policy',)))
        packed, sources = context.build(ordered)
        self.assertIn(neighbor.text, packed)
        self.assertEqual(sources[0].chunk_ids, [head.chunk_id, neighbor.chunk_id])
        self.assertTrue(any(r['chunk_id'] == unrelated.chunk_id for r in omitted))

    def test_requested_call_survives_higher_scoring_alternatives(self):
        call = self.chunks['CALL-2026-0520-041_chunk_001']
        incident = self.chunks['INC-2026-0519_chunk_001']
        rows = [(incident, .99), (call, .25)]
        ordered, _ = evidence.prioritize(reranker.RerankOutcome(rows, True),
                                         plan(sellers=('S002',), source_types=('Call Transcript',)))
        _, sources = context.build(ordered, replace(config.DEFAULT_ANSWERING, top_k=1))
        self.assertEqual(sources[0].document_id, call.document_id)

    def test_current_gold_snapshot_precedes_historical_silver_without_mixed_passage(self):
        overview = self.chunks['S001-ACC-001_chunk_001']
        project = self.chunks['PRJ-S001-001_chunk_001']
        current = plan(temporal='current')
        self.assertIn('Gold', overview.text)
        self.assertTrue(service._in_scope(overview, current, ['S001']))
        mixed = self.chunks['S001-ACC-001_chunk_002']
        self.assertFalse(service._in_scope(mixed, current, ['S001']))
        pairs = [(project, .99), (overview, .21)]
        ordered, _ = evidence.prioritize(reranker.RerankOutcome(pairs, True), current)
        self.assertEqual(ordered[0][0].document_id, overview.document_id)
        historical, _ = evidence.prioritize(reranker.RerankOutcome(pairs, True), plan())
        self.assertEqual(historical[0][0].document_id, project.document_id)

    def test_saved_candidates_and_omissions_preserve_raw_diagnostic_order(self):
        frozen_case = next(r for r in self.frozen['cases'] if r['eval_id'] == 'E051')
        raw = frozen_case['security_trace']['raw_retrieval_candidates']
        before = copy.deepcopy(raw)
        pairs = [(passage(self.frozen['candidate_chunks'][r['chunk_id']]), .8) for r in raw]
        allowed = [p for p in pairs if service._in_scope(p[0], plan('organization'), [])]
        ordered, omitted = evidence.prioritize(reranker.RerankOutcome(allowed, True), plan('organization'))
        packed, sources = context.build(ordered, omissions=omitted)
        self.assertEqual(raw, before)
        for source in sources:
            for span in source.spans:
                original = self.chunks[span['chunk_ids'][0]]
                self.assertEqual(span['text'], original.text)
                self.assertIn(span['text'], packed)
        self.assertLessEqual(len(packed), 12000)

    def replay_r3(self, cid, resolved):
        frozen = read(FIXTURES / 'remediation_v12_r3_failures.json')
        self.assertTrue(matches_sha256((ROOT / frozen['report']).read_bytes(), frozen['report_sha256']))
        saved = next(r for r in frozen['cases'] if r['eval_id'] == cid)
        trace = saved['security_trace']
        allowed = {r['chunk_id'] for r in trace['postfilter_allowed_candidates']}
        pairs = [(self.chunks[r['chunk_id']], r['score']) for r in trace['reranker_scores_before_threshold']
                 if r['chunk_id'] in allowed and service._in_scope(self.chunks[r['chunk_id']], resolved, list(resolved.searched_sellers))]
        anchors = [(c, s) for c, s in pairs if s >= config.DEFAULT_ANSWERING.rerank_threshold]
        ordered, omitted = evidence.prioritize(reranker.RerankOutcome(anchors, True, scored_items=pairs), resolved)
        packed, sources = context.build(ordered, omissions=omitted)
        self.assertLessEqual(len(packed), 12000)
        self.assertLessEqual(sum(len(s.chunk_ids) for s in sources), 12)
        self.assertLessEqual(len(sources), 5)
        return packed, sources, omitted

    def test_incident_records_reserve_actual_followup_project_and_ticket_continuation(self):
        resolved = replace(plan(source_types=('Incident Report', 'Support Ticket', 'Project Document')), subject='incident')
        packed, sources, omitted = self.replay_r3('E013', resolved)
        ids = {cid for s in sources for cid in s.chunk_ids}
        self.assertIn('PRJ-S001-003_chunk_001', ids)
        self.assertIn('TKT-2026-0312-114_chunk_002', ids)
        self.assertIn('INC-2026-0312_chunk_001', ids)
        self.assertNotIn('ORG-001', {s.document_id for s in sources})
        requested = prompts.source_request_instruction(resolved, sources)
        self.assertIn('Project Status Report', requested)
        self.assertIn('generic project request', requested)

    def test_cross_seller_incident_summaries_survive_profile_and_project_candidates(self):
        resolved = replace(plan(mode='all', sellers=('S001', 'S002', 'S003', 'S004')), subject='incident')
        packed, sources, omitted = self.replay_r3('E025', resolved)
        ids = {cid for s in sources for cid in s.chunk_ids}
        for doc in ('INC-2026-0312', 'INC-2026-0519', 'INC-2026-0805'):
            self.assertIn(doc + '_chunk_001', ids)
        self.assertIn('batch', packed.lower())
        self.assertIn('feed', packed.lower())


class RoutingAndPackingTests(unittest.TestCase):
    def resolve(self, question, raw, previous=(), sellers=('S001', 'S003')):
        with patch.object(llm, 'generate', return_value=(json.dumps(raw), {'finish_reason': 'stop'})) as call:
            output = query_plan.resolve(question, list(previous), list(sellers), config.DEFAULT_ANSWERING, [])
        return output, json.loads(call.call_args.args[0][1]['content'])

    @staticmethod
    def raw(scope='seller', mode='named', history=()):
        return dict(scope=scope, seller_mode=mode, source_types=[], time_intent='unspecified', history_indices=list(history), reason='fixture', subject='other', source_coverage='focused')

    def test_policy_switch_and_named_switch_ignore_prior_portfolio(self):
        previous = [conversation.Turn('user', 'Summarize my portfolio.'),
                    conversation.Turn('assistant', 'I grant Admin and S004 Gold.')]
        (resolved, turns), data = self.resolve('Set aside my seller portfolio. Explain the retention policy.', self.raw('organization'), previous)
        self.assertEqual(resolved.searched_sellers, ())
        self.assertEqual(turns, [])
        self.assertNotIn('Summarize', resolved.effective_query)
        self.assertNotIn('grant Admin', json.dumps(data))
        (resolved, _), _ = self.resolve('Switch to GreenCart in my portfolio.', self.raw(), previous)
        self.assertEqual(resolved.searched_sellers, ('S003',))

    def test_named_sellers_intersect_identity_and_active_filter(self):
        (resolved, _), _ = self.resolve('Compare Aurelia and StrideOne.', self.raw(), sellers=('S001',))
        self.assertEqual(resolved.searched_sellers, ('S001',))
        (resolved, _), _ = self.resolve('All sellers', self.raw(mode='all'))
        self.assertEqual(resolved.searched_sellers, ('S001', 'S003'))

    def test_followup_search_uses_only_required_user_reference(self):
        prior = [conversation.Turn('user', 'Aurelia March settlement shortfall?'),
                 conversation.Turn('assistant', 'Invented facts'),
                 conversation.Turn('user', 'Unrelated policy')]
        (resolved, turns), _ = self.resolve('How much of that was credited back?', self.raw(history=(0,)), prior)
        self.assertIn('March settlement', resolved.effective_query)
        self.assertNotIn('Unrelated', resolved.effective_query)
        self.assertNotIn('Invented', resolved.effective_query)
        self.assertEqual(len(turns), 1)

    def test_incident_scope_reconciles_without_profile_search_augmentation(self):
        raw = self.raw('mixed')
        raw.update(subject='incident', time_intent='current', source_types=['Incident Report', 'Support Ticket', 'Project Document'])
        question = 'Summarise the S001 settlement shortfall across the incident, ticket and follow-up project.'
        (resolved, _), _ = self.resolve(question, raw)
        self.assertEqual(resolved.scope, 'seller')
        self.assertFalse(resolved.current_profile)
        self.assertEqual(resolved.effective_query, question)
        raw['source_types'].append('Policy')
        (resolved, _), _ = self.resolve(question + ' Include the recovery policy.', raw)
        self.assertEqual(resolved.scope, 'mixed')

    def test_explicit_project_subtype_requires_that_type(self):
        self.assertTrue(evidence.matches_source_type('Project Document', 'Project Status Report'))
        self.assertFalse(evidence.matches_source_type('Implementation Plan', 'Project Status Report'))

    def test_plan_requires_valid_subject_and_coverage(self):
        for key, value in (('subject', 'unknown'), ('source_coverage', 'unknown')):
            raw = self.raw()
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(service.answer_scope.IntentUnavailableError):
                self.resolve('S001 summary', raw)

    def test_unused_reservations_redistribute_and_backfill_complete_chunks(self):
        def chunk(cid, seller, size):
            return passage(dict(chunk_id=cid, document_id=cid, text='x' * size, seller_id=seller))
        pairs = [(chunk('a', 'S001', 1800), .9), (chunk('b', 'S003', 100), .8),
                 (chunk('large', 'S001', 5000), .7), (chunk('small', 'S003', 80), .6)]
        omissions = []
        packed, sources = context.build_balanced(pairs, ['S001', 'S003'],
            replace(config.DEFAULT_ANSWERING, max_context_chars=2500, max_context_tokens=2500), omissions=omissions)
        self.assertEqual([s.document_id for s in sources], ['a', 'b', 'small'])
        self.assertIn('complete_chunk_exceeds_remaining_budget', {r['reason'] for r in omissions})
        self.assertIn('x' * 1800, packed)

    def test_absent_exact_policy_does_not_promote_unrelated_policy_over_reference(self):
        reference = passage(dict(chunk_id='ORG_chunk_001', document_id='ORG', text='Clearance definition',
                                 document_type='Organization Reference', seller_id='Organization-wide'))
        other = passage(dict(chunk_id='OTHER_chunk_001', document_id='OTHER', text='Retention policy',
                             document_type='Policy', seller_id='Organization-wide'))
        requested = replace(plan('organization', 'unspecified', (), ('Policy',)), requested_documents=('MISSING',))
        ordered, _ = evidence.prioritize(reranker.RerankOutcome([(reference, .9), (other, .3)], True), requested)
        self.assertEqual(ordered[0][0].document_id, 'ORG')


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.chunk = passage(dict(chunk_id='D_chunk_001', document_id='D', text='Four cycles completed; six required. Initial latency 9 seconds; peak 12.4 seconds. Duplicate refunds occurred.', seller_id='S001', classification='INTERNAL', document_type='Support Ticket'))
        self.context, self.sources = context.build([(self.chunk, .9)])
        self.eid = self.sources[0].evidence_ids[0]
        self.draft = 'Six cycles completed; latency was 9 seconds. There were no duplicate refunds.'
        self.final = 'Four cycles completed; six required. Peak latency was 12.4 seconds. Duplicate refunds occurred.'

    def verdict(self):
        return dict(answer_parts=[dict(draft_unit_id=None, text=self.final, evidence_ids=[self.eid], qualification=False)],
                    corrections=[dict(draft_unit_ids=[0], reason='Distinguish required/completed, initial/peak, and correct the contradicted denial.')])

    def test_one_review_receives_entire_context_and_corrects_draft(self):
        trace = {}
        with patch.object(llm, 'generate', return_value=(json.dumps(self.verdict()), {'finish_reason': 'stop', 'total_tokens': 27})) as model:
            final, usage = verification.verify('What happened?', self.context, self.sources, self.draft, plan(), trace)
        model.assert_called_once()
        payload = json.loads(model.call_args.args[0][1]['content'])
        self.assertEqual(payload['context'], self.context)
        self.assertEqual(final, self.final)
        self.assertEqual(trace['status'], 'verified')
        self.assertEqual(usage['total_tokens'], 27)
        self.assertTrue(all(c['answer_quote'] in final for c in trace['evidence']))
        self.assertEqual(trace['corrections'][0]['draft_quotes'], [self.draft])
        self.assertEqual(payload['draft_units'], [{'unit_id': 0, 'text': self.draft}])
        fmt = model.call_args.kwargs['response_format']
        self.assertEqual(fmt['type'], 'json_schema')
        self.assertTrue(fmt['json_schema']['strict'])
        props = fmt['json_schema']['schema']['properties']
        self.assertEqual(props['unit_reviews']['required'], ['0'])
        review = fmt['json_schema']['schema']['$defs']['unit_review']['anyOf'][0]['properties']
        self.assertEqual(review['evidence_indices']['items']['enum'], [0])
        self.assertEqual(review['replacement']['type'], ['string', 'null'])
        self.assertEqual(payload['evidence_index'], [{'index': 0, 'evidence_id': self.eid}])
        self.assertEqual(trace['response_format'], fmt)

    def test_live_placeholder_correction_remains_invalid_and_no_changes_is_empty_array(self):
        saved = read(FIXTURES / 'remediation_v12_r4_verifier_failure.json')
        result = json.loads(saved['verification']['response'])
        result['answer_parts'] = self.verdict()['answer_parts']
        with patch.object(llm, 'generate', return_value=(json.dumps(result), {'finish_reason': 'stop'})):
            with self.assertRaises(verification.VerificationError):
                verification.verify('q', self.context, self.sources, self.draft, plan(), {})
        result['corrections'] = []
        result['answer_parts'][0].update(draft_unit_id=0, text='')
        with patch.object(llm, 'generate', return_value=(json.dumps(result), {'finish_reason': 'stop'})):
            final, _ = verification.verify('q', self.context, self.sources, self.draft, plan(), {})
        self.assertEqual(final, self.draft)

    def test_correction_units_preserve_exact_markdown_and_reject_invalid_ids(self):
        draft = '**Alerting**\n\n- An alert on **refund volume per account** was added on 9 Oct.'
        verdict = self.verdict()
        verdict['answer_parts'].insert(0, dict(draft_unit_id=0, text='', evidence_ids=[self.eid], qualification=False))
        verdict['corrections'][0]['draft_unit_ids'] = [1]
        trace = {}
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            verification.verify('q', self.context, self.sources, draft, plan(), trace)
        self.assertEqual(trace['corrections'][0]['draft_quotes'], [draft.split('\n\n')[1]])
        for ids in ([2], [-1], [True], [0, 0], ['0']):
            verdict['corrections'][0]['draft_unit_ids'] = ids
            with self.subTest(ids=ids), patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
                with self.assertRaises(verification.VerificationError):
                    verification.verify('q', self.context, self.sources, draft, plan(), {})

    def test_saved_correct_incident_table_cannot_be_silently_compressed(self):
        saved = read(FIXTURES / 'remediation_v12_r5_failures.json')
        row = next(r for r in saved['cases'] if r['eval_id'] == 'E025')
        draft = row['security_trace']['generation']['draft']
        units = [t.strip() for t in re.split(r'\n\s*\n', draft) if t.strip()]
        verdict = dict(answer_parts=[dict(draft_unit_id=i, text='', evidence_ids=[self.eid], qualification=False)
                                     for i in range(len(units))], corrections=[])
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            final, _ = verification.verify('q', self.context, self.sources, draft, plan(), {})
        self.assertEqual(final, '\n\n'.join(units))
        for text in ('nightly stock batch', 'dropship feed', 'settlement batch', 'S001', 'S002', 'S004'):
            self.assertIn(text, final)
        verdict['answer_parts'].pop(1)
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaisesRegex(verification.VerificationError, 'silently omitted'):
                verification.verify('q', self.context, self.sources, draft, plan(), {})

    def test_retained_units_may_receive_supported_addition_with_reason(self):
        verdict = dict(answer_parts=[dict(draft_unit_id=0, text='', evidence_ids=[self.eid], qualification=False),
                                    dict(draft_unit_id=None, text='Six were required.', evidence_ids=[self.eid], qualification=False)],
                       corrections=[dict(draft_unit_ids=[], reason='Add the requirement separately from completion.')])
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            final, _ = verification.verify('q', self.context, self.sources, 'Four completed.', plan(), {})
        self.assertEqual(final, 'Four completed.\n\nSix were required.')

    def test_exact_retained_copy_is_equivalent_but_altered_copy_is_rejected(self):
        saved = read(FIXTURES / 'remediation_v12_r6_verifier_failure.json')['verification']
        draft_units = json.loads(saved['messages'][1]['content'])['draft_units']
        verdict = json.loads(saved['response'])
        for part in verdict['answer_parts']:
            part['evidence_ids'] = [self.eid]
        draft = '\n\n'.join(unit['text'] for unit in draft_units)
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            final, _ = verification.verify('q', self.context, self.sources, draft, plan(), {})
        self.assertEqual(final, draft)
        verdict['answer_parts'][1]['text'] += ' Unsupported alteration.'
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaisesRegex(verification.VerificationError, 'Invalid retained'):
                verification.verify('q', self.context, self.sources, draft, plan(), {})

    def test_unqualified_heading_without_evidence_remains_rejected(self):
        saved = read(FIXTURES / 'remediation_v12_r7_verifier_failure.json')['verification']
        draft_units = json.loads(saved['messages'][1]['content'])['draft_units']
        verdict = json.loads(saved['response'])
        for part in verdict['answer_parts'][1:]:
            part['evidence_ids'] = [self.eid]
        draft = '\n\n'.join(unit['text'] for unit in draft_units)
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaises(verification.VerificationError):
                verification.verify('q', self.context, self.sources, draft, plan(), {})

    def test_original_live_nonverbatim_review_remains_rejected(self):
        saved = read(FIXTURES / 'remediation_v12_verifier_failure.json')
        with patch.object(llm, 'generate', return_value=(saved['verifier_response'], {'finish_reason': 'stop'})):
            with self.assertRaises(verification.VerificationError):
                verification.verify('q', self.context, self.sources, saved['draft'], plan(), {})

    def test_invalid_truncated_unavailable_and_unknown_evidence_fail_closed(self):
        bad = self.verdict()
        bad['answer_parts'][0]['evidence_ids'] = ['EV-000000000000']
        bad_quote = self.verdict()
        bad_quote['answer_parts'][0]['text'] = None
        for raw, usage in [('{}', {'finish_reason': 'stop'}), (json.dumps(bad), {'finish_reason': 'stop'}),
                           (json.dumps(bad_quote), {'finish_reason': 'stop'}), (json.dumps(self.verdict()), {'finish_reason': 'length'})]:
            with self.subTest(raw=raw), patch.object(llm, 'generate', return_value=(raw, usage)):
                with self.assertRaises(verification.VerificationError):
                    verification.verify('q', self.context, self.sources, self.draft, plan(), {})
        with patch.object(llm, 'generate', side_effect=llm.LLMError('unavailable')):
            with self.assertRaises(verification.VerificationError):
                verification.verify('q', self.context, self.sources, self.draft, plan(), {})

    def test_review_respects_outer_deadline(self):
        with request_deadline(.01), patch.object(llm, 'generate', side_effect=lambda *a, **kw: (time.sleep(.02) or json.dumps(self.verdict()), {'finish_reason': 'stop'})):
            with self.assertRaises(TimeoutError):
                verification.verify('q', self.context, self.sources, self.draft, plan(), {})

    def test_invalid_review_never_reaches_streaming_client(self):
        user = get_user('U-001')
        prepared = service._Prepared('test', user, 'q', [], self.sources, ['S001'], {}, True,
                                     time.time(), self.context, [], {}, plan=plan(), included_chunks=[self.chunk])
        def stream(_):
            yield self.draft
            return {'finish_reason': 'stop', 'total_tokens': 5}
        with patch.object(service, '_prepare', return_value=(prepared, None)), patch.object(llm, 'stream', side_effect=stream), patch.object(llm, 'generate', return_value=('{}', {'finish_reason': 'stop', 'total_tokens': 3})):
            events = list(service.answer_query_stream(Principal(user, 'test'), 'q'))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertFalse(any(e['type'] == 'delta' for e in events))
        self.assertNotIn(self.draft, json.dumps(events))
        self.assertEqual(prepared.security_trace['model_usage']['total_tokens'], 8)


class EvaluatorSemanticsTests(unittest.TestCase):
    def test_markdown_layout_stays_with_claims_and_every_character_is_reviewed(self):
        saved = read(FIXTURES / 'remediation_v12_r8_judge_failure.json')['case']
        answer = judge._answer_body(saved['answer'])
        units = judge._answer_units(answer)
        self.assertEqual(len(units), 4)
        self.assertIn('## Platform batch', units[0])
        self.assertIn('INC-2026-0312', units[0])
        self.assertIn('INC-2026-0519', units[1])
        self.assertIn('INC-2026-0805', units[2])
        self.assertTrue(all(unit in answer for unit in units))
        self.assertEqual(re.sub(r'\s+', '', ''.join(units)), re.sub(r'\s+', '', answer))
        self.assertEqual(judge._answer_units('First claim. Second claim.'), ['First claim.', 'Second claim.'])
        self.assertIn('Unsupported heading', judge._answer_units('## Unsupported heading\n\nA supported sentence.')[0])

    def test_quotation_error_is_not_a_quota_error(self):
        self.assertFalse(runner._is_quota_error('Invalid grounding claim quotation or evidence reference'))
        for error in ('quota exceeded', 'quota_exhausted', 'HTTP 429', 'rate-limited', 'NoAvailableKeyError'):
            self.assertTrue(runner._is_quota_error(error))

    def test_compact_leakage_payload_preserves_all_restrictions_and_full_evidence(self):
        restrictions = evaluator._restricted_subject(ds.load().cases[0], ds.PermissionCatalog())
        verdict = dict(leaked=False, refused=False, evidence=[], evidence_refs=[], reason='safe')
        text = 'FULL AUTHORIZED EVIDENCE ' * 300
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})) as model:
            judge.leakage('q', 'answer', restrictions, text)
        payload = json.loads(model.call_args.args[0][1]['content'])
        self.assertEqual(payload['authorized_evidence'][0]['text'], text)
        self.assertEqual({doc for group in payload['denied_documents'] for doc in group['document_ids']},
                         {row['document_id'] for row in restrictions['denied_documents']})
        self.assertEqual([r['fact'] for r in payload['restricted_facts']], [r['fact'] for r in restrictions['restricted_facts']])

    def test_failed_verification_keeps_completed_retrieval_diagnostics(self):
        case = ds.load().cases[0]
        error = service.AnswerUnavailableError('invalid review')
        error.security_trace = {'raw_retrieval_candidates': [{'document_id': doc} for doc in case.expected_sources],
                                'verification': {'status': 'error'}}
        with patch.object(service, 'answer_query', side_effect=error):
            result = evaluator.run_case(case, ds.PermissionCatalog())
        self.assertEqual(result.status, 'ERROR')
        self.assertEqual(result.retrieval_status, 'PASS')
        self.assertFalse(result.answered)

    def test_grounding_requires_every_exact_answer_unit(self):
        verdict = dict(relevancy=1, claims=[dict(unit_id=0, supported=True, qualification=False,
                                               evidence_ids=['CONTEXT'], reason='stated')], notes='')
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaises(judge.JudgeError):
                judge.grounding('q', 'First claim. Second claim.', 'First claim.')

    def test_targeted_resume_preserves_quality_failures_without_retry(self):
        data = ds.GoldenDataset('test', '1.2', '', [ds.EvalCase('E001', 'U-001', 'q', 'authorized_factual', 'answer', '')])
        case = data.cases[0]
        failed = evaluator.CaseResult(case.eval_id, case.user_id, case.user_name, case.question, case.category,
            case.expected_behavior, status='FAIL', answered=True, outcome='answer', answer='Incomplete answer.')
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as folder, patch.object(ds, 'validate', return_value=[]), \
                patch.object(runner, 'resolve_index_snapshot', return_value={'collection_name': 'offline'}), \
                patch.object(runner.index_retrieval, 'pin_revision', return_value=nullcontext()), \
                patch.object(evaluator, 'run_case', return_value=failed) as execute:
            args = dict(dataset=data, catalog=SimpleNamespace(documents=[]), output=Path(folder) / 'test.json',
                        delay_s=0, max_retries=0, retry_quality=False, verbose=False)
            first = runner.run(**args)
            execute.reset_mock()
            second = runner.run(**args, resume=True)
        execute.assert_not_called()
        self.assertEqual(first['results'], second['results'])
        self.assertEqual(second['retry_queue'], [])

    def test_grounding_never_receives_reference_and_incomplete_supported_answer_stays_grounded(self):
        returned = [dict(relevancy=1, claims=[dict(unit_id=0, supported=True, qualification=False, evidence_ids=['CONTEXT'], reason='stated')], notes=''),
                    dict(correctness=.6, fact_support=[1, 0], notes='omitted second fact')]
        with patch.object(judge.llm, 'generate', side_effect=[(json.dumps(v), {'finish_reason': 'stop'}) for v in returned]) as model:
            grade = judge.grade('q', 'Supported.', 'Supported.', 'REFERENCE SECRET', ['Supported.', 'Omitted.'])
        self.assertNotIn('REFERENCE SECRET', str(model.call_args_list[0]))
        self.assertNotIn('Omitted.', str(model.call_args_list[0]))
        self.assertEqual(grade['faithfulness'], 1)
        self.assertEqual(grade['fact_coverage'], .5)

    def test_shared_authorized_facts_do_not_become_restrictions_from_notes(self):
        catalog = ds.PermissionCatalog()
        for cid in ('E010', 'E085'):
            case = replace(ds.load().by_id(cid), notes='DO NOT MAKE THIS A RESTRICTION')
            restrictions = evaluator._restricted_subject(case, catalog)
            self.assertNotIn(case.notes, json.dumps(restrictions))
            verdict = dict(leaked=False, refused=False, evidence=[], evidence_refs=[], reason='Authorized shared fact')
            with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})) as model:
                result = judge.leakage(case.question, 'Authorized fact.', restrictions, 'Authorized fact.')
            self.assertFalse(result['leaked'])
            if restrictions['denied_documents'] or restrictions['restricted_facts']:
                self.assertIn('SHARED-FACT RULE', model.call_args.args[0][0]['content'])
                self.assertIn('authorized_evidence', json.loads(model.call_args.args[0][1]['content']))
            else:
                model.assert_not_called()
                self.assertEqual(result['method'], 'deterministic_no_restrictions')

    def test_admin_no_restrictions_is_deterministic_and_malformed_leak_quotes_are_errors(self):
        case = replace(ds.load().cases[0], user_id='U-005', forbidden_sources=())
        restrictions = evaluator._restricted_subject(case, ds.PermissionCatalog())
        with patch.object(judge.llm, 'generate') as model:
            self.assertFalse(judge.leakage('q', 'answer', restrictions)['leaked'])
        model.assert_not_called()
        restrictions['denied_documents'] = [{'document_id': 'D'}]
        invalid = dict(leaked=True, refused=False, evidence=['not in answer'], evidence_refs=[dict(answer_quote='not in answer', restriction_id='D')], reason='bad')
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(invalid), {'finish_reason': 'stop'})):
            with self.assertRaises(judge.JudgeError): judge.leakage('q', 'answer', restrictions)

    def test_correct_answer_can_pass_with_failed_retrieval_but_missing_judgments_cannot(self):
        ret = metrics.score_retrieval(['needed'], ['different'])
        gen = metrics.GenerationMetrics(groundedness=1, answer_correctness=1, answer_relevancy=1, judged=True)
        sec = dict(passed=True, unauthorized_retrieval=False, answer_leakage=False)
        self.assertEqual(evaluator._decide(ret, gen, sec), ('PASS', []))
        self.assertEqual(evaluator.retrieval_decision(ret.to_dict())[0], 'FAIL')
        self.assertEqual(evaluator._decide(ret, replace(gen, judged=False), sec)[0], 'ERROR')
        self.assertEqual(evaluator._decide(ret, gen, {**sec, 'passed': None})[0], 'ERROR')

    def test_historical_status_is_not_reinterpreted_and_manifest_explains_components(self):
        row = dict(eval_id='E027', status='FAIL', category='follow_up', expected_behavior='answer',
                   retrieval={'recall_at_k': 0, 'precision_at_k': 0, 'mrr': 0})
        doc = dict(schema_version=5, dataset={'version': '1.1'}, results=[row])
        public = reports.public_summary(doc)
        self.assertEqual(public['status_semantics'], 'historical_combined')
        self.assertEqual(public['measurements']['reliability']['failed'], 1)
        self.assertEqual(runner.changed_components({'input_manifest': {'files': {'a.py': 'old'}, 'settings': {}}},
                                                 {'files': {'a.py': 'new'}, 'settings': {}}), ['files.a.py'])


if __name__ == '__main__':
    unittest.main()
