"""Regressions for the real 6 November verifier and grounding errors."""
import copy
import hashlib
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from answering.src import llm, verification
from evaluation import judge
from evaluation.artifacts import matches_sha256

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'evaluation/fixtures/execution_errors_20261106.json'


class ExecutionErrorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = json.loads(FIXTURE.read_text(encoding='utf-8'))
        assert matches_sha256((ROOT / cls.saved['report']).read_bytes(), cls.saved['report_sha256'])
        cls.cases = {r['eval_id']: r for r in cls.saved['cases']}

    def fixture_review(self, cid):
        saved = self.cases[cid]['verification']
        payload = json.loads(saved['messages'][1]['content'])
        units = payload['draft_units']
        ids = sorted(set(re.findall(r'(?m)^Evidence \[(EV-[0-9a-f]{12})\]', payload['context'])))
        self.assertTrue(ids)
        sources = [SimpleNamespace(evidence_ids=ids)]
        verdict = dict(unit_reviews={str(u['unit_id']): dict(replacement=None,
            evidence_indices=list(range(len(ids))), qualification=False, reason='') for u in units}, additions=[])
        return payload, sources, verdict

    def verify(self, payload, sources, verdict, trace=None):
        draft = '\n\n'.join(u['text'] for u in payload['draft_units'])
        with patch.object(llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})) as model:
            final, _ = verification.verify(payload['question'], payload['context'], sources, draft, None,
                                           trace if trace is not None else {})
        model.assert_called_once()
        return final

    def test_e026_table_is_copied_exactly_without_model_retyping_it(self):
        payload, sources, verdict = self.fixture_review('E026')
        expected = '\n\n'.join(u['text'] for u in payload['draft_units'])
        verdict['unit_reviews'] = dict(reversed(list(verdict['unit_reviews'].items())))
        self.assertEqual(self.verify(payload, sources, verdict), expected)
        self.assertNotIn('text', verdict['unit_reviews']['1'])

    def test_e048_qualification_does_not_conflict_with_retaining_text(self):
        payload, sources, verdict = self.fixture_review('E048')
        verdict['unit_reviews']['0'].update(qualification=True, evidence_indices=[],
                                          reason='The requested account overview is absent from the supplied sources.')
        trace = {}
        final = self.verify(payload, sources, verdict, trace)
        self.assertIn(payload['draft_units'][0]['text'], final)
        self.assertEqual(trace['corrections'], [])
        self.assertTrue(trace['evidence'][0]['qualification'])

    def test_e051_uses_small_evidence_indices_and_resolves_original_ids(self):
        payload, sources, verdict = self.fixture_review('E051')
        trace = {}
        self.verify(payload, sources, verdict, trace)
        self.assertEqual(trace['evidence'][0]['evidence_ids'], sources[0].evidence_ids)
        self.assertIn('evidence_index', json.loads(trace['messages'][1]['content']))
        self.assertNotIn('EV-', json.dumps(trace['response_format']))

    def test_missing_extra_units_and_unknown_evidence_still_fail_closed(self):
        payload, sources, original = self.fixture_review('E026')
        bad = copy.deepcopy(original)
        del bad['unit_reviews']['1']
        variants = [bad]
        bad = copy.deepcopy(original)
        bad['unit_reviews']['99'] = copy.deepcopy(bad['unit_reviews']['0'])
        variants.append(bad)
        for index in (-1, True, '0', len(sources[0].evidence_ids)):
            bad = copy.deepcopy(original)
            bad['unit_reviews']['0']['evidence_indices'] = [index]
            variants.append(bad)
        bad = copy.deepcopy(original)
        bad['unit_reviews']['0']['evidence_indices'] = []
        variants.append(bad)
        for bad in variants:
            with self.subTest(bad=bad), self.assertRaises(verification.VerificationError):
                self.verify(payload, sources, bad)

    def test_replacements_deletions_and_additions_require_reasons(self):
        payload, sources, original = self.fixture_review('E026')
        for replacement in ('', 'Changed text.'):
            bad = copy.deepcopy(original)
            bad['unit_reviews']['0']['replacement'] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(verification.VerificationError):
                self.verify(payload, sources, bad)
        bad = copy.deepcopy(original)
        bad['additions'] = [dict(replacement='New fact.', evidence_indices=[0], qualification=False, reason='')]
        with self.assertRaises(verification.VerificationError):
            self.verify(payload, sources, bad)
        valid = copy.deepcopy(original)
        valid['unit_reviews']['0'].update(replacement='Documented duplication cases.', reason='Correct the heading.')
        valid['unit_reviews']['2'].update(replacement='', reason='Remove an unsupported exhaustive claim.')
        valid['additions'] = [dict(replacement='Only the supplied records are summarized.',
                                  evidence_indices=[], qualification=True, reason='State the scope limitation.')]
        trace = {}
        final = self.verify(payload, sources, valid, trace)
        self.assertTrue(final.startswith('Documented duplication cases.'))
        self.assertTrue(final.endswith('Only the supplied records are summarized.'))
        self.assertEqual([c['draft_unit_ids'] for c in trace['corrections']], [[0], [2], []])
        self.assertEqual(trace['corrections'][1]['draft_quotes'], [payload['draft_units'][2]['text']])

    def test_e054_numbered_list_labels_stay_with_real_claims(self):
        answer = judge._answer_body(self.cases['E054']['answer'])
        units = judge._answer_units(answer)
        self.assertLess(len(units), 22)
        self.assertFalse(any(re.fullmatch(r'\d+[.)]', u) for u in units))
        self.assertIn('### Next Steps', units[0])
        self.assertIn('Review the Post', units[0])
        self.assertTrue(all(u in answer for u in units))
        self.assertEqual(re.sub(r'\s+', '', ''.join(units)), re.sub(r'\s+', '', answer))
        self.assertEqual(judge._answer_units('1. First claim. Second claim.'), ['1. First claim.', 'Second claim.'])

    def test_grounding_qualified_advice_is_distinct_from_unsupported_facts(self):
        verdict = dict(relevancy=1, notes='', claims=[
            dict(unit_id=0, kind='qualified', evidence_ids=[], reason='Explicit recommendation.'),
            dict(unit_id=1, kind='unsupported', evidence_ids=[], reason='Outcome is not established.')])
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            result = judge.grounding('q', 'Consider reviewing the incident. This will prevent every outage.', 'Incident record.')
        self.assertEqual(result['faithfulness'], .5)
        self.assertEqual(result['unsupported_claims'], ['This will prevent every outage.'])
        verdict['claims'][0]['kind'] = 'supported'
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaises(judge.JudgeError):
                judge.grounding('q', 'A factual claim. Another claim.', 'Evidence.')

    def test_grounding_requires_one_key_for_every_answer_unit(self):
        verdict = dict(relevancy=1, notes='', claims={
            '0': dict(kind='supported', evidence_ids=['CONTEXT'], reason='Stated.'),
            '1': dict(kind='unsupported', evidence_ids=[], reason='Not stated.')})
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            result = judge.grounding('q', 'One claim. Second claim.', 'One claim.')
        self.assertEqual(result['faithfulness'], .5)
        schema = judge._grounding_format(2, {'CONTEXT'})['json_schema']['schema']
        self.assertEqual(schema['properties']['claims']['required'], ['0', '1'])
        del verdict['claims']['1']
        with patch.object(judge.llm, 'generate', return_value=(json.dumps(verdict), {'finish_reason': 'stop'})):
            with self.assertRaisesRegex(judge.JudgeError, 'omitted'):
                judge.grounding('q', 'One claim. Second claim.', 'One claim.')


if __name__ == '__main__':
    unittest.main()
