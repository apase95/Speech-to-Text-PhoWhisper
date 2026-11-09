import copy
import unittest
from unittest.mock import patch
from tools import human_reference_review as h


class Smoke(unittest.TestCase):
    def test_literal_spans(self):
        for spans in h.CORRECTIONS.values():
            old = ' / '.join(s[0] for s in spans)
            expected = ' / '.join(s[1] for s in spans)
            self.assertEqual(h.replace_spans(old, spans), expected)

    def test_missing_multiple_and_source_guards(self):
        for text in ('absent', 'old old'):
            with self.assertRaises(ValueError):
                h.replace_spans(text, [('old', 'new')])
        row = dict(source='google/fleurs', id='test:266', original_id=1,
                   source_revision='a', sha256='b', reference='old',
                   reference_raw='2002', reference_convention='fleurs60-spoken-v2')
        annotation = dict(version=h.VERSION, expected_samples=1, entries=[dict(
            source='google/fleurs', id='test:266', original_id=1,
            source_revision='a', audio_sha256='b', raw='2002', old_v2='old',
            reviewed='new', spans=[['old', 'new']])])
        result = h.transform([row], annotation)
        self.assertEqual(result[0]['reference_v2'], 'old')
        self.assertEqual(h.transform(result, annotation), result)
        bad = copy.deepcopy(row)
        bad['original_id'] = 2
        with self.assertRaises(ValueError):
            h.transform([bad], annotation)

    def test_metric_never_reselects(self):
        old = [dict(source='s', id=split+':1', split=split,
                    reference='old words', reference_raw='raw',
                    reference_convention='fleurs60-spoken-v2') for split in ('test', 'validation')]
        reviewed = [dict(r, reference='fixed words') for r in old]
        fixed = dict(records=[dict(source=r['source'], id=r['id'], split=r['split'],
            hypotheses={'gpt2_tuned_beam15': dict(candidate_id='immutable', text='fixed words')}) for r in old])
        with patch.object(h.m, 'fixed_choice', side_effect=AssertionError('reselection')), \
             patch.object(h.o, 'choose', side_effect=AssertionError('reselection')), \
             patch.object(h.b, 'file_hash', return_value='synthetic'):
            result = h.evaluate(old, reviewed, fixed)
        self.assertEqual({r['candidate_id'] for r in result['records']}, {'immutable'})
        self.assertTrue(all(r['metrics']['errors']==0 for r in result['records'] if r['convention']=='human-v3'))


if __name__ == '__main__':
    unittest.main()
