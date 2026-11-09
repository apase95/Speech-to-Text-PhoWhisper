import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from tools import normalize_benchmark_references as n
from tools import reevaluate_numeric_metrics as m


class MetricSmoke(unittest.TestCase):
    def test_grammar(self):
        for raw, spoken in [
            ('bơi 100m và 200m', 'bơi một trăm mét và hai trăm mét'),
            ('ngày 6 tháng 10 năm 1789', 'ngày sáu tháng mười năm một nghìn bảy trăm tám mươi chín'),
            ('năm 2002', 'năm hai nghìn không trăm linh hai'),
            ('108 đĩa và 56 món', 'một trăm linh tám đĩa và năm mươi sáu món'),
            ('thứ 1.000', 'thứ một nghìn'),
            ('cao 1,5 m!', 'cao một phẩy năm mét!'),
            ('29 tuổi', 'hai mươi chín tuổi')]:
            self.assertEqual(m.verbalize(raw)[0], spoken)
            self.assertEqual(m.verbalize(spoken)[0], spoken)
            self.assertEqual(m.normalized_pair(raw, spoken, True)[0],
                             m.normalized_pair(raw, spoken, True)[1])
        for raw in ('mã A100m', 'louis xvi', '6/10/1789', 'ngày 31 tháng 2 năm 2002',
                    'mã 100m', '01 tuổi', '1.2.3 m'):
            self.assertEqual(m.verbalize(raw)[0], raw.lower())
        self.assertEqual(m.verbalize('thứ 1 và thứ 4')[0], 'thứ nhất và thứ tư')
        self.assertEqual(m.verbalize('70km/h...')[0], 'bảy mươi ki lô mét trên giờ...')
        self.assertEqual(m.verbalize('năm 400.000')[0], 'năm 400.000')
        self.assertEqual(m.verbalize('oravec 2002')[1], ['2002'])

    def test_metrics_independent_and_edges(self):
        result = m.measure(['a b', 'c'], ['a x', 'c d'])
        self.assertEqual((result['words'], result['characters'], result['errors']), (3, 4, 2))
        self.assertEqual((result['S'], result['D'], result['I']), (1, 0, 1))
        self.assertEqual(result['independent']['errors'], 2)
        self.assertEqual(m.measure([''], ['a'])['I'], 1)
        self.assertEqual(m.measure(['a'], [''])['D'], 1)

    def test_original_hash_and_swapped_reference(self):
        rows = [dict(source='s',id='a',split='test',reference='one'),
                dict(source='s',id='b',split='test',reference='two')]
        active = [dict(r,reference_raw=r['reference'],reference_convention='fleurs60-spoken-v2') for r in rows]
        self.assertEqual(m.original_rows(active), rows)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'original.json'
            n.b.save(p, m.original_rows(active))
            result = dict(manifest_sha256=n.b.file_hash(p),records=copy.deepcopy(rows))
            m.verify_original_cache(p,result)
            result['manifest_sha256'] = '0'*64
            with self.assertRaises(ValueError):
                m.verify_original_cache(p,result)
            result['manifest_sha256'] = n.b.file_hash(p)
            result['records'][0]['reference'] = 'two'
            with self.assertRaises(ValueError):
                m.verify_original_cache(p,result)

    def test_identity_swap_and_fixed_choice(self):
        rows = [dict(source='s', id='a', split='test', reference='one'),
                dict(source='s', id='b', split='test', reference='two')]
        records = copy.deepcopy(rows)
        m.identity_join(rows, records, references=True)
        records[0]['reference'], records[1]['reference'] = 'two', 'one'
        with self.assertRaises(ValueError):
            m.identity_join(rows, records, references=True)
        cs = [dict(id='a', text='100m', avg_logprob=-1., lm=-1.),
              dict(id='b', text='một trăm mét', avg_logprob=-2., lm=-1.)]
        self.assertEqual(m.fixed_choice(cs, .4, 0)['id'], 'a')
        cs[0]['lm'] = float('nan')
        with self.assertRaises(ValueError):
            m.fixed_choice(cs, .4, 0)

    def test_fixed_prediction_not_reselected_for_new_reference(self):
        candidates = [dict(id='original',text='100m',avg_logprob=-1.,lm=-1.),
                      dict(id='new-reference-match',text='một trăm mét',avg_logprob=-2.,lm=-1.)]
        chosen = m.fixed_choice(candidates,.4,0)
        self.assertEqual(chosen['id'],'original')
        with patch.object(m,'fixed_choice',side_effect=AssertionError('metric must not select')):
            a = m.measure(*[[s] for s in m.normalized_pair('100m',chosen['text'])])
            c = m.measure(*[[s] for s in m.normalized_pair('một trăm mét',chosen['text'])])
            d = m.measure(*[[s] for s in m.normalized_pair('100m',chosen['text'],True)])
        self.assertEqual((a['errors'],c['errors'],d['errors']),(0,3,0))
        self.assertEqual(chosen,candidates[0])


class V2Smoke(unittest.TestCase):
    def test_v2_migration_and_idempotence(self):
        a = n.b.load(n.DEFAULT_V2)
        entries = n.expanded_annotations(a)['entries']
        rows = [dict(source=e['source'], id=e['id'], original_id=e['original_id'],
                     source_revision=e['source_revision'], reference=e['raw']) for e in entries]
        a['expected_samples'] = len(rows)
        v1 = n.b.load(n.DEFAULT)
        v1['expected_samples'] = len(rows)
        before = n.transform(rows, v1)
        after = n.transform(before, a)
        self.assertEqual(n.transform(after, a), after)
        self.assertFalse(any(any(c.isdigit() for c in r['reference']) for r in after))
        self.assertIn('louis xvi', next(r['reference'] for r in after if r['id']=='validation:237'))
        self.assertIn('thứ một nghìn', next(r['reference'] for r in after if r['id']=='validation:260'))

    def test_migration_refuses_tampering(self):
        a = n.b.load(n.DEFAULT_V2)
        entries = n.expanded_annotations(a)['entries']
        rows = [dict(source=e['source'], id=e['id'], original_id=e['original_id'],
                     source_revision=e['source_revision'], reference=e['raw']) for e in entries]
        v1 = n.b.load(n.DEFAULT)
        v1['expected_samples'] = len(rows)
        rows = n.transform(rows, v1)
        a['expected_samples'] = len(rows)
        n.transform(rows, a)  # Control: this exact fixture must otherwise pass.
        rows[0]['reference'] += ' unrelated edit'
        with self.assertRaises(ValueError):
            n.transform(rows, a)

    def test_annotation_base_and_duplicate_guards(self):
        a = n.b.load(n.DEFAULT_V2)
        bad = copy.deepcopy(a)
        bad['base_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            n.expanded_annotations(bad)
        a['entries'].append(copy.deepcopy(a['entries'][0]))
        with self.assertRaises(ValueError):
            n.expanded_annotations(a)


if __name__ == '__main__':
    unittest.main()
