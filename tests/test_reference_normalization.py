import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import normalize_benchmark_references as n


class Smoke(unittest.TestCase):
    def test_versioned_measurements(self):
        annotations = n.b.load(n.DEFAULT)
        rows = [dict(source=e['source'], id=e['id'], original_id=e['original_id'],
                     source_revision=e['source_revision'], reference=e['raw'])
                for e in annotations['entries']]
        annotations['expected_samples'] = len(rows)
        result = n.transform(rows, annotations)
        self.assertEqual(sum(len(e['spans']) for e in annotations['entries']), 7)
        for row, e in zip(result, annotations['entries']):
            for old, new, rule in e['spans']:
                self.assertNotIn(old, row['reference'])
                self.assertIn(new, row['reference'])
        self.assertIn('29 tuổi', result[-1]['reference'])

    def fixture(self):
        return [dict(source='google/fleurs', id='v', original_id=1,
                     source_revision='a'*40, split='validation', reference='bơi 100m và 200m')], dict(
            version='fleurs60-spoken-v1', expected_samples=1, entries=[dict(
                source='google/fleurs', id='v', original_id=1, source_revision='a'*40,
                raw='bơi 100m và 200m', current='bơi 100m và 200m',
                new='bơi một trăm mét và hai trăm mét', audio_validated=False,
                replacements=[dict(start=4,end=8,old='100m',new='một trăm mét',rule='explicit-distance'),
                              dict(start=12,end=16,old='200m',new='hai trăm mét',rule='explicit-distance')])])

    def test_conversion_and_ambiguous_preservation(self):
        rows, annotations = self.fixture()
        result = n.transform(rows, annotations)
        self.assertEqual(result[0]['reference'], annotations['entries'][0]['new'])
        self.assertEqual(result[0]['reference_raw'], rows[0]['reference'])
        self.assertEqual(n.transform(result, annotations), result)
        rows[0]['reference'] = 'năm 2002 mã 100m giá 50 đô'
        annotations['entries'] = []
        self.assertEqual(n.transform(rows, annotations)[0]['reference'], rows[0]['reference'])

    def test_invalid_raw_span_and_revision(self):
        rows, annotations = self.fixture()
        for field, value in [('raw','wrong'), ('source_revision','b'*40), ('new','wrong')]:
            bad = copy.deepcopy(annotations)
            bad['entries'][0][field] = value
            with self.assertRaises(ValueError):
                n.transform(rows, bad)

    def test_partitions_backups_and_rollback(self):
        rows, annotations = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'splits').mkdir()
            paths = [root/'manifest.json', root/'splits/validation.json', root/'splits/test.json']
            for p, value in zip(paths, [rows, rows, []]):
                p.write_text(json.dumps(value))
            before = [p.read_bytes() for p in paths]
            replace = n.os.replace
            calls = 0
            def fail_once(src, dst):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('injected write failure')
                return replace(src, dst)
            with patch.object(n.os, 'replace', side_effect=fail_once):
                with self.assertRaises(OSError):
                    n.apply(root/'manifest.json', annotations, validate=lambda r: None)
            self.assertEqual([p.read_bytes() for p in paths], before)
            n.apply(paths[0], annotations, validate=lambda r: None)
            result = json.loads(paths[0].read_text())
            self.assertEqual(json.loads(paths[1].read_text()), result)
            self.assertEqual(json.loads(paths[2].read_text()), [])
            after = [p.read_bytes() for p in paths]
            n.apply(paths[0], annotations, validate=lambda r: None)
            self.assertEqual([p.read_bytes() for p in paths], after)


if __name__ == '__main__':
    unittest.main()
