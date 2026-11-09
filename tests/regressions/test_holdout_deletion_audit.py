import copy
import unittest
from tools.audit_holdout_deletions import concentration, verify_tsv, verify_cache_ids, verify_bindings, physical_tsv_audit


class DeletionAuditSmoke(unittest.TestCase):
    def test_concentration(self):
        rows=[dict(id='b',metrics=dict(D=1)),dict(id='a',metrics=dict(D=8)),dict(id='c',metrics=dict(D=1))]
        out=concentration(rows)
        self.assertEqual(out['top1_share'],.8)
        self.assertEqual(out['top3_share'],1)
        self.assertEqual(out['top_ids'],['a','b','c'])
        self.assertEqual(concentration([dict(id='z',metrics=dict(D=0))])['top1_share'],0)

    def test_tsv_mapping_and_swaps(self):
        rows=[dict(id='test:0',original_id=5,archive_member='test/a.wav',reference='normalized')]
        self.assertEqual(verify_tsv(rows,'5\ta.wav\tRaw Text!\tnormalized\tn o r m |\t16000\tMALE\n')[0]['transcription_column'],3)
        for text in ('5\tb.wav\tRaw Text!\tnormalized\tn o r m |\t16000\tMALE\n',
                     '5\ta.wav\tnormalized\twrong\tw r o n g |\t16000\tMALE\n'):
            with self.assertRaises(ValueError):
                verify_tsv(rows,text)

    def test_candidate_token_identity(self):
        import json
        from tools import long_audio_benchmark as b
        r=dict(source='s',id='test:0',candidates=[dict(tokens=[1,2],text='x')])
        r['candidates'][0]['id']=b.digest(('stest:00'+json.dumps([1,2])).encode())
        verify_cache_ids([r])
        forged=copy.deepcopy(r); forged['candidates'][0]['tokens']=[2,1]
        with self.assertRaises(ValueError):
            verify_cache_ids([forged])

    def test_strict_result_binding(self):
        import tempfile
        from pathlib import Path
        from tools import long_audio_benchmark as b, gpt2_optimization as o
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'; b.save(p,[])
            result=o.seal(dict(manifest_sha256=b.file_hash(p)))
            verify_bindings(result,dict(manifest_sha256=p))
            b.save(p,[dict(id='swapped')])
            with self.assertRaises(ValueError):
                verify_bindings(result,dict(manifest_sha256=p))
            with self.assertRaises(ValueError):
                verify_bindings(dict(result,fingerprint='forged'),dict(manifest_sha256=p))

    def test_quote_contamination(self):
        text='5\ta.wav\t"Raw sentence."\t"raw sentence.\t" r a w | s e n t e n c e . |\t16000\tMALE\n'
        import csv, io
        broken=list(csv.reader(io.StringIO(text),delimiter='\t'))[0][3]
        rows=[dict(id='test:0',original_id=5,archive_member='test/a.wav',reference=broken,duration=1)]
        evidence=physical_tsv_audit(rows,text)
        self.assertEqual(evidence[0]['stored_normalized_words'],13)
        self.assertEqual(evidence[0]['physical_transcription_normalized_words'],2)
        self.assertTrue(evidence[0]['reference_contaminated'])
        with self.assertRaises(ValueError):
            physical_tsv_audit(rows,text.replace('a.wav','other.wav'))


if __name__=='__main__':
    unittest.main()
