import unittest
import tempfile
import os
import copy
from unittest.mock import patch
from pathlib import Path
from tools import gpt2_optimization as o


class Smoke(unittest.TestCase):
    def setUp(self):
        load = o.b.load
        def portable_load(path):
            if self._testMethodName != 'test_saved_metrics_independent' and (Path(path).is_relative_to(o.ROOT/'data') or Path(path).is_relative_to(o.ROOT/'outputs')):
                raise AssertionError('default unit test read ignored benchmark artifacts')
            return load(path)
        guard = patch.object(o.b, 'load', side_effect=portable_load)
        guard.start()
        self.addCleanup(guard.stop)
        self.historical = tempfile.TemporaryDirectory()
        self.addCleanup(self.historical.cleanup)
        root = Path(self.historical.name)
        self.manifest = root/'validation.json'
        rows = [dict(source='google/fleurs', id='synthetic', split='validation', reference='xin chào')]
        o.b.save(self.manifest, rows)
        protocol = dict(acoustic=dict(checkpoint_sha256='a'*64,
            decoding=dict(beam_size=5, language='vi', task='transcribe', temperature=0,
                          without_timestamps=True, fp16=False,
                          ranker='MaximumLikelihoodRanker:length', mel='standard-fp32-v1'),
            runtime=dict(python='3.14.8', torch='2.14.1', **{'openai-whisper':'20250625', 'numpy':'2.5.3'})),
            lm=dict(model_id='synthetic/gpt2', revision=o.REV, tokenizer_id='synthetic/gpt2',
                    tokenizer_revision=o.REV, scoring='normalized-mean-v3-model-prefix-or-special-excluded-pll',
                    prefix_id=0, prefix_policy='model-config-bos', segmentation='none',
                    vocabulary=dict(embedding_size=2, config_size=2, tokenizer_size=2, tokenizer_sha256='b'*64),
                    runtime=dict(python='3.14.8', torch='2.14.1', transformers='4.57.6',
                                 tokenizers='0.22.2', pyvi='0.1.1', numpy='2.5.3')))
        records = [dict(source='google/fleurs', id='synthetic', split='validation', candidates=[
            dict(id='a', text='xin chào', avg_logprob=-.1),
            dict(id='b', text='xin bạn', avg_logprob=-.2)])]
        self.nbest, self.scored = root/'nbest.json', root/'scores.json'
        o.b.artifact(self.nbest, 'nbest', o.b.file_hash(self.manifest), records, protocol=protocol['acoustic'])
        records = copy.deepcopy(records)
        for c in records[0]['candidates']:
            c['lm'] = -.5
        o.b.artifact(self.scored, 'scored', o.b.file_hash(self.manifest), records,
                     protocol=protocol, lm='gpt2', resolved_revision=o.REV,
                     nbest_sha256=o.b.file_hash(self.nbest))

    def test_validation_selection_and_frozen_protocol(self):
        with self.assertRaises(ValueError):
            o.select_validation([dict(split='test')], [])
        grid = [dict(errors=2,chars=3,alpha=.4,beta=.01),
                dict(errors=2,chars=3,alpha=.3,beta=0)]
        self.assertEqual(o.select_validation([dict(split='validation')],grid),grid[1])
        f = o.seal(dict(protocol={'beam':5},manifest_sha256='validation'))
        o.verify_test(f, {'protocol':{'beam':5}}, 'test')
        for protocol, mh in [({'beam':10},'test'), ({'beam':5},'validation')]:
            with self.assertRaises(ValueError):
                o.verify_test(f, {'protocol':protocol}, mh)

    def test_winner_before_test(self):
        from unittest.mock import patch
        validation = o.b.load(self.scored)
        mapping = o.b.join(self.manifest,validation)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            def checked(manifest, nbest, scored):
                beam = 5 if 'beam' not in str(scored.parent.name) else int(scored.parent.name.split('-')[0][4:])
                d = copy.deepcopy(validation)
                d['protocol']['acoustic']['decoding']['beam_size'] = beam
                if manifest.name == 'test.json':
                    self.assertTrue((out/'winner.json').exists())
                    for size in (5,10,15):
                        self.assertTrue((out/f'beam{size}-freeze.json').exists())
                    raise StopIteration('test gate reached')
                return d,mapping
            with patch.object(o,'OUT',out), patch.object(o,'checked',side_effect=checked), \
                 patch.object(o.b,'decode'), patch.object(o.b,'score'), \
                 patch.object(o.b,'file_hash',return_value='validation'), \
                 patch.object(o,'diagnostic',return_value={}):
                with self.assertRaisesRegex(StopIteration,'test gate'):
                    o.main()

    @unittest.skipUnless(os.environ.get('FLEURS60_PERSISTED_AUDIT') == '1',
                         'opt-in historical artifact audit; see benchmark documentation')
    def test_saved_metrics_independent(self):
        # Independent jiwer alignment, not the implementation's backtrace.
        from jiwer import process_words, process_characters
        for split,beam in [('validation',5),('validation',10),('validation',15),('test',5),('test',15)]:
            d = o.b.load(o.OUT/f'beam{beam}-{split}.json')
            refs = [r['reference'] for r in d['records']]
            self.assertEqual(sum(len(r.split()) for r in refs),1267 if split=='validation' else 1353)
            self.assertEqual(sum(map(len,refs)),5750 if split=='validation' else 6085)
            for name, totals in d['summary'].items():
                hyps = [r['metrics'][name]['hypothesis'] for r in d['records']]
                w = process_words(refs,hyps)
                c = process_characters(refs,hyps)
                self.assertEqual(w.substitutions+w.deletions+w.insertions,totals['errors'])
                self.assertEqual(c.substitutions+c.deletions+c.insertions,totals['chars'])
                self.assertAlmostEqual(w.wer,totals['wer'])
                self.assertAlmostEqual(c.cer,totals['cer'])
                for k in ('S','D','I'):
                    self.assertEqual(sum(r['metrics'][name][k] for r in d['records']),totals[k])

    def test_alignment(self):
        self.assertEqual(o.edits(['a', 'b'], ['a', 'c']), dict(S=1, D=0, I=0))
        self.assertEqual(o.edits([], ['a']), dict(S=0, D=0, I=1))
        self.assertEqual(o.edits(['a'], []), dict(S=0, D=1, I=0))

    def test_synthetic_diagnostic_totals(self):
        data = o.b.load(self.scored)
        mapping = o.b.join(self.manifest, data)
        result = o.diagnostic(data, mapping, .4, 0)
        self.assertEqual(result['summary']['beam'],
                         dict(S=0, D=0, I=0, errors=0, chars=0, wer=0, cer=0))
        self.assertEqual(result['records'][0]['words'], 2)
        self.assertEqual(result['records'][0]['characters'], len('xin chào'))
        mapping['google/fleurs', 'synthetic']['reference'] = 'xin bạn'
        result = o.diagnostic(data, mapping, .4, 0)
        self.assertEqual(result['summary']['beam']['S'], 1)
        self.assertEqual(result['summary']['beam']['errors'], 1)
        self.assertEqual(result['summary']['beam']['wer'], .5)
        self.assertEqual(result['summary']['word_oracle']['errors'], 0)

    def test_freeze_integrity(self):
        f = o.seal(dict(alpha=.4, beta=0))
        o.verify(f)
        f['alpha'] = .5
        with self.assertRaises(ValueError):
            o.verify(f)

    def test_synthetic_cached_source_guards(self):
        manifest = self.manifest
        nbest = self.nbest
        scored = self.scored
        o.checked(manifest, nbest, scored)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'scores.json'
            for mutation in ('missing', 'forged', 'protocol', 'source'):
                d = o.b.load(scored)
                if mutation == 'missing':
                    del d['protocol']
                elif mutation == 'forged':
                    d['records'][0]['candidates'][0]['lm'] += 1
                elif mutation == 'protocol':
                    d['protocol']['acoustic']['decoding']['beam_size'] = 10
                else:
                    d['nbest_sha256'] = '0'*64
                if mutation in ('protocol','source'):
                    records = d.pop('records')
                    kind = d.pop('kind')
                    mh = d.pop('manifest_sha256')
                    d.pop('configuration_sha256')
                    o.b.artifact(target,kind,mh,records,**d)
                else:
                    o.b.save(target,d)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    o.checked(manifest,nbest,target)

if __name__ == '__main__':
    unittest.main()
