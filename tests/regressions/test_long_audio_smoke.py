import importlib.util
import pathlib
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

PATH = pathlib.Path(__file__).resolve().parents[2] / 'benchmarks/runners/long_audio_benchmark.py'


class Smoke(unittest.TestCase):
    def protocol(self, lm='gpt2'):
        return dict(acoustic=dict(checkpoint_sha256='a' * 64,
            decoding=dict(beam_size=5, language='vi', task='transcribe', temperature=0,
                          without_timestamps=True, fp16=False,
                          ranker='MaximumLikelihoodRanker:length', mel='standard-fp32-v1'),
            runtime=dict(python='3.14.8', torch='2.14.1', **{'openai-whisper': '20250625', 'numpy': '2.5.3'})),
            lm=dict(model_id='org/' + lm, revision='a' * 40, tokenizer_id='org/' + lm,
                    tokenizer_revision='a' * 40,
                    scoring='normalized-mean-v3-model-prefix-or-special-excluded-pll',
                    prefix_id=None if lm == 'phobert' else 50256,
                    prefix_policy='tokenizer-specials-pll' if lm == 'phobert' else 'model-config-bos',
                    vocabulary=dict(embedding_size=50257, config_size=50257,
                                    tokenizer_size=50258, tokenizer_sha256='b' * 64),
                    segmentation='pyvi' if lm == 'phobert' else 'none',
                    runtime=dict(python='3.14.8', torch='2.14.1', transformers='4.57.6',
                                 tokenizers='0.22.2', pyvi='0.1.1', numpy='2.5.3')))

    def test_strict_protocol_schema(self):
        import copy
        m = self.module()
        protocol = self.protocol()
        m.validate_protocol(protocol, scored=True)
        # Every field is required, including all nested runtime/options fields.
        def paths(value, prefix=()):
            for key, child in value.items():
                yield prefix + (key,)
                if isinstance(child, dict):
                    yield from paths(child, prefix + (key,))
        for path in paths(protocol):
            bad = copy.deepcopy(protocol)
            parent = bad
            for key in path[:-1]:
                parent = parent[key]
            del parent[path[-1]]
            with self.subTest(missing=path), self.assertRaisesRegex(ValueError, 'regenerate'):
                m.validate_protocol(bad, scored=True)
        for path, value in [(('acoustic', 'checkpoint_sha256'), 'bad'),
                            (('acoustic', 'decoding', 'beam_size'), True),
                            (('acoustic', 'decoding', 'beam_size'), 1),
                            (('acoustic', 'decoding', 'fp16'), 'false'),
                            (('lm', 'revision'), 'main'), (('lm', 'runtime'), []),
                            (('lm', 'runtime', 'torch'), ''), (('lm', 'segmentation'), 'unknown')]:
            bad = copy.deepcopy(protocol)
            parent = bad
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
            with self.subTest(invalid=path), self.assertRaisesRegex(ValueError, 'regenerate'):
                m.validate_protocol(bad, scored=True)

    def test_protocol_configuration_integrity(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'artifact.json'
            protocol = self.protocol()
            m.artifact(path, 'scored', 'manifest', [], protocol=protocol)
            original = m.load(path)
            m.verify_configuration(original)
            import copy
            for section, field in [('acoustic', 'checkpoint_sha256'), ('acoustic', 'decoding'),
                                   ('lm', 'tokenizer_id'), ('lm', 'scoring'), ('lm', 'runtime')]:
                changed = copy.deepcopy(original)
                changed['protocol'][section][field] = 'changed'
                with self.assertRaisesRegex(ValueError, 'regenerate'):
                    m.verify_configuration(changed)
            with self.assertRaisesRegex(ValueError, 'regenerate'):
                m.verify_configuration({'records': []})
            for invalid in [None, {}, {'acoustic': protocol['acoustic']},
                            dict(protocol, lm={}), dict(protocol, unexpected='legacy')]:
                m.artifact(path, 'scored', 'manifest', [], protocol=invalid)
                with self.subTest(legacy=invalid), self.assertRaisesRegex(ValueError, 'regenerate'):
                    m.verify_configuration(m.load(path))

    def test_position_limit_and_special_targets(self):
        m = self.module()
        config = SimpleNamespace(max_position_embeddings=258, pad_token_id=1)
        tokenizer = SimpleNamespace(model_max_length=256, all_special_ids=[0, 1, 2])
        self.assertEqual(m.lm_context_limit(config, tokenizer, 'phobert'), 256)
        m.check_context_length(256, 256)
        with self.assertRaises(ValueError):
            m.check_context_length(257, 256)
        self.assertEqual(m.pll_positions([0, 42, 43, 2], tokenizer), [1, 2])

    def test_replacement_validation_failure_keeps_active_manifest(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rows = []
            for i in range(60):
                audio = root / (str(i) + '.wav')
                audio.write_bytes(str(i).encode())
                split = 'validation' if i < 30 else 'test'
                rows.append(dict(source='google/fleurs', id=str(i), split=split,
                    original_split=split, source_revision='a' * 40,
                    source_url='https://example.invalid', license='cc-by-4.0',
                    selection='fallback', reference='text ' + str(i), duration=8,
                    audio_path=str(audio), sha256=m.file_hash(audio), pcm_sha256=str(i)))
            m.save(root / 'manifest.json', rows)
            before = (root / 'manifest.json').read_bytes()
            def acquire(http, stage, pool, *args, **kwargs):
                # External acquisition hands back corrupt metadata; real validation rejects it.
                bad = dict(rows[0], id='new', duration=22, sha256='incorrect', pcm_sha256='new')
                pool.append(bad)
            with patch.object(m, 'pcm_info', side_effect=lambda path: (8, pathlib.Path(path).stem)), \
                 patch.object(m, 'fleurs_archive', side_effect=acquire):
                with self.assertRaises(ValueError):
                    m.replace_short(SimpleNamespace(output=root, max_bytes=1000, seconds=1,
                                                    scan_rows=10, min_duration=20))
            self.assertEqual((root / 'manifest.json').read_bytes(), before)

    def test_replacement_preference_and_incomplete(self):
        m = self.module()
        def row(i, split, duration):
            return dict(source='google/fleurs', id=i, split=split, duration=duration,
                        sha256=i, pcm_sha256=i, reference=i)
        old = [row('short', 'test', 8), row('keep', 'test', 16)]
        candidates = [row('fallback', 'test', 18), row('preferred', 'test', 22)]
        result, changes = m.replacement_plan(old, candidates)
        self.assertEqual([r['id'] for r in result], ['preferred', 'keep'])
        self.assertEqual(len(changes), 1)
        self.assertEqual(m.replacement_plan(old, [row('bad', 'validation', 25)])[0], old)
        self.assertEqual(m.replacement_plan(old, [row('tiny', 'test', 14)])[0], old)

    def test_official_archive_selection_and_leak_rejection(self):
        import io
        import tarfile
        m = self.module()
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode='w:gz') as archive:
            for name, data in [('a.wav', b'first'), ('b.wav', b'second')]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        class HTTP(m.BudgetHTTP):
            def get(self, url, token=None):
                # Both splits have the same first transcript: must skip test a.
                return b'1\ta.wav\traw\tshared\ts h a r e d |\t336000\tMALE\n2\tb.wav\traw\tunique test\tu n i q u e |\t128000\tFEMALE\n'
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            with patch.object(m, 'safe_urlopen', side_effect=lambda *a, **k: io.BytesIO(payload.getvalue())), \
                 patch.object(m, 'pcm_info', side_effect=[(21, 'one'), (21, 'one'), (8, 'two')]):
                m.fleurs_archive(HTTP(10000, 1), pathlib.Path(tmp), rows, 'a' * 40, 1, {})
            self.assertEqual([r['split'] for r in rows], ['validation', 'test'])
            self.assertEqual(rows[1]['reference'], 'unique test')
            self.assertTrue(rows[1]['selection'].startswith('fallback'))

    def test_corrected_credential_and_targets(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / 'auth.env'
            p.write_text('export HUGGING_FACE_API_TOKEN=FAKE\n')
            self.assertEqual(m.env_file_token(p), 'FAKE')
        self.assertEqual(m.acquisition_targets(['google/fleurs'], 30), {'google/fleurs': 30})
        with self.assertRaises(ValueError):
            m.acquisition_targets(['google/fleurs'], 0)
    def test_env_file_literal_and_missing(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / 'credentials.env'
            p.write_text('# comment\nexport HF_TOKEN="$(touch NEVER)"\n')
            self.assertEqual(m.env_file_token(p), '$(touch NEVER)')
            p.write_text('OTHER=secret\n')
            with self.assertRaisesRegex(ValueError, 'credential'):
                m.env_file_token(p)

    def test_duration_units(self):
        m = self.module()
        self.assertEqual(m.advertised_duration({'num_samples': 320000}), 20)
        self.assertEqual(m.advertised_duration({'duration': 8.0}), 8)
        self.assertEqual(m.advertised_duration({'duration': -1}), 0)

    def module(self):
        spec = importlib.util.spec_from_file_location('long_audio', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_selection_and_validation_only_tuning(self):
        m = self.module()
        c = [{'id': 'a', 'text': 'xin chào', 'avg_logprob': -0.1, 'lm': -2},
             {'id': 'b', 'text': 'chào', 'avg_logprob': -0.2, 'lm': -0.1}]
        self.assertEqual(m.choose(c, 0)['id'], 'a')
        self.assertEqual(m.choose(c, 1)['id'], 'b')
        self.assertEqual(m.distance('xin chào'.split(), 'chào'.split()), 1)
        self.assertEqual(m.normalize(' XIN, chào! '), 'xin chào')

    def test_reject_test_tuning_and_cross_split_leak(self):
        m = self.module()
        with self.assertRaisesRegex(ValueError, 'validation'):
            m.require_validation([{'split': 'test'}])
        rows = [{'source': 'x', 'id': '1', 'split': 'validation',
                 'sha256': 'abc', 'reference': 'Xin chào', 'group': None},
                {'source': 'x', 'id': '2', 'split': 'test',
                 'sha256': 'def', 'reference': 'xin, chào!', 'group': None}]
        with self.assertRaisesRegex(ValueError, 'leak'):
            m.check_leaks(rows)

    def test_frozen_shared_candidate_pipeline(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            configs = []
            for split in ['validation', 'test']:
                manifest = root / (split + '.json')
                m.save(manifest, [{'source': 'google/fleurs', 'id': split,
                                   'split': split, 'reference': 'xin chào'}])
                for lm in ['gpt2', 'phobert']:
                    scored = root / (split + lm + '.json')
                    m.artifact(scored, 'scored', m.file_hash(manifest), [dict(
                        source='google/fleurs', id=split, split=split, greedy='xin',
                        seconds=.1, lm_seconds=.01, candidates=[
                            dict(id='a', text='xin', avg_logprob=-.1, lm=-5),
                            dict(id='b', text='xin chào', avg_logprob=-.2, lm=-.1)])],
                        protocol=self.protocol(lm),
                        nbest_sha256='shared-' + split, lm=lm, model_id=lm,
                        resolved_revision='a' * 40)
                    if split == 'validation':
                        config = root / (lm + '-config.json')
                        m.tune(SimpleNamespace(manifest=manifest, input=scored, output=config))
                        configs.append(config)
                        self.assertEqual(m.load(config)['alpha'], .05)
                if split == 'test':
                    args = SimpleNamespace(manifest=manifest, gpt2=root / 'testgpt2.json',
                        phobert=root / 'testphobert.json', gpt2_config=configs[0],
                        phobert_config=configs[1], output=root / 'result.json')
                    # Audio decoding is external to this artifact/metric test.
                    with patch.object(m, 'validate', return_value={}):
                        m.evaluate(args)
                        result = m.load(args.output)
                        metrics = result['summary']['test:pooled']['metrics']
                        self.assertEqual(metrics['gpt2']['wer'], 0)
                        self.assertEqual(metrics['beam_top1']['wer'], .5)
                        import copy
                        originals = [m.load(args.gpt2), m.load(args.phobert)]
                        mutations = [(('acoustic', 'checkpoint_sha256'), 'b' * 64),
                            (('acoustic', 'decoding', 'beam_size'), 6),
                            (('acoustic', 'runtime', 'torch'), '2.14.0'),
                            (('lm', 'tokenizer_id'), 'org/other'),
                            (('lm', 'tokenizer_revision'), 'b' * 40),
                            (('lm', 'scoring'), 'normalized-mean-v3-eos-prefix-or-special-excluded-pll'),
                            (('lm', 'runtime', 'transformers'), '4.57.5')]
                        for path, value in mutations:
                            # Recompute fingerprints: this must fail protocol comparison,
                            # not merely the corruption/hash check. Change both acoustics
                            # to reach validation/test comparison independently.
                            for both in ([False, True] if path[0] == 'acoustic' else [False]):
                                for index, target in enumerate([args.gpt2, args.phobert]):
                                    changed = copy.deepcopy(originals[index])
                                    if index == 0 or both:
                                        parent = changed['protocol']
                                        for key in path[:-1]:
                                            parent = parent[key]
                                        parent[path[-1]] = value
                                    records = changed.pop('records')
                                    kind = changed.pop('kind')
                                    manifest_hash = changed.pop('manifest_sha256')
                                    changed.pop('configuration_sha256')
                                    m.artifact(target, kind, manifest_hash, records, **changed)
                                    if path == ('lm', 'scoring') and index == 0:
                                        # A legacy scoring name is now rejected at schema validation,
                                        # before validation/test protocol comparison.
                                        with self.assertRaisesRegex(ValueError, 'regenerate'):
                                            m.verify_configuration(m.load(target))
                                        continue
                                    m.verify_configuration(m.load(target))
                                if path == ('lm', 'scoring'):
                                    with self.assertRaisesRegex(ValueError, 'regenerate'):
                                        m.evaluate(args)
                                    continue
                                expected = 'shared acoustic' if path[0] == 'acoustic' and not both else 'validation/test'
                                with self.subTest(path=path, both=both), self.assertRaisesRegex(ValueError, expected):
                                    m.evaluate(args)
                        for target, original in zip([args.gpt2, args.phobert], originals):
                            m.save(target, original)
                        frozen = m.load(args.gpt2_config)
                        m.save(args.gpt2_config, dict(frozen, alpha=.4))
                        with self.assertRaisesRegex(ValueError, 'regenerate'):
                            m.evaluate(args)
                        m.save(args.gpt2_config, frozen)
                        bad = m.load(args.phobert)
                        bad['records'][0]['candidates'][0]['id'] = 'different'
                        m.save(args.phobert, bad)
                        with self.assertRaisesRegex(ValueError, 'candidate identity'):
                            m.evaluate(args)

    def test_http_byte_budget(self):
        m = self.module()
        import io
        http = m.BudgetHTTP(3, 1)
        with patch.object(m, 'safe_urlopen', return_value=io.BytesIO(b'1234')):
            with self.assertRaisesRegex(RuntimeError, 'byte budget'):
                http.get('https://example.invalid')

    def test_redirect_does_not_forward_token(self):
        m = self.module()
        request = m.urllib.request.Request('https://huggingface.co/file',
                                           headers={'Authorization': 'Bearer FAKE'})
        redirected = m.SafeRedirect().redirect_request(request, None, 302, 'Found', {},
                                                        'https://cdn.example.invalid/file')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaisesRegex(ValueError, 'HTTPS'):
            m.SafeRedirect().redirect_request(request, None, 302, 'Found', {},
                                              'http://cdn.example.invalid/file')

    def test_duration_limit_and_hash(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            audio = pathlib.Path(tmp) / 'audio.wav'
            audio.write_bytes(b'audio')
            row = dict(source='google/fleurs', id='v:0', split='validation',
                       source_revision='a' * 40, source_url='https://example.invalid',
                       license='cc-by-4.0', original_split='validation', selection='preferred',
                       audio_path=str(audio), reference='xin chào', sha256=m.file_hash(audio),
                       pcm_sha256='pcm', duration=22)
            with patch.object(m, 'pcm_info', return_value=(22, 'pcm')):
                self.assertEqual(m.validate([row])['samples'], 1)
            with patch.object(m, 'pcm_info', return_value=(30, 'pcm')):
                with self.assertRaisesRegex(ValueError, 'duration'):
                    m.validate([row])


if __name__ == '__main__':
    unittest.main()
