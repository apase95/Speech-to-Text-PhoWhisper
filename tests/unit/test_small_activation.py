"""Kiểm tra kích hoạt bằng tệp nhỏ, không nạp trọng số thật."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class Defaults(unittest.TestCase):
    def test_public_defaults(self):
        tree = ast.parse((ROOT/'src/phowhisper/pipeline.py').read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'DEFAULT_MODEL_PATH')
        self.assertEqual(ast.literal_eval(assignment.value.args[0]), 'models/phowhisper-small.pt')
        self.assertIn('default="vinai/PhoWhisper-small"', (ROOT/'scripts/convert/convert_phowhisper.py').read_text())

    def test_environment_override_without_importing_weights(self):
        import os
        tree = ast.parse((ROOT/'src/phowhisper/pipeline.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'model_path')
        namespace = dict(Path=Path, os=os, DEFAULT_MODEL_PATH=Path('models/phowhisper-small.pt'))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<model_path>', 'exec'), namespace)
        with patch.dict(os.environ, PHOWHISPER_MODEL_PATH='custom.pt'):
            self.assertEqual(namespace['model_path'](), Path('custom.pt'))


class Activation(unittest.TestCase):
    def fixture(self, root):
        from scripts.models import activate_small as a
        source = root/'official/native.pt'
        source.parent.mkdir()
        for name, content in [('native.pt', b'small'), ('config.json', b'{"d_model":768}'), ('pytorch_model.bin', b'hf')]:
            (source.parent/name).write_bytes(content)
        hashes = {key: a.digest(source.parent/name) for key, name in
                  [('native_sha256', 'native.pt'), ('config_sha256', 'config.json'), ('source_sha256', 'pytorch_model.bin')]}
        a.sidecar(source).write_text(json.dumps(dict(hashes, origin='huggingface', model_id=a.MODEL_ID, resolved_revision=a.REVISION)))
        models = root/'models'
        models.mkdir()
        old = {}
        for name in a.BASE_HASHES:
            (models/name).write_bytes(('old-'+name).encode())
            old[name] = a.digest(models/name)
        return a, source, models, hashes, old

    def test_install_archive_dry_check_and_idempotency(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, source, models, hashes, old = self.fixture(Path(tmp))
            with patch.object(a, 'SMALL_HASHES', hashes), patch.object(a, 'BASE_HASHES', old):
                a.activate(source, models, archive_base=True)
                self.assertFalse((models/'archive').exists())
                a.activate(source, models, archive_base=True, apply=True)
                self.assertEqual((models/'phowhisper-small.pt').read_bytes(), b'small')
                self.assertEqual((models/'config.json').read_bytes(), (source.parent/'config.json').read_bytes())
                self.assertFalse((models/'phowhisper-base.pt').exists())
                self.assertFalse((models/'pytorch_model.bin').exists())
                archive = models/'archive/phowhisper-base'
                for name, expected in old.items():
                    self.assertEqual(a.digest(archive/name), expected)
                a.activate(source, models, archive_base=True, apply=True)
                self.assertTrue((archive/'manifest.json').is_file())

    def test_mismatch_and_archive_conflict_preserve_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, source, models, hashes, old = self.fixture(Path(tmp))
            with patch.object(a, 'SMALL_HASHES', hashes), patch.object(a, 'BASE_HASHES', old):
                metadata = a.sidecar(source)
                original_metadata = metadata.read_text()
                metadata.write_text(json.dumps(dict(json.loads(original_metadata), model_id='vinai/PhoWhisper-base')))
                with self.assertRaises(ValueError):
                    a.activate(source, models, archive_base=True, apply=True)
                self.assertFalse((models/'archive').exists())
                metadata.write_text(original_metadata)
                source.write_bytes(b'tampered')
                with self.assertRaises(ValueError):
                    a.activate(source, models, archive_base=True, apply=True)
                self.assertFalse((models/'archive').exists())
                source.write_bytes(b'small')
                archive = models/'archive/phowhisper-base'
                archive.mkdir(parents=True)
                (archive/'config.json').write_bytes(b'conflict')
                with self.assertRaises(ValueError):
                    a.activate(source, models, archive_base=True, apply=True)
                for name, expected in old.items():
                    self.assertEqual(a.digest(models/name), expected)
                self.assertEqual((archive/'config.json').read_bytes(), b'conflict')

    def test_canonical_conflict_and_failed_publication_keep_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, source, models, hashes, old = self.fixture(Path(tmp))
            with patch.object(a, 'SMALL_HASHES', hashes), patch.object(a, 'BASE_HASHES', old):
                target = models/'phowhisper-small.pt'
                target.write_bytes(b'unknown')
                with self.assertRaises(ValueError):
                    a.activate(source, models, archive_base=True, apply=True)
                self.assertEqual(target.read_bytes(), b'unknown')
                target.unlink()
                original = a.publish
                def fail_small(src, dest, expected, replace=False):
                    if dest == target:
                        raise OSError('simulated publication failure')
                    return original(src, dest, expected, replace)
                with patch.object(a, 'publish', side_effect=fail_small):
                    with self.assertRaises(OSError):
                        a.activate(source, models, archive_base=True, apply=True)
                for name, expected in old.items():
                    self.assertEqual(a.digest(models/name), expected)
                    self.assertEqual(a.digest(models/'archive/phowhisper-base'/name), expected)


class HistoricalPaths(unittest.TestCase):
    def test_archive_preferred_and_root_fallback(self):
        from benchmarks.checkpoints import legacy_base_file
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            self.assertEqual(legacy_base_file(root=models), models/'phowhisper-base.pt')
            archive = models/'archive/phowhisper-base'
            archive.mkdir(parents=True)
            (archive/'phowhisper-base.pt').write_bytes(b'base')
            (models/'phowhisper-small.pt').write_bytes(b'small')
            self.assertEqual(legacy_base_file(root=models), archive/'phowhisper-base.pt')
            with self.assertRaises(FileNotFoundError):
                legacy_base_file('config.json', root=models)


if __name__ == '__main__':
    unittest.main()
