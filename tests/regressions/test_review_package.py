import tempfile
import unittest
import wave
from pathlib import Path
from tools import human_reference_review as h


class Smoke(unittest.TestCase):
    def test_portable_copy_zip_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root/'source.wav'
            with wave.open(str(audio), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\0\0'*160)
            row = dict(id='test:1', source='google/fleurs', original_id=1,
                       source_revision='rev', audio_path=str(audio),
                       sha256=h.b.file_hash(audio), reference_raw='<raw>',
                       reference_v2='old', reference='a b')
            hyp = dict(candidate_id='fixed', text='a <script> c')
            result = h.package(root/'package', [row], {'test:1': hyp}, {})
            self.assertEqual(result['samples'], 1)
            self.assertEqual(result['audio_hashes_verified'], 1)
            page = (root/'package/test_1/annotated.html').read_text()
            self.assertIn('&lt;script&gt;', page)
            self.assertNotIn('<script>', page)
            self.assertNotIn('file://', page)
            self.assertIn('substitute', page)
            self.assertFalse((root/'package/test_1/audio.wav').is_symlink())
            h.package(root/'package', [row], {'test:1': hyp}, {}, resume=True)
            edited = root/'package/test_1/transcript_reviewed.txt'
            edited.write_text('human edit\n')
            with self.assertRaisesRegex(ValueError, 'edited package file'):
                h.package(root/'package', [row], {'test:1': hyp}, {}, resume=True)
            self.assertEqual(edited.read_text(), 'human edit\n')

    def test_alignment_and_no_selection(self):
        spans = h.m.edit_spans('a b', 'a c d')
        self.assertTrue(any(x['type'] != 'equal' for x in spans))
        with self.assertRaises(ValueError):
            h.safe_id('../outside')


if __name__ == '__main__':
    unittest.main()
