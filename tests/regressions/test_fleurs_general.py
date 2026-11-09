import unittest
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch
from tools import curate_fleurs_general as c


class Smoke(unittest.TestCase):
    def test_reference_only(self):
        self.assertEqual(c.automatic_flags('một con chim đang bay'), [])
        self.assertTrue(c.automatic_flags('năm 2002'))
        self.assertTrue(c.automatic_flags('hai mươi tuổi'))
        self.assertEqual(c.automatic_flags('khối thịnh vượng và lông vũ'), [])
        with self.assertRaises(TypeError):
            c.automatic_flags('chim', hypothesis='chim')

    def test_identity_and_literal_schema(self):
        row = dict(source='google/fleurs', id='test:1', original_id=1,
                   original_split='test', split='test', sha256='a', pcm_sha256='b', reference='chim')
        c.unique([row])
        with self.assertRaises(ValueError):
            c.unique([row, dict(row, id='validation:1', split='validation')])
        with self.assertRaises(ValueError):
            c.unique([dict(row, split='validation')])
        text = '1\tx.wav\t"raw\t"chim\tc h i m\t240000\tFEMALE\n'
        self.assertEqual(c.b.parse_fleurs_tsv(text)[0][3], '"chim')
        sealed = c.o.seal(dict(alpha=.3, beta=-.02))
        c.o.verify(sealed)
        with self.assertRaises(ValueError):
            c.o.verify(dict(sealed, alpha=.4))
        self.assertEqual(c.m.measure(['chim'], ['chim'])['errors'], 0)
        self.assertEqual(c.m.measure(['chim'], [''])['D'], 1)

    def test_screened_spans_and_portable_package(self):
        self.assertEqual(c.manual_flags('Một con chim có lông vũ', 'một con chim có lông vũ'), [])
        self.assertTrue(any(x['span']=='java' for x in c.manual_flags('ẩm thực Java', 'ẩm thực java')))
        self.assertTrue(c.manual_flags('trong một năm', 'trong một năm'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); audio=root/'source.wav'
            with wave.open(str(audio),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(b'\0\0'*160)
            c.b.save(root/'manifest.json',[])
            row=dict(id='test:1',audio_path=str(audio),sha256=c.b.file_hash(audio),
                     reference_source_raw='<chim>',reference_raw='chim',reference='chim',
                     source_revision='revision',curation_role='replacement')
            record=dict(id='test:1',candidates=[dict(id='c',text='chim <script>',avg_logprob=0,lm=0)])
            with patch.object(c.b,'ROOT',root),patch.object(c,'DATA',root):
                result=c.review_package([row],[record])
                self.assertTrue(result['verified'])
                page=(root/result['path']/'test_1/review.html').read_text()
                self.assertIn('&lt;script&gt;',page)
                self.assertNotIn('file://',page)
                with self.assertRaises(ValueError):
                    c.review_package([row],[record])
