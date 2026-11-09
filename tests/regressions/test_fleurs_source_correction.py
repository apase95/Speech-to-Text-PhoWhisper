import copy
import unittest
from tools import long_audio_benchmark as b


class SourceCorrectionSmoke(unittest.TestCase):
    def test_unmatched_quote_literal_seven_fields(self):
        text='5\ta.wav\t"Raw sentence."\t"raw sentence.\t" r a w | s e n t e n c e . |\t16000\tMALE\n'
        rows=b.parse_fleurs_tsv(text)
        self.assertEqual(len(rows[0]),7)
        self.assertEqual(rows[0][3],'"raw sentence.')
        self.assertNotIn('r a w',rows[0][3])

    def test_schema_rejects(self):
        valid='5\ta.wav\tRaw\ttext\tt e x t |\t16000\tMALE\n'
        for text in (valid.replace('\tMALE',''),valid.replace('a.wav','../a.wav'),
                     valid.replace('16000','0'),valid.replace('5\t','bad\t'),valid+valid):
            with self.subTest(text=text),self.assertRaises(ValueError):
                b.parse_fleurs_tsv(text)

    def test_overlay_guard_and_preservation(self):
        from tools.correct_fleurs_sources import correct_rows, verify_reference_only
        row=dict(source='google/fleurs',id='test:0',original_id=5,original_split='test',split='test',
                 source_revision='a'*40,reference='sentence c h a r',duration=1,archive_member='test/a.wav',
                 sha256='file',pcm_sha256='pcm',source_url='https://example.invalid',license='cc-by-4.0')
        metadata=b.parse_fleurs_tsv('5\ta.wav\tSentence\tsentence\ts e n t e n c e |\t16000\tMALE\n')
        result,evidence=correct_rows([row],{'test':metadata},{'test':'metadata-sha'})
        self.assertEqual(result[0]['reference'],'sentence')
        self.assertEqual(result[0]['reference_previous'],'sentence c h a r')
        self.assertEqual(result[0]['source_reference_v2'],'sentence')
        self.assertEqual(row['reference'],'sentence c h a r')
        self.assertTrue(evidence[0]['normalized_difference'])
        verify_reference_only([row],result)
        forged=copy.deepcopy(result); forged[0]['sha256']='swapped'
        with self.assertRaises(ValueError):
            verify_reference_only([row],forged)
        forged=copy.deepcopy(result); forged[0]['reference']='arbitrary text'
        with self.assertRaises(ValueError):
            verify_reference_only([row],forged)
        for key,value in [('original_id',6),('archive_member','test/wrong.wav')]:
            with self.assertRaises(ValueError):
                correct_rows([dict(row,**{key:value})],{'test':metadata},{'test':'sha'})

    def test_annotated_old_reference_not_overwritten(self):
        from tools.correct_fleurs_sources import correct_rows
        row=dict(source='google/fleurs',id='test:0',original_id=5,original_split='test',split='test',
                 source_revision='a'*40,reference='một tuổi',reference_raw='1 tuổi',
                 reference_convention='fleurs60-spoken-v2',duration=1,archive_member='test/a.wav')
        metadata=b.parse_fleurs_tsv('5\ta.wav\t1 tuổi\t1 tuổi\t1 | t u ổ i |\t16000\tMALE\n')
        result,evidence=correct_rows([row],{'test':metadata},{'test':'sha'},activate=False)
        self.assertEqual(result[0]['reference'],'một tuổi')
        self.assertEqual(result[0]['reference_raw'],'1 tuổi')
        self.assertEqual(result[0]['source_reference_v2'],'1 tuổi')
        self.assertFalse(evidence[0]['normalized_difference'])

    def test_viewer_ordinal_is_not_tsv_ordinal(self):
        from tools.correct_fleurs_sources import correct_rows
        row=dict(source='google/fleurs',id='validation:99',original_id=5,original_split='validation',
                 split='validation',source_revision='a'*40,reference='text',duration=1)
        metadata=b.parse_fleurs_tsv('5\ta.wav\tText\ttext\tt e x t |\t16000\tMALE\n')
        result,evidence=correct_rows([row],{'validation':metadata},{'validation':'sha'},activate=False)
        self.assertEqual(result[0]['id'],'validation:99')
        self.assertEqual(evidence[0]['matching_tsv_indices'],[0])
        self.assertFalse(evidence[0]['archive_filename_persisted'])


if __name__=='__main__':
    unittest.main()
