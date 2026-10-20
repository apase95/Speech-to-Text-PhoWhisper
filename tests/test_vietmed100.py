import unittest

from tools.vietmed100_experiment import select_rows


class SelectionTests(unittest.TestCase):
    def test_disjoint_selection(self):
        rows = {s: [dict(speaker_name=s, seq_name=s, audio_name=s,
                         duration=2, text='xin chào', utterance_id=s, accent='North')]
                for s in ('train', 'dev', 'test')}
        self.assertEqual({s: len(v) for s, v in select_rows(rows, (1, 1, 1)).items()},
                         dict(train=1, dev=1, test=1))

    def test_shared_source_fails_closed(self):
        rows = {s: [dict(speaker_name=s, seq_name=s, audio_name='shared',
                         duration=2, text='xin chào', utterance_id=s, accent='North')]
                for s in ('train', 'dev', 'test')}
        with self.assertRaises(ValueError):
            select_rows(rows, (1, 1, 1))
