import unittest
from tools.run_fleurs_holdout import disjoint, evaluate


class HoldoutSmoke(unittest.TestCase):
    def test_disjoint(self):
        old = dict(source='google/fleurs',id='test:1',original_id=1,sha256='a',pcm_sha256='b',reference='c')
        new = dict(old,id='test:2',original_id=2,sha256='d',pcm_sha256='e',reference='f')
        self.assertTrue(disjoint(new,[old]))
        for key in ('id','original_id','sha256','pcm_sha256','reference'):
            self.assertFalse(disjoint(dict(new,**{key:old[key]}),[old]))

    def test_frozen_not_oracle(self):
        cs=[dict(id='a',text='wrong',avg_logprob=0,lm=0),dict(id='b',text='right',avg_logprob=-10,lm=0)]
        summary, rows=evaluate([dict(source='s',id='i',reference='right')], [dict(source='s',id='i',greedy='right',candidates=cs)], .3,-.02)
        self.assertEqual(rows[0]['selected_candidate_id'],'a')
        self.assertEqual(summary['A']['gpt2_frozen']['errors'],1)
        self.assertEqual(summary['A']['oracle']['errors'],0)


if __name__ == '__main__':
    unittest.main()
