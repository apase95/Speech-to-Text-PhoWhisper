import unittest
from tools import diagnose_general_cached as d


class Smoke(unittest.TestCase):
    def test_longer_candidate_and_comparison(self):
        cs=[dict(id='short',text='chim bay',avg_logprob=0,lm=0),
            dict(id='long',text='chim co long bay',avg_logprob=-1,lm=-1)]
        result=d.sample('chim co long bay','chim co long bay',cs,.3,-.02)
        self.assertEqual(result['metrics']['gpt2']['D'],2)
        self.assertEqual(result['metrics']['greedy']['D'],0)
        self.assertEqual(result['deletion_spans'][0]['phrase'],'co long')
        self.assertEqual(result['deletion_spans'][0]['candidate_indices_containing_phrase'],[1])
        self.assertEqual(d.outcome(2,0),'loss')
        self.assertEqual(d.outcome(0,0),'tie')

    def test_absent_phrase(self):
        cs=[dict(id='a',text='chim bay',avg_logprob=0,lm=0)]
        result=d.sample('chim co long bay','chim co long bay',cs,.3,-.02)
        self.assertEqual(result['deletion_spans'][0]['candidate_indices_containing_phrase'],[])
        self.assertTrue(result['deletion_spans'][0]['greedy_contains_phrase'])
