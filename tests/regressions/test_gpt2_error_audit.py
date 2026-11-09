import unittest
from tools.audit_gpt2_errors import alignment, local_substitutions, highlighted


class AuditSmoke(unittest.TestCase):
    def test_swap(self):
        cs = [dict(id='a', text='x b'), dict(id='b', text='a b')]
        rows = local_substitutions('a b', cs, 'a')
        self.assertEqual(rows[0]['classification'], 'local_selection_opportunity')
        self.assertEqual(rows[0]['correct_candidate_ids'], ['b'])
        self.assertEqual(local_substitutions('a b', cs, 'b'), [])

    def test_repeated_insert_delete(self):
        path, options = alignment('a a'.split(), ['a'])
        self.assertEqual(len(path), 2)
        self.assertTrue(any(len(x) > 1 for x in options.values()))
        self.assertEqual(alignment([], ['z'])[0], [('I', None, 0)])
        self.assertEqual(alignment(['z'], [])[0], [('D', 0, None)])
        cs = [dict(id='a', text='x'), dict(id='b', text='y')]
        self.assertEqual(local_substitutions('z', cs, 'a')[0]['classification'], 'missing_from_aligned_set')

    def test_html(self):
        self.assertEqual(highlighted('<script>2002</script>'), '&lt;script&gt;<mark>2002</mark>&lt;/script&gt;')


if __name__ == '__main__':
    unittest.main()
