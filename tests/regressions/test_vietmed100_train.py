import unittest
from tools.vietmed100_train import teacher_tokens, accumulation_size


class TrainingSmoke(unittest.TestCase):
    def test_checkpoint_default_and_override(self):
        from training.local.vietmed100_train import parse_args
        self.assertEqual(str(parse_args([]).checkpoint), 'models/phowhisper-small.pt')
        self.assertEqual(str(parse_args(['--checkpoint', 'custom.pt']).checkpoint), 'custom.pt')

    def test_initializer_is_independent_of_frozen_data(self):
        import ast
        import inspect
        from training.local import vietmed100_train
        tree = ast.parse(inspect.getsource(vietmed100_train))
        assertions = [ast.unparse(node.test) for node in ast.walk(tree)
                      if isinstance(node, ast.Assert)]
        self.assertFalse(any("prep['base_sha256_before']" in text for text in assertions))
        self.assertIn("sha(csv_path) == prep['stages'][split]['manifest_sha256']", assertions)
        self.assertIn("sha(row['audio_path']) == row['sha256']", assertions)
        self.assertIn("sha(base) == report['base_sha256_before']", assertions)

    def test_teacher_shift(self):
        x, y = teacher_tokens([1, 2, 3], [10, 11], 99)
        self.assertEqual(x, [1, 2, 3, 10, 11])
        self.assertEqual(y, [-100, -100, 10, 11, 99])

    def test_final_partial_batch(self):
        self.assertEqual(accumulation_size(68, 70, 4), 2)
        with self.assertRaises(ValueError):
            teacher_tokens([], [10], 99)
