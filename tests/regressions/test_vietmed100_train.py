import unittest
from tools.vietmed100_train import teacher_tokens, accumulation_size


class TrainingSmoke(unittest.TestCase):
    def test_teacher_shift(self):
        x, y = teacher_tokens([1, 2, 3], [10, 11], 99)
        self.assertEqual(x, [1, 2, 3, 10, 11])
        self.assertEqual(y, [-100, -100, 10, 11, 99])

    def test_final_partial_batch(self):
        self.assertEqual(accumulation_size(68, 70, 4), 2)
        with self.assertRaises(ValueError):
            teacher_tokens([], [10], 99)
