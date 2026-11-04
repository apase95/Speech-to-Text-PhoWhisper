import unittest
from types import SimpleNamespace
from tools import long_audio_benchmark as b


class PrefixSmoke(unittest.TestCase):
    def test_model_prefix_not_tokenizer_eos(self):
        config = SimpleNamespace(bos_token_id=50256, eos_token_id=50256)
        tokenizer = SimpleNamespace(eos_token_id=50257)
        self.assertEqual(b.gpt2_prefix(config, 50257), (50256, 'model-config-bos'))
        self.assertNotEqual(b.gpt2_prefix(config, 50257)[0], tokenizer.eos_token_id)
        config.bos_token_id = 50257
        self.assertEqual(b.gpt2_prefix(config, 50257), (50256, 'model-config-eos'))
        b.validate_candidate_ids([0, 50256], 50257)

    def test_invalid_ids_and_no_prefix(self):
        for ids in ([50257], [-1], [True]):
            with self.assertRaises(ValueError):
                b.validate_candidate_ids(ids, 50257)
        with self.assertRaises(ValueError):
            b.gpt2_prefix(SimpleNamespace(bos_token_id=None, eos_token_id=50257), 50257)


if __name__ == '__main__':
    unittest.main()
