import unittest

from tools.convert_phowhisper import convert_key, whisper_dimensions


class ConversionTests(unittest.TestCase):
    def test_maps_encoder_attention_key(self):
        self.assertEqual(
            convert_key("model.encoder.layers.0.self_attn.q_proj.weight"),
            "encoder.blocks.0.attn.query.weight",
        )

    def test_maps_decoder_cross_attention_key(self):
        self.assertEqual(
            convert_key("model.decoder.layers.1.encoder_attn_layer_norm.weight"),
            "decoder.blocks.1.cross_attn_ln.weight",
        )

    def test_dimensions_follow_source_config(self):
        config = {
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "d_model": 512,
            "encoder_attention_heads": 8,
            "encoder_layers": 6,
            "vocab_size": 51865,
            "max_target_positions": 448,
            "decoder_attention_heads": 8,
            "decoder_layers": 6,
        }
        self.assertEqual(whisper_dimensions(config)["n_audio_state"], 512)
        self.assertEqual(whisper_dimensions(config)["n_vocab"], 51865)


if __name__ == "__main__":
    unittest.main()
