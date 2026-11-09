import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import convert_phowhisper as c


class ProvenanceTests(unittest.TestCase):
    def test_local_roundtrip_strict_validation(self):
        import torch
        from whisper.model import ModelDimensions, Whisper
        config_data = dict(num_mel_bins=2, max_source_positions=4, d_model=4,
                           encoder_attention_heads=2, encoder_layers=1,
                           vocab_size=8, max_target_positions=4,
                           decoder_attention_heads=2, decoder_layers=1)
        state = Whisper(ModelDimensions(**c.whisper_dimensions(config_data))).state_dict()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, output = root / "source.bin", root / "config.json", root / "native.pt"
            torch.save({k.replace("encoder.ln_post", "model.encoder.layer_norm"): v for k, v in state.items()}, source)
            config.write_text(json.dumps(config_data))
            c.convert(source, config, output)
            result = torch.load(output, weights_only=True)
            self.assertTrue(torch.equal(result["model_state_dict"]["encoder.conv1.weight"], state["encoder.conv1.weight"]))
            record = json.loads(c.sidecar(output).read_text())
            self.assertEqual(record["origin"], "local/unverified")
            self.assertEqual(record["native_sha256"], c.digest(output))
            state["encoder.conv1.weight"] = torch.zeros(1)
            with self.assertRaises(RuntimeError):
                c.validate_state(c.whisper_dimensions(config_data), state)

    def test_pinned_download_and_verified_cache(self):
        sha = "a" * 40
        urls = []
        def fetch(url, path):
            urls.append(url)
            path.write_bytes(b'{}')
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(c, "hub_metadata", return_value={
                "sha": sha, "siblings": [{"rfilename": "pytorch_model.bin"}]
            }), patch.object(c, "download", side_effect=fetch):
                source, config, provenance = c.acquire("vinai/PhoWhisper-small", "main", Path(tmp))
                self.assertEqual(provenance["resolved_revision"], sha)
                self.assertTrue(all(f"/resolve/{sha}/" in u for u in urls))
                c.acquire("vinai/PhoWhisper-small", "main", Path(tmp))
                self.assertEqual(len(urls), 2)
                source.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    c.acquire("vinai/PhoWhisper-small", "main", Path(tmp))

    def test_weight_selection(self):
        self.assertEqual(c.weight_filename(["model.safetensors"]), "model.safetensors")
        for files in (["model.safetensors.index.json"], ["weights.h5"]):
            with self.assertRaises(ValueError):
                c.weight_filename(files)

    def test_no_overwrite_and_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, config, output = root / "source.bin", root / "config.json", root / "out.pt"
            source.write_bytes(b"source")
            config.write_text('{}')
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                c.convert(source, config, output)
            self.assertEqual(output.read_bytes(), b"keep")
            output.unlink()
            with patch.object(c, "load_state_dict", return_value={}), patch.object(c, "whisper_dimensions", return_value={}), patch.object(c, "validate_state", side_effect=ValueError("bad shapes")):
                with self.assertRaises(ValueError):
                    c.convert(source, config, output)
            self.assertFalse(output.exists())
            self.assertFalse(c.sidecar(output).exists())


if __name__ == "__main__":
    unittest.main()
