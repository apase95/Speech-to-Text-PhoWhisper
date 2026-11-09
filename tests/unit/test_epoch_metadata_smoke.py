"""Isolated persistence checks; no model or network dependencies."""
import json
from unittest.mock import patch

from training.metadata import EpochManifest


def test_epoch_save_prune_resume(tmp_path):
    manifest = EpochManifest(tmp_path, {"model": "local"})
    manifest.epoch_end(1, 10, {"loss": 0.5})
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    manifest.saved(10, checkpoint)
    manifest.final_model(tmp_path, 10)
    checkpoint.rmdir()
    resumed = EpochManifest(tmp_path, {"resume": "checkpoint-10"})
    resumed.flush()
    data = json.loads((tmp_path / "epochs.json").read_text())
    assert len(data["invocations"]) == 2
    assert data["epochs"][0]["checkpoint_steps"] == [10]
    assert data["checkpoints"][0]["exists"] is False
    assert data["final_models"][0]["global_step"] == 10


def test_partial_nonfinite_and_atomic(tmp_path):
    manifest = EpochManifest(tmp_path, {})
    manifest.epoch_end(0.4, 4, {"loss": float("nan")})
    data = json.loads((tmp_path / "epochs.json").read_text())
    assert data["epochs"] == []
    manifest.epoch_end(1, 10, {"loss": float("inf")})
    assert json.loads((tmp_path / "epochs.json").read_text())["epochs"][0]["metrics"]["loss"] is None
    previous = (tmp_path / "epochs.json").read_bytes()
    def fail(*args):
        raise OSError("replacement failed")
    with patch("training.metadata.os.replace", fail):
        try:
            manifest.flush()
        except OSError:
            pass
        else:
            raise AssertionError("Expected replacement failure")
    assert (tmp_path / "epochs.json").read_bytes() == previous


def test_save_before_epoch_and_late_metrics(tmp_path):
    manifest = EpochManifest(tmp_path, {})
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    manifest.saved(10, checkpoint)
    manifest.epoch_end(1, 10, {"loss": 0.5})
    manifest.observed_metrics(10, {"eval_wer": 0.25})
    manifest.epoch_end(1, 10)
    data = json.loads(manifest.path.read_text())
    assert len(data["epochs"]) == 1
    assert data["epochs"][0]["metrics"] == {"loss": 0.5, "eval_wer": 0.25}
    assert data["epochs"][0]["checkpoint_steps"] == [10]


def test_no_save_and_new_output_resume(tmp_path):
    manifest = EpochManifest(tmp_path / "new", {"resume": "/other/checkpoint-10"})
    manifest.observed_metrics(11, {"loss": 0.8})
    manifest.epoch_end(2, 20)
    manifest.epoch_end(2.5, 25)
    data = json.loads(manifest.path.read_text())
    assert len(data["epochs"]) == 1
    assert data["epochs"][0]["metrics"] == {}
    assert data["epochs"][0]["checkpoint_steps"] == []
    assert data["checkpoints"] == []


if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory
    for test in (test_epoch_save_prune_resume, test_partial_nonfinite_and_atomic,
                 test_save_before_epoch_and_late_metrics, test_no_save_and_new_output_resume):
        with TemporaryDirectory() as directory:
            test(Path(directory))
    print("4 epoch metadata smoke checks passed")
