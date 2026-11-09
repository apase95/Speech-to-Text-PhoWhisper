"""Atomic, lightweight training provenance, without changing save policy."""
import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class EpochManifest:
    """One writer per output directory; callers must enforce process-zero writes.

    Only integral epoch-end events are completed epochs. Saved checkpoints are
    associated by exact global step, not by nearest epoch or directory name.
    Repeated events within one invocation are idempotent; resume invocations
    retain separate provenance. No checkpoint is saved or deleted here.
    """

    def __init__(self, output, provenance):
        self.output = Path(output).resolve()
        self.path = self.output / "epochs.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "schema_version": 1, "invocations": [], "epochs": [],
            "checkpoints": [], "final_models": [],
        }
        if self.data["schema_version"] != 1:
            raise ValueError("Unsupported epoch metadata schema")
        self.invocation = uuid.uuid4().hex
        self._metrics = {}
        self.data["invocations"].append(dict(id=self.invocation, started_at=_now(), provenance=provenance))
        self.flush()

    def epoch_end(self, epoch, global_step, metrics=None):
        if epoch is None or not math.isfinite(epoch) or epoch <= 0 or abs(epoch - round(epoch)) > 1e-8:
            return
        record = next((r for r in self.data["epochs"] if
                       (r["invocation"], r["epoch"], r["global_step"]) ==
                       (self.invocation, round(epoch), global_step)), None)
        if record is None:
            self.data["epochs"].append(dict(invocation=self.invocation, epoch=round(epoch),
                global_step=global_step, timestamp=_now(),
                metrics={**self._metrics.get(global_step, {}), **(metrics or {})}, checkpoint_steps=[]))
        elif metrics:
            record["metrics"].update(metrics)
        self.flush()

    def observed_metrics(self, global_step, metrics):
        """Reconcile late evaluation/log events only at their actual step.

        Metrics at other steps are never attributed to an epoch boundary.
        Callers must supply only observed values, not cumulative log history.
        """
        self._metrics.setdefault(global_step, {}).update(metrics)
        for record in self.data["epochs"]:
            if record["invocation"] == self.invocation and record["global_step"] == global_step:
                record["metrics"].update(metrics)
        self.flush()

    def saved(self, global_step, path):
        path = Path(path).resolve()
        if not path.is_dir():
            raise ValueError("Checkpoint save must be observed after directory creation")
        if not any(r["invocation"] == self.invocation and r["path"] == str(path)
                   for r in self.data["checkpoints"]):
            self.data["checkpoints"].append(dict(invocation=self.invocation,
                global_step=global_step, path=str(path), timestamp=_now()))
        self.flush()

    def final_model(self, path, global_step):
        if not Path(path).exists():
            raise ValueError("Final model must exist after successful save")
        self.data["final_models"].append(dict(invocation=self.invocation,
            path=str(Path(path).resolve()), global_step=global_step, timestamp=_now()))
        self.flush()

    def flush(self):
        for checkpoint in self.data["checkpoints"]:
            checkpoint["exists"] = Path(checkpoint["path"]).is_dir()
        for epoch in self.data["epochs"]:
            epoch["checkpoint_steps"] = [r["global_step"] for r in self.data["checkpoints"]
                if r["invocation"] == epoch["invocation"] and r["global_step"] == epoch["global_step"]]
        self.output.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_clean(self.data), ensure_ascii=False, allow_nan=False, indent=2)
        fd, temporary = tempfile.mkstemp(prefix=".epochs-", suffix=".tmp", dir=self.output)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class EpochMetadataCallback:
    """Lightweight mixin; combine with TrainerCallback at the integration boundary."""

    def __init__(self, provenance):
        self.provenance = provenance
        self.manifest = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.manifest = EpochManifest(args.output_dir, self.provenance)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.manifest and logs:
            self.manifest.observed_metrics(state.global_step, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.on_log(args, state, control, logs=metrics)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.manifest:
            self.manifest.epoch_end(state.epoch, state.global_step)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.manifest:
            checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
            if checkpoint.is_dir():
                self.manifest.saved(state.global_step, checkpoint)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.manifest:
            self.manifest.flush()

    def final_model(self, path, state):
        if state.is_world_process_zero and self.manifest:
            self.manifest.final_model(path, state.global_step)
