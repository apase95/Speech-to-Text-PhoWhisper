# VietMed100 public workflow comparison protocol

Run from repository root with `PYTHONPATH=.:src` and the benchmark interpreter.
Completed results are linked below; no inference was rerun for this guide.
Current default is official small by user decision on 2026-11-10, not automatic promotion.

Full command:
```
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_vietmed_modes --output-dir outputs/benchmarks/vietmed-base-small-modes-20261109-full
```
Pilot (not full-dataset evidence; use a fresh output directory):
```
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_vietmed_modes --output-dir outputs/benchmarks/vietmed-base-small-modes-20261109-pilot --limit 2
```

Four sequential fresh processes use official unmodified base/small checkpoints,
not the historical local fine-tuned model. Public `pipeline.transcribe` is used
unchanged: fast retains Whisper's default temperature fallback (unlike the
preceding VietBud500 greedy experiment); rescoring uses public beam 5,
alpha .2, beta -.02, gamma -.2 and acoustic gate -.020. This compares complete
workflows, not a pure causal LM ablation. Small inherits untuned base settings.
No training, tuning, stretching or default-model promotion occurs.

The LM is pinned to revision 65818d14816b42be09e2201933bf07106d9a2647,
offline/local-only, FP32 on CUDA. Fast never loads the LM. CUDA ASR uses the
public FP16 path. Failures/OOM and rescorer warnings block completion; no CPU
fallback is allowed. One excluded dev[0] warmup follows preload in every cell.
Batch 1, seed 42, four threads; synchronized complete API calls include audio I/O.
Load and measured CUDA peaks include resident allocations; measured peaks reset
after warmup. RSS is the whole worker process high-water mark, not incremental
model memory. Fixed-order single-run resource numbers are not robust timing claims.

Frozen manifests/CSV and every audio hash are checked before execution; input,
checkpoint, sidecar, pinned LM file and actual implementation source hashes are
archived and checked after execution. Per-sample JSONL is flushed for durability.
Any failure or incomplete cell invalidates comparison. Metrics are recomputed
from saved predictions, paired by exact ordered IDs/reference/audio hashes.
Pooled all100 and train70/dev15/test15 are reported from the same predictions.
These split names came from a previous local fine-tuning experiment; the official
model descriptive all100 comparison does not claim unseen pretraining data.
Test15 is too small to establish general ASR quality. Medical-domain labels and
possible near duplicates/pretraining exposure limit generalization.

WER/CER use project normalization retaining accents; word/character S/D/I/hits,
exact match, latency mean/median/nearest-rank p95 and pooled RTF are reported.
Four pairwise comparisons include paired utterance bootstrap WER B-A percentage
points, 2000 draws, seed 42. Negative favors B. The reported p-value is one-sided
fraction of nonnegative differences, whereas the percentile 95% interval is
two-sided; do not interpret these as identical significance criteria. No automatic
promotion or base default change is authorized.

## Completed saved four-cell run (2026-11-09)

The full run is completed and independently audited from saved predictions:
four cells × 100 rows, no errors, not a pilot. See the
[Vietnamese final report](../reports/vietmed100-base-small-fast-rescoring-20261109.md)
and [curated result/config](../configs/evaluations/vietmed100-base-small-modes.json).
The opening protocol-only disclaimer does not supersede these linked actual results.
Small-rescore leads descriptive all100; small-fast leads tiny test15. Small WER/CER:
all100 fast 15.8623%/11.6275%, rescore 15.0712%/11.1571%; test15 fast
17.4479%/12.4456%, rescore 18.4896%/13.2545%. Rescoring can hurt.
100 clips total about 625 seconds, at most 8 seconds each, not 100 hours.
The run did not change defaults; later user promotion is documented in
[migration/archive](model-migration-small.md). No inference rerun for the final audit.
