# Official base / small evaluation (no rescoring)

> **Current status 2026-11-10:** user promoted official small after qualitative
> review and additional public-mode measurements. Default is now
> `models/phowhisper-small.pt`; see [archive/rollback](model-migration-small.md).
> This does not reverse the failed historical gate below. Base-retention
> statements describe the 2026-11-09 decision only.

## Historical decision — 2026-11-09

Actual controller runs completed on 2026-11-09: pilot2, dev250 and holdout500.
**Base was retained because the promotion gate failed**, not because evaluation
was deferred. Holdout WER improved 10.8155255% → 9.4853903%, but CER increased
6.2704875% → 6.4872581% and the paired two-sided 95% WER CI contains zero
[-2.75725059, +0.04330839] percentage points. The existing one-sided bootstrap
p=0.0295 / significance flag does not satisfy the two-sided CI gate.
See the [measured report](../reports/phowhisper-base-small-vietbud500-20261109.md)
and [machine-readable decision](../configs/evaluations/phowhisper-base-small-vietbud500.json).
Default remains `models/phowhisper-base.pt`; never overwrite base or historical outputs.

## Frozen protocol

Both subprocesses use seed 42, Vietnamese transcription, temperature 0 (no fallback
schedule), beam_size=None, best_of=None, no timestamps, no previous-text conditioning,
no rescoring, no padding/stretch, and FP16 only on CUDA. No tuning on holdout.
Normalization retains Vietnamese accents and digits. WER/CER are corpus ratios;
paired bootstrap uses 2,000 seeded utterance resamples and reports B-A in percentage
points, where B is small. IDs, references, audio hashes, manifest and runtime protocol
must match. Failures and incomplete runs are rejected, not dropped.

Each checkpoint runs in a fresh sequential process (separate loader cache/RSS).
Default warmup is zero: the first sample is cold, so latency is not claimed to be
steady-state. Optional `--warmup N` repeats the first utterance N times, excluded
from quality and timed samples, identically for both models. It still affects peaks.
RTF includes waveform loading/preprocessing/decoding, not model load or sf.info.
RSS is Linux process high-water memory. CUDA allocated/reserved peaks are allocator
statistics, NOT total nvidia-smi VRAM; total capacity is recorded separately. Other
processes, thermal state and run order can affect results; reserve the GPU exclusively.
Git tracked diff hash and dirty/status hash supplement HEAD. Explicit hashes cover
the benchmark runner, comparator, metrics and pipeline, including untracked comparator
contents; archive the implementation diff and source files for other dependencies too.

## Controller execution and rerun

Run from repository root with existing local WAVs and converted weights. Output
directories must not already exist. Recorded outputs are
`outputs/benchmarks/official-base-small-dev250-20261109` and
`outputs/benchmarks/official-base-small-holdout500-20261109`.
The following rerun commands use the actual official checkpoint paths and fresh
output names; they were not executed by the documentation task. First dev250,
then holdout500 without tuning:

```sh
BASE=models/official/acquisition-20261109/vinai/PhoWhisper-base/7ebdb9e88f5cc5271fb88f4d642c82ff9388650e/native.pt
SMALL=models/official/acquisition-20261109/vinai/PhoWhisper-small/a86b604c346caf7148c37512eafe783a16420adb/native.pt
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_model_sizes --base-checkpoint "$BASE" --small-checkpoint "$SMALL" --manifest benchmarks/manifests/viet_bud500_dev_250.csv --disjoint-from benchmarks/manifests/viet_bud500_holdout_500.csv --output-dir outputs/benchmarks/official-base-small-dev250-rerun --warmup 0
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_model_sizes --base-checkpoint "$BASE" --small-checkpoint "$SMALL" --manifest benchmarks/manifests/viet_bud500_holdout_500.csv --disjoint-from benchmarks/manifests/viet_bud500_dev_250.csv --output-dir outputs/benchmarks/official-base-small-holdout500-rerun --warmup 0
```

For a pilot use a different output directory and `--limit 2`; this is never promotion
evidence. Inspect `comparison.json` / `comparison.md`: `quality` contains absolute
and relative changes, `resources` contains small/base ratios, and
`paired_wer_bootstrap_percentage_points` contains CI bounds. Relative changes are
null when base is zero. Raw summary/predictions remain under base/ and small/.

## Predeclared review rule

No automatic promotion. A favorable candidate requires the full frozen holdout500,
zero failures, small WER < base WER, CER <= base CER, paired WER 95% CI upper < 0,
RTF < 1, and CUDA allocated AND reserved peaks fitting available device memory.
Inspect capacity and competing device usage rather than assuming all capacity is
available. Disclose median/p95 latency and all resource ratios; the user may impose
stricter budgets. CPU-only results cannot establish GPU fit. Review checkpoint
provenance before labeling either result official. The script does not certify the
promotion rule, official origin, or exact holdout identity automatically.

Current rollback uses `PHOWHISPER_MODEL_PATH=models/archive/phowhisper-base/phowhisper-base.pt`
on a machine with the verified local archive; prefer an absolute deployment path
and restart workers. This comparison tooling itself does not change defaults.
