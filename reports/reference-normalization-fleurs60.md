# FLEURS60 spoken references v1

Text-only curated orthography, **not human audio validation**. No models run.
Reference changes affect scoring and validation selection, NOT actual model input.
`transcript_audit` remains pending; annotations carry `audio_validated: false`.

## Inventory and actual application

Original 60 samples: 13 digit-bearing references, 23 numeric expressions (a
punctuated numeral counts once), including seven explicit measurements. Applied
seven spans in five references: validation 2, test 3. All 60 rows now preserve
`reference_raw` and carry `reference_convention: fleurs60-spoken-v1`.

| Sample | Original → spoken form |
|---|---|
| validation:151 | 100m → một trăm mét; 200m → hai trăm mét |
| validation:231 | 4892 m → bốn nghìn tám trăm chín mươi hai mét |
| test:121 | 50 kilomet → năm mươi ki lô mét; 31 dặm → ba mươi mốt dặm |
| test:653 | 70km/h → bảy mươi ki lô mét trên giờ |
| test:661 | 15 dặm → mười lăm dặm |

Remaining: 16 numeric expressions in nine references. These are deliberately
pending separate text/audio review, not claimed to be intrinsically ambiguous:

| Sample | Preserved forms / reason |
|---|---|
| validation:126 | 400.000: grouped count, outside v1 measurement scope |
| validation:237 | 6, 10, 1789: date; 11, 4: ages |
| validation:73 | 108, 56: counts |
| validation:260 | 1.000: ordinal; 2000: year |
| test:153 | 45, 117: counts |
| test:266 | 2002: citation year |
| test:813 | 1988: year |
| test:472 | 2002: year |
| test:661 | 29: age |

No currency expressions found. No blanket `m`→`mét` rule: swimming distances and
summit-height context justify these specific entries only. No lexical corrections
or date/code/year expansion. Dialectal number/unit realizations still need audio
review; this is an explicit scoring convention, not evidence of what was spoken.

## Reproduction and safety

Versioned `benchmarks/annotations/fleurs60-spoken-references.json` is compact:
full raw/current source text plus ordered unique old/new spans and context rules;
current equals raw, new text and Unicode character offsets are deterministically
expanded in memory. Source, revision, stable ID and original ID are guarded.
No audio paths or runtime results are included in the annotations.

```
.venv-benchmark/bin/python tools/normalize_benchmark_references.py audit
.venv-benchmark/bin/python tools/normalize_benchmark_references.py apply
```

Audit is read-only. Apply validates the complete proposed manifest with existing
file/PCM hashes, full FFmpeg decoding and leakage checks before writing. It refuses
incoherent old partitions and raw/provenance mismatches. All other fields survive.
Each file is atomically replaced; caught write failures restore written files.
This is NOT a crash-safe multi-file transaction: avoid concurrent writers and
restore the three backup files after an interrupted publication if needed.
Backups are under ignored `data/benchmark/long-audio/reference-backups/`.
Actual first backup: `1793754446700508593` (manifest, validation, test JSON).
Repeated apply changes zero files. Native validation30/test30 remain coherent;
audio unchanged, 15.24–29.94 seconds, total 1236.24 seconds.

Historical outputs under `fleurs60-v3-20261104` and
`gpt2-optimization-20261104` are untouched. Old WER/CER and frozen selection are
**HISTORICAL under original references**. Existing hash guards deliberately reject
joining these outputs to active manifests. Default regression tests use tiny
synthetic manifests and hash-bound score/N-best artifacts in temporary directories,
not ignored local manifests or historical outputs.
No new WER/CER is claimed. Re-evaluation must use a named convention consistently
for references and hypotheses and preserve provenance; never edit old hashes.

## Verification and priorities

Standard `unittest discover -s tests -p FILE -v` is filename-limited (no custom
runner). New tests RED: missing module ImportError (0.000s), then GREEN. Tests
exercise all seven curated spans, ambiguous preservation, raw/revision rejection,
invalid replacement output, idempotence, partition coherence and injected write
failure rollback. External audio decoding is excluded from the isolated unit
fixture; actual apply runs real complete audio validation.

Mandatory existing four isolated test files pass (25 tests). Optimization tests
use independent persisted-metric auditing and take 2.459s test time, exceeding
the 2s target; no full suite run. Actual repeated apply with full validation
passed in 6.558s wall. No inference, downloads, tuning, commits or pushes.

Next: human audio audit, review remaining numbers and dialectal forms, then a
separately versioned evaluation with validation-only selection; use fresh held-out
speech before generalization claims. Controller owns commit grouping and push.

## Portable checkout tests / optional historical audit

The five scoped files run without ignored data/audio/model outputs. The ranker
test uses installed Whisper code on synthetic tokens (skips if unavailable);
normalization tests read only the versioned annotation JSON and temporary fixtures.
Optimization default tests cover validation-only selection, freeze fingerprint and
protocol rejection, synthetic hash-bound candidate/source integrity, S/D/I and
diagnostic aggregation, and winner/freeze persistence before the test boundary.
No production checks were relaxed.

The independent persisted-result test is explicitly opt-in:

```
FLEURS60_PERSISTED_AUDIT=1 .venv-benchmark/bin/python -m unittest discover -s tests -p test_gpt2_optimization.py -v
```

Prerequisites: `jiwer` in the interpreter and the original, unmodified historical
JSON diagnostics under `outputs/benchmarks/long/gpt2-optimization-20261104/`:
`beam5-validation.json`, `beam10-validation.json`, `beam15-validation.json`,
`beam5-test.json`, `beam15-test.json`. No audio, model weights or active manifest
is required for this audit. Missing prerequisites fail when explicitly enabled;
the audit is skipped by default, not a mandatory checkout test. The existing
`tools/gpt2_optimization.py --audit` is also an explicit local persisted audit,
but writes audit/report artifacts; use the opt-in test for read-only verification.

Portability fix RED: denying ignored artifact reads exposed six failures (0.008s).
GREEN default scoped results: normalization4, long-audio15, prefix2, ranker2,
optimization6 passed plus one opt-in audit skipped. Optimization execution is now
0.010s (0.110s process wall), below the isolated smoke target. Historical timings
above describe the prior artifact-dependent tests, not the portable defaults.
