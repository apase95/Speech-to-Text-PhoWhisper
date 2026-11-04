# FLEURS60 staged benchmark — completed

> **HISTORICAL: original reference orthography.** The active manifests now use
> `fleurs60-spoken-v1`; none of the WER/CER below was recomputed. Old artifacts
> retain their original manifest hashes and must not be rebound to changed data.
> See [reference normalization](reference-normalization-fleurs60.md).

## Final completion (2026-11-04; supersedes historical blockers below)

Completed both declared LMs, validation-only tuning/freeze, one test decode,
both test scores and frozen-config evaluation. No validation re-decode, training,
test tuning, transcript changes, model downloads or weight writes in continuation.
Before resuming, `ps -eo pid,ppid,comm,etime` showed no Python benchmark process
left from the previous timeout. No incremental-scoring or threading code changes
were necessary. Existing `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`
settings were retained across validation/test; no new runtime setting was introduced.

### Results

Pooled equals per-source google/fleurs (the only acquired source). Each split
contains30 native clips, each with5 persisted candidates (150 per split).
Validation reference words1267; test reference words1353. CER includes spaces.

| Method | Validation WER % | Validation CER % | Test WER % | Test CER % | Test word errors |
|---|---:|---:|---:|---:|---:|
| Greedy | 22.8887 | 16.9565 | 19.4383 | 11.3065 | 263 |
| Beam Top-1 | 24.4672 | 14.2261 | 18.5514 | 10.8628 | 251 |
| GPT2, frozen alpha0.4 | 23.0466 | 13.7391 | 17.5166 | 10.6984 | 237 |
| PhoBERT, frozen alpha0.2 | 22.7309 | 13.5826 | 17.5166 | 10.6327 | 237 |
| N-best word-error oracle (diagnostic only) | 22.2573 | 13.3391 | 16.8514 | 10.1726 | 228 |

Validation grid `[0, .05, .1, .2, .4]`, word-error minimum and smallest-alpha
tie-break selected GPT2 alpha0.4 and PhoBERT alpha0.2 before any test decode.
Both reduced test word errors by14 versus beam and26 versus greedy. This is
1.0347 absolute WER percentage points versus beam, not a universal advantage.

Per-sample word-error comparisons (wins / ties / losses):

| Split/method | versus greedy | versus beam |
|---|---|---|
| Validation GPT2 | 14 / 9 / 7 | 17 / 10 / 3 |
| Validation PhoBERT | 13 / 11 / 6 | 16 / 13 / 1 |
| Test GPT2 | 15 / 11 / 4 | 9 / 21 / 0 |
| Test PhoBERT | 16 / 9 / 5 | 10 / 18 / 2 |

### Timing and execution

All continuation stages ran with a3600s tool bound and the benchmark interpreter.
Safe wrappers called the production functions with the CLI-equivalent arguments
below, suppressed third-party output, refused existing output files, measured
monotonic wall time, and appended `continuation-progress.json` after each success.
`R` below denotes `outputs/benchmarks/long/fleurs60-v3-20261104` and `P` denotes
`.venv-benchmark/bin/python tools/long_audio_benchmark.py`; variables are shorthand,
not separately executed shell assignments. Every command inherited the three
environment settings above.

```sh
$P score --input $R/validation/nbest.json --output $R/validation/phobert.json --lm phobert --revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1
$P tune --manifest data/benchmark/long-audio/splits/validation.json --input $R/validation/gpt2.json --output $R/gpt2-frozen.json
$P tune --manifest data/benchmark/long-audio/splits/validation.json --input $R/validation/phobert.json --output $R/phobert-frozen.json
$P evaluate --manifest data/benchmark/long-audio/splits/validation.json --gpt2 $R/validation/gpt2.json --phobert $R/validation/phobert.json --gpt2-config $R/gpt2-frozen.json --phobert-config $R/phobert-frozen.json --output $R/validation/results.json
$P decode --manifest data/benchmark/long-audio/splits/test.json --output $R/test/nbest.json --beam-size 5
$P score --input $R/test/nbest.json --output $R/test/gpt2.json --lm gpt2 --revision 65818d14816b42be09e2201933bf07106d9a2647
$P score --input $R/test/nbest.json --output $R/test/phobert.json --lm phobert --revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1
$P evaluate --manifest data/benchmark/long-audio/splits/test.json --gpt2 $R/test/gpt2.json --phobert $R/test/phobert.json --gpt2-config $R/gpt2-frozen.json --phobert-config $R/phobert-frozen.json --output $R/test/results.json
```

| Successful stage | Wall seconds |
|---|---:|
| Validation PhoBERT | 746.956 |
| Freeze GPT2 / PhoBERT | 0.061 / 0.061 |
| Validation evaluation | 5.216 |
| Test decode (including load/validation) | 50.381 |
| Test GPT2 | 25.465 |
| Test PhoBERT | 715.757 |
| Test evaluation | 5.552 |

Earlier successful validation decode50.840s and GPT225.284s are recorded below.
Summed per-record decode/GPT2/PhoBERT times, excluding model loading:
validation43.764/24.647/741.850s; test41.792/19.825/715.420s.
These are not end-to-end service latency estimates. Prior interrupted work is
additional runtime, not included in successful-stage timing.

### Provenance and verification

Artifacts reside under `$R/{validation,test}/` with `nbest.json`, `gpt2.json`,
`phobert.json`, `results.json`; frozen configs and progress files are at `$R`.
Evaluation passed strict score/frozen fingerprints, validation/test protocol
equality, shared acoustic/N-best identity, sample identity and manifest checks.
Finite scores were enforced during scoring. No scorer changes in this continuation,
so the passing isolated prefix tests were not redundantly rerun; no full suite run.

SHA256:

| Artifact | Hash |
|---|---|
| Local checkpoint | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Combined manifest | `e25f1d2305bdc23b1e58728b15efd1c7a4cc83d29dc315c511c5632e62b49688` |
| Validation N-best | `3fa56e110fdad26d680c139c84c141c351a6386c0e10467a587fb8ae53f36ab8` |
| Test N-best | `d92148b46386fca3ef25d9c6cc771c1c1125c57ed755abd2d47202a257df8b21` |
| GPT2 frozen config | `540ff543b3a3a9a85258f95293ffecc443b84b46b0e6a059958da5eb112c513d` |
| PhoBERT frozen config | `bad1f9bca6c4b0af0d306fac96f188e2b685193e1e1b87b0ee44678cdafdcef4` |

Environment: Python3.14.8 overlay, Torch2.14.1, Transformers4.57.6,
tokenizers0.22.2, NumPy2.5.3, PyVi0.1.1; existing RTX3050 4GB acoustic decoding,
CPU LM scoring. Protocol v3 and immutable model revisions are detailed below.
Human reference audit remains pending and training contamination unknown. This
small Vietnamese read-speech set under30s does not establish long-context benefit,
speaker independence, statistical significance or universal rescoring superiority.
No execution blocker remains for this60-clip benchmark. Only this report and
ignored result artifacts were changed during this final continuation; no commits.

## Authorized v3 continuation (supersedes blocker status below)

Minimal prefix fix implemented and tested on2026-11-04. Cached config inspection
confirmed BOS50256, EOS50256, vocabulary50257, tokenizer EOS50257. Policy is
valid model-config BOS first, otherwise valid model-config EOS, otherwise reject.
For this model it resolves to **model-config-bos50256**, numerically identical
to the requested fixed model-config EOS token. This is an explicit scoring
semantics change, not a tokenizer repair: GPT2 encoding disables automatic special
tokens; every encoded candidate ID must fit the existing embedding vocabulary.
No embeddings/weights are resized or saved.

Scoring version is `normalized-mean-v3-model-prefix-or-special-excluded-pll`.
Strict schema/fingerprints now bind prefix ID/policy, embedding/config vocabulary
sizes, tokenizer length and SHA256 of its sorted token-to-ID mapping. Existing
v2 scored/frozen artifacts are rejected; unchanged acoustic N-best remains valid.
Both LMs' score artifacts were regenerated under v3.

Isolated standard-unittest command (filename argument limits discovery):
`.venv-benchmark/bin/python -m unittest discover -s tests -p test_gpt2_prefix_smoke.py -v`.
RED:2 missing-helper errors,0.002s. GREEN:2 passed,0.000s. Covers mismatched
tokenizer EOS, valid BOS, EOS fallback, valid candidate IDs, invalid/negative/bool
candidate IDs, and no valid model prefix. `git diff --check` passed.

Actual execution command:
```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/execute_staged_fleurs60.py
```
Bounded tool timeout900s terminated the command during validation PhoBERT scoring.
Completed stages, recorded in `outputs/benchmarks/long/fleurs60-v3-20261104/progress.json`:

| Stage | Wall seconds | Result |
|---|---:|---|
| Smoke GPT2, reused original persisted N-best | 5.184 | passed |
| Smoke PhoBERT, same persisted N-best | 30.680 | passed |
| Validation decode, greedy + beam5,30 rows | 50.840 | passed |
| Validation GPT2,150 candidates | 25.284 | passed |
| Validation PhoBERT | interrupted at command timeout | no output artifact |

Smoke verified both configuration fingerprints and identical candidate
IDs/text/acoustic scores plus shared N-best SHA. Production scoring checks finite
LM scores. Validation has exactly5 candidates per row,150 total. GPT2 output
schema/hash verification passed. Validation N-best SHA256:
`3fa56e110fdad26d680c139c84c141c351a6386c0e10467a587fb8ae53f36ab8`.
Smoke N-best SHA256:
`3577b5e11dd11072be46238812fc878147ca78b0076d2d75c1aaae8d011d8df8`.
Tokenizer vocabulary SHA256:
`bb89c3bdb61460573e315986d3091c6167291b2ff35bc9b534945fe6c4768179`.

No freezing or test inference occurred. WER/CER, selected alphas, wins/losses and
four-method test results are not available. Offline mode prevented model downloads
during this continuation. Prior data and artifacts were preserved.

### Resume without decoding validation again

The staged runner deliberately refuses an existing directory; **do not rerun it**
for this partial run. First run the missing score with an explicitly larger bounded
time allowance (the unsuccessful attempt spent roughly788s in this stage):

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/long_audio_benchmark.py score --input outputs/benchmarks/long/fleurs60-v3-20261104/validation/nbest.json --output outputs/benchmarks/long/fleurs60-v3-20261104/validation/phobert.json --lm phobert --revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1
```

Only after it succeeds, run the documented validation-only `tune` commands with
the output root replaced by `outputs/benchmarks/long/fleurs60-v3-20261104`.
Then decode test once, score both LMs, and evaluate using those frozen files, per
`docs/long-audio-benchmark.md`. Never tune test. Runtime remains the blocker;
the current progress file records completed stages, not the outer tool timeout.

Changed in continuation: `tools/long_audio_benchmark.py`,
`tests/test_gpt2_prefix_smoke.py`, `tools/execute_staged_fleurs60.py`, this report.
No full suite run; scope verified is the isolated prefix tests and actual stages
listed above, not all prior schema-test fixtures or the complete benchmark.

Executed 2026-11-04. No training, full-validation decoding, tuning or test
decoding was performed. Accuracy results are not available.

## Actual commands and results

All model operations used `.venv-benchmark/bin/python`.

1. `.venv-benchmark/bin/python tools/check_benchmark_environment.py`
   passed: Python 3.14.8, 138 effective pins, explicit bridge; 0.367s.
2. `.venv-benchmark/bin/python tools/long_audio_benchmark.py validate --manifest data/benchmark/long-audio/manifest.json`
   passed: native FLEURS validation30/test30, 15.24–29.94s,
   total 1236.24s. This checks audio integrity, not test hypotheses.
3. `.venv-benchmark/bin/python tools/run_authorized_benchmark_smoke.py --output outputs/benchmarks/long/smoke-20261104-staged`
   checked validation partition equality, saved a separate first-validation-row
   manifest, then ran actual checkpoint greedy+beam5 decoding: passed in 5.770s.
   GPT2 scoring failed with IndexError in 10.041s. Progress is saved in that
   directory's `progress.json`; it records the stopped sequential runner only.
4. Offline GPT2 scoring retry reproduced IndexError at
   `tools/long_audio_benchmark.py:721`, through GPT2 token embedding.
   Safe numeric inspection established tokenizer length50258 and EOS50257,
   versus model vocabulary50257 and config EOS50256. Candidate text tokens
   remained in bounds (maximum38470); the explicitly prepended tokenizer EOS
   is out of bounds. No weights were resized or token IDs silently replaced.
5. Independent PhoBERT score of the same persisted N-best passed in38.316s,
   with `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, using
   `score(SimpleNamespace(input=Path("outputs/benchmarks/long/smoke-20261104-staged/nbest.json"), output=Path("outputs/benchmarks/long/smoke-20261104-staged/phobert.json"), lm="phobert", revision="86cd7fd4c148980922ac11a2cf5e257f2ba639e1"))`.
   Production scorer rejects nonfinite scores. Five acoustic candidates exist.

## Artifacts, cache and safety

Smoke directory contains manifest.json, nbest.json, phobert.json and progress.json.
Decode artifact contains checkpoint SHA256, runtime protocol, manifest hash,
candidate tokens and timing. PhoBERT artifact contains input N-best hash and
immutable revision. GPT2 revision was65818d14816b42be09e2201933bf07106d9a2647.
No complete shared-two-LM smoke or frozen configuration exists.

Post-attempt targeted cache checks found GPT2 pytorch_model.bin510397801 bytes
and PhoBERT pytorch_model.bin540322347 bytes; neither safetensors file cached.
The initial GPT2 attempt allowed default Hub access before this cache assessment;
network transfer was not measured, so zero-download execution is not claimed.
Subsequent model checks/scoring were offline. No credentials were loaded or
printed; third-party output/errors were suppressed, with error types and stack
frame names only exposed. No packages/CUDA stack installed.

## Resumable blocker and limitations

Resolve the declared GPT2 tokenizer/model EOS mismatch with an explicit,
tested scoring-protocol decision before regenerating scores. Changing EOS prefix
semantics must not silently retain the existing scoring version. Existing N-best
can be reused if the acoustic protocol is unchanged. Run GPT2 and PhoBERT scores
against that same N-best, verify both artifacts and identities, then continue
the documented validation30 -> tune/freeze -> test30 -> evaluate sequence.
Do not tune on test. The runner refuses an existing output directory; use a fresh
directory to repeat its complete smoke rather than overwrite evidence.

No production runtime fix was made, so no RED/GREEN fix test is claimed.
Full suites were skipped (not requested); full benchmark stages were skipped
because the smoke gate failed. Human reference audit remains pending,
contamination unknown, and these under30s read-speech clips cannot establish
general long-context or universal rescoring advantages.
