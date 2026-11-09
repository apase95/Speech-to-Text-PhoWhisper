# Bounded Vietnamese long-clip benchmark

## Reference convention update

Active FLEURS60 manifests now use `fleurs60-spoken-v1`, with raw references
preserved. This changes scoring references, not audio or model inputs. Historical
hash-bound results are unchanged and require a separately named re-evaluation.
See [policy, inventory and reproduction](reference-normalization-fleurs60.md).

## Readiness fixes (2026-11-04; current status)

**Offline readiness: COMPLETE / READY.** Model execution remains deliberately
UNVERIFIED; that is outside offline completion, not a remaining offline blocker.
Use `/home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/.venv-benchmark/bin/python`.
This Python 3.14 overlay is **not fully isolated**: its explicit
`lib/python3.14/site-packages/project-torch-bridge.pth` appends the existing
`.venv/lib/python3.14/site-packages` absolute path. Existing .venv was not changed.
Local packages take precedence: Transformers4.57.6, tokenizers0.22.2,
huggingface-hub0.36.2. Bridged imports verified: Torch2.14.1,
Whisper20250625, PyVi0.1.1, NumPy2.5.3, soundfile0.14.0, jiwer4.0.0.
CUDA availability true; no CUDA computation or models loaded. Overlay size145MiB;
network transfer size was not measured. Only three binary wheels installed with
`--only-binary=:all: --no-deps`; no Torch/CUDA weights or packages downloaded.
Python3.12.14 is also available, but cannot share Python3.14 binary packages.

Recreate at this workspace path with `python -m venv .venv-benchmark`, install
`requirements-benchmark.txt` with the above flags, then create the explicit .pth
bridge described above. `requirements-benchmark.txt` covers only the overlay;
`requirements-benchmark-effective.lock` captures all138 effective installed
distribution versions, including inherited packages and pip25.1.1, from this
Python3.14.8 Linux interpreter's `pip freeze --all`. Captured2026-11-04,
sanitized to name/version pins only; zero non-pin entries were omitted. Effective
metadata resolution selects the local override when base and overlay both contain
a distribution. No direct/editable URLs, credentials or machine-specific executable
URLs are persisted.

Bridge provenance: source `.venv/lib/python3.14/site-packages`, destination
`.venv-benchmark/lib/python3.14/site-packages/project-torch-bridge.pth`. The local
`.pth` contains the workspace's absolute source path; recreate it for the target
workspace rather than copying that absolute path to another machine. Base versions
shadowed by the overlay are Transformers5.20.2, tokenizers0.23.3 and
huggingface-hub2.1.1; they remain untouched. This is an explicit shared-base overlay,
**not dependency isolation**. The effective lock is an inventory/version lock,
not a wheel-hash/OS-library lock or a claim that a fresh dependency solve succeeds.
Do not install the full lock during modest setup: it includes the already-provisioned
large Torch/CUDA stack. Reuse a matching provisioned base and install only the three
overlay pins. No further packages were installed during this completion pass.

Offline provenance/version validation (does not import model libraries):
```sh
.venv-benchmark/bin/python tools/check_benchmark_environment.py
```
The checker captures `pip freeze --all` privately, permits only name/version pins,
uses effective metadata versions, compares all138 pins, and verifies Python and
the explicit bridge. Raw installer/freeze diagnostics are never printed.
Import checks do not establish that future model loading works with this runtime.
Installed-ranker regression also passes2 tests (0.000s test execution), using
synthetic token lists only.

New artifacts hash-check configuration and record contents. Decode protocol includes
checkpoint SHA, beam/language/task/temperature/timestamps/precision/ranker/mel
semantics and Python/Torch/Whisper/NumPy versions. Scores carry that protocol plus
LM/tokenizer IDs and immutable requested revisions, scoring version, segmentation
and LM runtime versions. Validation tuning freezes invariant protocol independently
of split manifests and timing. Evaluation requires that each LM matches its own
validation protocol and both LMs share the acoustic protocol and N-best file.
Legacy/missing/tampered configuration fails closed: regenerate decode, both scores,
and validation-only tuning; never tune test. Strict schema validation requires all
protocol fields and nested option/runtime fields and rejects unknown fields,
invalid SHA/revision lengths, wrong types (including boolean beam sizes), invalid
beam limits, mutable revisions, empty versions, unsupported fixed decode semantics
and unknown segmentation. Scoring identifiers must contain an explicit positive
implementation version; validation/test equality is exact, including runtime
versions and tokenizer revisions. Recomputed fingerprints do not bypass schema or
cross-run/shared-acoustic comparisons. Hashes detect corruption, not a malicious
author who recomputes every hash. Human transcript audit remains pending.

PhoBERT total encoded length (including specials) is bounded by
`min(tokenizer.model_max_length, max_position_embeddings - pad_token_id - 1)`.
Special tokens are excluded from PLL targets, not removed from conditioning.
Exact boundary accepted, overflow rejected; no truncation.

Current completion verification: benchmark-interpreter isolated unittest RED15/one
missing-schema-helper error (0.047s), then GREEN15 (0.075s); system Python same
file GREEN15 (0.066s). Coverage includes every missing nested protocol field,
wrong types/values, recomputed checkpoint/beam/acoustic-runtime mismatches both
between LMs and validation/test, tokenizer ID/revision/scoring/LM-runtime mismatch,
configuration and frozen-alpha tampering, legacy artifacts and happy validation/test
evaluation. Environment lock checker passes138 pins in0.431s. Installed-ranker
regression passes2 in0.000s test execution (Whisper import/startup excluded).
Compile, CLI help and diff whitespace checks pass. No inference/model loading or
downloads occurred. Existing data was not modified or reacquired.

Prior readiness-pass manifest validation:60 native30/30,15.24–29.94s,
1236.24s total; existing partitions exactly equal manifest filters. The mandatory
full default-Python discovery previously ran:
26 discovered,17 passed,9 dependency import errors (0.024s), missing Torch,
soundfile, gradio or jiwer. These are not model failures. Historical results below
describe earlier revisions and are not current readiness evidence.

## Longer-original replacement completed (2026-11-04)

Actual FFmpeg inspection and replacement retained native validation30/test30.
All 39 originals below15s were replaced:26 with20–30s originals and13 with
15–20s originals. All21 existing>=15s rows, IDs and files were preserved.
No concatenation, truncation, credentials, terms acceptance or inference used.

| Split | Before <15 / 15–20 / >=20 | After <15 / 15–20 / >=20 | Before min/max/mean/total seconds | After min/max/mean/total seconds |
|---|---|---|---|---|
| validation | 17 / 3 / 10 | 0 / 16 / 14 | 6.90 / 29.94 / 15.516 / 465.48 | 15.24 / 29.94 / 20.378 / 611.34 |
| test | 22 / 7 / 1 | 0 / 7 / 23 | 6.60 / 22.56 / 12.892 / 386.76 | 15.30 / 25.32 / 20.830 / 624.90 |

Overall: min6.60/max29.94/mean14.204/total852.24s became
min15.24/max29.94/mean20.604/total1236.24s. Zero short clips remain.

Executed three bounded official immutable TSV/tar scans:

```sh
python tools/long_audio_benchmark.py replace-short
python tools/long_audio_benchmark.py replace-short --max-bytes 536870912 --seconds 240
python tools/long_audio_benchmark.py replace-short --min-duration 15 --max-bytes 268435456 --seconds 240
```

First:268,435,456 bytes/68.847s, scanned361 validation and84 test members,
8 replacements; byte budget exhausted (reported RuntimeError).
Second:441,870,121 bytes/107.560s,361 validation and350 test members,
18 replacements. Third:102,073,561 bytes/30.406s,174 validation members,
13 replacements. Total812,379,138 response-body bytes,10 requests,206.813s
recorded acquisition/validation elapsed across three runs. Each run <=240s;
budget expansion was bounded512MiB, not a whole-dataset download.
TSV contains no authoritative duration/sample-count column, so actual decoded
archive-prefix durations were used. Validation preferred scan reached its archive
end361 members; test stopped at quota. No global maximum claim.

`replace-short` defaults to minimum20s; explicit minimum15s is a second pass
after preferred acquisition. It resumes validated staged candidates, skips active
identities/original IDs, and ranks available candidates20+ first then descending
duration. It retains all superseded audio for rollback. Before any active manifest
update, it validates the complete proposed60 rows (full FFmpeg decode, file/PCM
hashes, IDs and cross-split normalized transcripts), and checks unchanged split
counts. Active manifest replacement is atomic; partitions are then regenerated
atomically per file (not a multi-file transaction).

Ignored provenance: `data/benchmark/long-audio/replacement-provenance.json`
contains latest pass; all three pass reports and original manifests/partitions
are retained at `replacement-backups/{1793746871837140826,1793746959492519032,1793747082868629642}/replacement-plan.json`.
Each report records old/new IDs, durations, paths, immutable revision, bounds,
scan statistics and validation. Original reports/retry provenance are backed up
unchanged; historical acquisition report is not the current duration report.
New active audio resides in `replacement-staging/`; do not delete that directory.
Restore the first backup manifest and partitions to roll back all replacements;
all original files still exist. Transcript human audit remains pending; speaker
identity and training independence remain unknown.

Offline filename-limited smoke demonstrates RED (missing replacement planner)
then GREEN; coverage includes preference, unchanged eligible rows/splits,
incomplete candidates and failed validation leaving active manifest untouched.
No full suite or model inference is required for this acquisition-only change.

## Completed authorized retry (2026-11-04; supersedes historical status below)

**Acquired and independently validated TOTAL60 FLEURS vi_vn clips: native
validation30/test30.** Existing10 validation clips preserved;50 new clips.
viVoice0/0 and VietSpeech0/0: credential access to official shard HEAD succeeded,
but bounded viewer acquisition failed HTTP502 and HTTP500 respectively. These
are service failures, not access denials. No terms accepted or bypassed.

The corrected credential path is `/home/hodangthaiduy/.config/opencode/huggingface.env`.
Its literal `HUGGING_FACE_API_TOKEN` key is now supported. No token output or
persistence. Commands actually executed:

```sh
python tools/long_audio_benchmark.py acquire --env-file /home/hodangthaiduy/.config/opencode/huggingface.env --sources capleaf/viVoice NhutP/VietSpeech --scan-rows 300 --max-bytes 268435456 --seconds 120
python tools/long_audio_benchmark.py acquire --env-file /home/hodangthaiduy/.config/opencode/huggingface.env --sources google/fleurs --per-split 30 --official-fleurs --max-bytes 268435456 --seconds 240
python tools/long_audio_benchmark.py validate --manifest data/benchmark/long-audio/manifest.json
python tools/long_audio_benchmark.py partition --manifest data/benchmark/long-audio/manifest.json --output data/benchmark/long-audio/splits
```

Gated retry:325,483 successful GET-body bytes,46.788s,10 requests. Initial retry
hit a duration-helper shadowing bug (fixed):174,457 bytes,5.902s. Fallback:
34,642,421 bytes,11.302s,5 requests, within256MiB/240s; no budget expansion.
Counts exclude headers/error bodies/TLS. Durations actually decoded with FFmpeg:
**min6.60s/max29.94s/total852.24s**. File/PCM hashes, duplicate IDs/audio and
cross-split normalized text checks passed. No global duration maxima inferred.

Official fallback reads immutable `data/vi_vn/dev.tsv` and `test.tsv`, joins
filenames to their original native splits/IDs, and streams gzip/tar prefixes
from official `audio/dev.tar.gz` and `audio/test.tar.gz`. Closes each stream at
quota, without whole-shard downloads or filesystem archive extraction. Scanned
22 validation/32 test audio members. Unlike viewer seeded preference sorting,
this bounded fallback selects archive order; shorter valid clips are explicitly
labeled fallback, without claiming preferred clips elsewhere are exhausted.
`--scan-rows` bounds examined members; `--sources` and `--per-split` configure
targets. `complete` means selected source/split quotas, not a three-source mix.

Artifacts (ignored): `data/benchmark/long-audio/{manifest.json,acquisition-report.json,retry-provenance.json,splits/validation.json,splits/test.json}`.
Manifest includes official immutable revision, archive/member, transcript,
duration and hashes. Historical source statuses are retained in retry provenance.
No speaker independence or training independence claim; transcript human audit
pending. Under30 selection satisfies under60 acquisition but **does not support
30–60s inference** or establish the long semantic-context hypothesis. No model
downloads/inference/training were run. No balanced three-source comparison.

Verification: standard filename-limited unittest runner (below), RED9 tests/one
error before credential-key support, GREEN10 tests in0.031s including archive
selection/leak rejection, target configuration, literal credential parsing,
budget/redirect and pipeline tests. Validation/partition, acquire help,
`python -m py_compile tools/long_audio_benchmark.py tests/test_long_audio_smoke.py`
and `git diff --check` passed. No full suite rerun: this retry's mandatory scope
is isolated smoke; prior full-suite dependency failures remain historical.

## Historical authenticated follow-up (wrong path; superseded)

Added `acquire --env-file PATH`: literal KEY=value, optional export/quotes,
no shell execution/interpolation. Supported keys: HF_TOKEN,
HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACEHUB_API_TOKEN.
Explicit file credentials take precedence; missing/unreadable files fail closed.
Bearer headers are restricted to huggingface.co/datasets-server.huggingface.co;
cross-host redirects strip credentials. No credentials are persisted.

Actual retry:
`python tools/long_audio_benchmark.py acquire --env-file /home/hodangthaiduy/.config/opencode/hugging-face.env --scan-rows 300 --max-bytes 67108864 --seconds 180`.
The authorized path does **not exist** in this execution environment (existence
and readability checks both false). Acquisition stopped before any network
request; no new audio or metadata acquired and the prior report was preserved.
Authenticated access state remains **unknown**, not denied. Provide/mount that
file at the authorized path to continue; do not paste credentials into chat.
Official targeted fallback and authenticated duration investigation remain
unimplemented/unverified pending context/access; this is not task completion.

Existing manifest revalidation and partition succeeded: 10 FLEURS validation,
0 FLEURS test, 0/0 each other source; decoded range 21.36–29.94 seconds.
Requested target remains 60, 20/source, 10/source/split. Viewer sorting by channel
does not establish a duration maximum. Duration metadata now recognizes FLEURS
num_samples / 16000 versus numeric duration in seconds; new bounded observations
are explicitly marked not global caps. No newly observed distributions exist.
Short fallback clips would form a short-clip stratum, not evidence for the long
semantic-context hypothesis; other datasets may be needed for natural long clips.

Offline RED: 8 tests, 2 missing-helper errors, 0.026s. GREEN: 8 tests passed,
0.024s. Isolated unittest discovery is the standard runner and filename-limited.
Acquire help, compile and git diff --check passed. No model/full-suite runs in
this follow-up; no external services mocked as production behavior.

Implementer: `ses_e5fd1c4adffeJqsok45vkEZ8vW` (openai/gpt-6.1-sol).

## Historical acquisition status (superseded)

This is **partial preparation, not a completed 60-sample benchmark**. No model
inference, model download, training, or accuracy claim was made.

| Source | Validation acquired | Test acquired | Blocker |
|---|---:|---:|---|
| capleaf/viVoice | 0 | 0 | Official shard HEAD: HTTP 401; gated access required |
| google/fleurs, vi_vn | 10 | 0 | Test `/rows` repeatedly HTTP 500; `/first-rows` HTTP 501 |
| NhutP/VietSpeech | 0 | 0 | Official shard HEAD: HTTP 401; gated access required |

Two attempts used `--scan-rows 300 --max-bytes 67108864 --seconds 180`.
Attempt 1: 15,936,815 response bytes, 59.43 seconds; attempt 2 resumed without
redownloading selected clips: 725,995 response bytes, 10.66 seconds. These byte
figures count successful GET bodies, not HTTP headers/error bodies/TLS traffic.
The subsequent implementation also counts gated HEAD requests in its request
counter. Ten complete audio files were decoded by FFmpeg and checked: durations
21.36–29.94 seconds, **10 preferred, 0 fallback**, no detected ID/audio/normalized
transcript leakage. Files and machine-readable manifest/report are ignored under
`data/benchmark/long-audio/`; the report describes the latest attempt, not cumulative
network usage. Rerun `validate` for independently recomputed file/PCM hashes and
durations. Transcripts are **not human validated**. Speaker/recording identity is
unavailable for these FLEURS rows. Training-data independence is **unknown**.

Source revisions observed:

- viVoice: `693e9882c8e73b31e96a723741fa1861c3252b75`, CC-BY-NC-SA-4.0
- FLEURS: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0
- VietSpeech: `0f18688afa9b64ea515d3fdaad850999b86a0f52`, Apache-2.0

Respect each dataset's additional access and research-use terms. The tool does
not accept terms, submit account details, or use mirrors. It checks `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, then the existing HF token file, without printing tokens.
Obtain access manually from the official dataset pages if desired; rerun with
already authorized credentials. If an authorized source's viewer cannot provide
rows, acquisition reports a blocker rather than fetching a multi-GB shard.

## Acquisition and selection

From the repository root, Python standard library + FFmpeg suffice:

```sh
python tools/long_audio_benchmark.py acquire --scan-rows 300 --max-bytes 67108864 --seconds 180
python tools/long_audio_benchmark.py validate --manifest data/benchmark/long-audio/manifest.json
python tools/long_audio_benchmark.py partition --manifest data/benchmark/long-audio/manifest.json --output data/benchmark/long-audio/splits
```

Selection examines a bounded metadata pool per original split, shuffles with
seed 42, then stably prioritizes 20 <= duration < 30 seconds. Positive shorter
clips are allowed only as recorded fallbacks. Actual decoded duration overrides
advertised duration. **This implementation explicitly restricts the benchmark
to under 30 seconds**, a permitted fallback to avoid Whisper's 30-second context
truncation. It cannot measure 30–60-second behavior; no chunking/concatenation is
performed. It does not establish that preferred clips are globally exhausted,
only that the configured bounded pool is exhausted.

FLEURS uses native validation/test splits. Other sources use a deterministic
channel-group hash for validation/test when channel metadata exists, otherwise
row identity with an explicit missing-group limitation. IDs are original split
plus viewer row index, with original dataset IDs retained separately. Duplicate
file/decoded hashes and cross-split normalized text, IDs, and available groups
are rejected. This cannot prove speaker independence without speaker metadata.
Manifest entries contain immutable revision, original split/ID, source URL and
license, reference, local audio path, duration, file/decoded SHA256, selection
status, seed and audit status. Mutable viewer revisions differing from official
metadata are refused. Resuming also refuses source revision changes: use a new
output directory rather than silently mixing revisions.

Network operations have timeouts and a shared response-body/time budget; no
whole dataset or model is fetched. A failed source does not masquerade as success.
The ignored report's `complete` flag remains false until all 60 rows exist.

## Explicit benchmark stages (run only when ready)

Use the benchmark overlay environment described above. Optional dependencies are in
`requirements-benchmark.txt`; do not install a new CUDA stack just to run tests.
Missing packages produce an actionable CLI error. The existing converted local
checkpoint is loaded through `src.inference`; `PHOWHISPER_MODEL_PATH` is supported.
Model inference below is intentionally **not run during setup**.

Use the concrete immutable HF commit SHAs below, never `main`. Validate/partition the combined manifest
first. Use the same commands separately for validation and test, saving outputs
under different directories. Example for validation:

```sh
python tools/long_audio_benchmark.py decode --manifest data/benchmark/long-audio/splits/validation.json --output outputs/benchmarks/long/validation/nbest.json --beam-size 5
python tools/long_audio_benchmark.py score --input outputs/benchmarks/long/validation/nbest.json --output outputs/benchmarks/long/validation/gpt2.json --lm gpt2 --revision 65818d14816b42be09e2201933bf07106d9a2647
python tools/long_audio_benchmark.py score --input outputs/benchmarks/long/validation/nbest.json --output outputs/benchmarks/long/validation/phobert.json --lm phobert --revision 86cd7fd4c148980922ac11a2cf5e257f2ba639e1
python tools/long_audio_benchmark.py tune --manifest data/benchmark/long-audio/splits/validation.json --input outputs/benchmarks/long/validation/gpt2.json --output outputs/benchmarks/long/gpt2-frozen.json
python tools/long_audio_benchmark.py tune --manifest data/benchmark/long-audio/splits/validation.json --input outputs/benchmarks/long/validation/phobert.json --output outputs/benchmarks/long/phobert-frozen.json
```

Generate `test/nbest.json`, `test/gpt2.json`, and `test/phobert.json` using the
three decode/score commands above with `validation` changed to `test`, **without
running tune on test**. Then:

```sh
python tools/long_audio_benchmark.py evaluate --manifest data/benchmark/long-audio/splits/test.json --gpt2 outputs/benchmarks/long/test/gpt2.json --phobert outputs/benchmarks/long/test/phobert.json --gpt2-config outputs/benchmarks/long/gpt2-frozen.json --phobert-config outputs/benchmarks/long/phobert-frozen.json --output outputs/benchmarks/long/test/results.json
```

The current60 rows support the staged FLEURS evaluation, not a balanced
three-source comparison. Do not proceed with empty test manifests.

## Protocol and artifacts

- Decode runs deterministic Vietnamese transcription (`temperature=0`, no
  timestamps, no conditioning on references). One beam pass persists candidate
  tokens, IDs, text, sum log probability, and average acoustic log probability;
  a separate greedy pass uses the same audio and language/task options.
- Beam Top-1 uses the installed Whisper MaximumLikelihoodRanker, default
  `sum_logprob / number of generated tokens` (prefix and EOS excluded from
  token count; cumulative probability includes EOS). The separate acoustic
  average for rescoring remains `sum_logprob / (generated tokens + 1)`. Both LMs
  rank those **same persisted candidates**, never independently decode them.
  Beam output is the completed candidate set exposed by the existing Whisper
  decoder, not a guarantee of exactly five unique strings.
- The scorer sees candidate text, not reference transcripts. GPT2 uses mean
  next-token log probability with an EOS prefix. PhoBERT uses PyVi word
  segmentation and mean single-masked-token PLL, not causal loss. Context
  overflow raises an error rather than silently truncating LM input. CPU scoring
  is deliberate for predictable optional execution; it is not performance tuned.
- Select `acoustic_average + alpha * LM_average`. Validation-only grid
  `[0, .05, .1, .2, .4]` minimizes total word edits, ties favoring smaller alpha.
  No test tuning, length bonus or confidence gate. Frozen files record model
  identity, revision and validation artifact hashes. Tuning refuses any test row.
- Evaluation checks manifest hashes, sample/split identity, frozen model
  identity, shared N-best file hashes, and candidate IDs/text/acoustic scores.
  Pooled and per-source WER/CER use corpus edit totals, with NFC/lowercase,
  punctuation-to-space and whitespace normalization. CER includes word spaces.
  Oracle N-best minimizes word errors and is a diagnostic, not a deployable
  selection rule. Per-sample normalized hypotheses/references and decode/LM
  times are persisted. Decode timing includes greedy plus beam; LM timing is
  separate. Model loading is excluded; these are not end-to-end latency claims.
- Artifacts record manifest, configuration, local checkpoint and N-best hashes,
  LM IDs/immutable revisions and the resolved model revision. The local
  checkpoint hash, not the nominal source name, is authoritative when overridden.
  Existing inference/rescorer APIs remain unchanged.

## Verification

`python -m unittest discover -s tests -p test_long_audio_smoke.py -v` selects only
the named file using standard unittest discovery (no custom runner). It tests
selection/edits, leakage/test-tuning rejection, shared-candidate frozen evaluation,
network byte budget, credential-safe redirects and duration/hash validation. External HTTP/audio operations
are mocked where isolated; production selection/evaluation logic is not mocked.
Initial RED: two missing-module errors (0.001 seconds), then GREEN after implementation.
Final isolated run: **6 passed**, 0.026 seconds test execution, 0.115 seconds wall.
Actual acquired-manifest validation passed in 1.360 seconds wall; partition and
compile checks also passed. Top-level/acquire/decode/score/tune/evaluate help
commands and `git diff --check` passed. The decode command using the actual
manifest was explicitly attempted: it exited 2 after 1.282 seconds with the
expected actionable missing-`torch` message, before loading any model.

The handoff mandates `python -m unittest discover tests -v`; this full discovery
was run despite the usual isolated-only testing rule. Final run: 19 tests discovered,
10 passed, 9 import errors (0.017 seconds test execution; 0.249 seconds wall)
in default Python 3.14.8, missing
`torch`, `soundfile`, `gradio`, or `jiwer`. These include unrelated existing
tests. No claim is made about model execution or a fully passing project suite.
