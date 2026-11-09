# Personal audio and frozen VietMed100 comparison — 2026-11-04

## Result and conservative decision

Actual offline inference completed. **Keep production unchanged; preserve long beam15 as opt-in.** No restore was necessary: production was already the original workflow. Long has two more pooled word errors and 36 more character errors on these same 100 recordings, and is about 3.01 times slower by measured processing time. No tuning, training, downloads, package installation, checkpoint mutation or commits occurred.

Artifacts (relative to repository root):

- Full private UTF-8 transcript: `outputs/benchmarks/personal-vietmed-compare-20261104T035943Z/personal-transcript.txt`
- Timestamped chunks, selected ranks and all candidates: same directory, `personal-segments.json`
- Personal measurements: same directory, `personal-timing.json`
- Complete measured comparison: same directory, `report.json`
- All 100 predictions/candidates per profile: same directory, `original-production.jsonl` and `current-long.jsonl`
- Per-sample word-error comparison: same directory, `paired-comparison.json`
- Saved-result recomputation, stage sums and coverage audit: same directory, `independent-audit.json`
- Versioned profiles: `configs/experiments/personal-vietmed-20261104T035943Z/current-long-beam15.json` and `original-production.json`
- Original source copies/hashes and checkpoint metadata: `outputs/config-snapshots/20261104T035943Z/`. Weights were not duplicated.

Exact transcript and segment paths are ignored, verified with `git check-ignore`; no personal transcript is included here. Snapshot/config files contain no personal transcript or credentials. New profile files are preservation records, not an automatic application configuration loader.

## Profiles and checkpoint

Checkpoint `models/phowhisper-base.pt`, 290454061 bytes, SHA-256:
`e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d`.
Before/after checkpoint and original named source hashes match. Existing user changes were preserved.

**Original production rescoring:** beam5, FP16 acoustic decoding, descending sum-logprob candidate order, alpha .2, beta -.02 per normalized whitespace word, gamma -.2 acoustic margin, confidence gate -.020. GPT2 negative mean causal loss on production-normalized text, tokenizer encoding defaults, no explicit model-config prefix. The application's default remains Fast (`rescore=False`); this comparison's baseline is the original Accurate/rescoring workflow, not Fast.

**Frozen long experiment:** beam15, FP32 acoustic decoding, installed Whisper MaximumLikelihoodRanker length ranking, acoustic score `sum_logprob / (tokens + 1)`, alpha .3, beta -.02 per normalized whitespace word, no confidence gate or acoustic margin. Long normalization is NFC/lowercase, replace non-alphanumeric characters with spaces, collapse whitespace. GPT2 scores every text token after explicit model-config BOS 50256, no automatic special tokens; mean loss, no context truncation or cross-chunk context. Exact protocol/runtime/vocabulary fingerprints, validation-selection metadata, winner fingerprint, source hashes and candidate provenance remain in the profile and original frozen artifacts.

GPT2/tokenizer revision: `65818d14816b42be09e2201933bf07106d9a2647`. Frozen winner/freeze hash and fingerprints, checkpoint hash, prefix policy/ID and tokenizer vocabulary hash were checked. No validation search was repeated. These are **whole-workflow comparisons**, changing beam, precision, ordering, normalization, prefix and selection/gating together, not weight optimization or a causal estimate of any single change. Both new runs used GPT2 CPU FP32 to match the frozen long experiment; the original production API normally places GPT2 on CUDA. Baseline scoring/gating was mirrored explicitly in the new runner without fallback. All 100 baseline hypotheses exactly match the archived production-API predictions, independently checked after the run.

## Personal recording: complete coverage and timing

Exact input: `data/benchmark/my-audio/ThuanLoi5.m4a`; actual existence checked with ffprobe, not glob. AAC, 48000 Hz, mono. Container duration **84.714667 seconds**. Input SHA-256 `bd2f964f7485414478f0e28bd2f0cb53456cacec4e3745bbef99fbe909cde8af`.

FFmpeg/Whisper loaded the complete file once as 16 kHz mono: **1355435 samples**, 84.7146875 seconds (container/decoded rounding differs by 0.0000205 seconds). Deterministic chunks: `[0,28)`, `[28,56)`, `[56,84)`, `[84,84.7146875)` seconds. Every decoded sample covered exactly once, with no overlap or dropped tail. Each chunk uses the frozen long workflow independently. Chunk boundaries are not word timestamps or silence boundaries; they may split words. Even the final 0.715-second tail was processed. No VAD, new context policy or hidden parameter search was added.

| Measured component | Seconds |
|---|---:|
| Complete audio load/resample | 0.160643 |
| Chunk mel preprocessing | 0.078840 |
| Acoustic decode | 10.057953 |
| GPT2 scoring and selection | 8.491728 |
| Sum processing | **18.789164** |
| Personal processing wall including chunk/result-write overhead | **18.931727** |
| Whisper model load | 1.112111 |
| Tokenizer load | 0.097127 |
| GPT2 CPU model load | 0.153121 |

Warm processing RTF = **0.221794**, using the original container duration. Warm measured-wall RTF = **0.223476**. Accounted cold wall (personal processing wall + measured model/tokenizer loads) = 20.294087 seconds, RTF **0.239558**. This is not an instrumented whole-process cold start: Python imports, source/checkpoint hashing, configuration snapshots and ffprobe are excluded. The initial timing JSON's `cold_rtf=0.237875` is the **stage-sum-plus-load** RTF, not cold wall; the separately audited wall figures above make this distinction explicit. Personal chunk execution had no excluded warmup and includes first-use overhead. CUDA synchronization surrounds measured operations, including CPU LM work. Overall comparison process wall was 472.076 seconds, not personal latency.

Transcript correctness is unknown: there is no reference and no personal WER/CER claim. The full transcript is saved privately at the path above, not silently truncated to 30 seconds.

## Frozen VietMed100

Exactly 70 train + 15 dev + 15 test clips, **625 seconds**, **2528 normalized words / 10630 characters** (spaces included). Evaluated the original base checkpoint only; no fine-tuned weights loaded. All split JSON/CSV hashes equal the historical frozen experiment, every audio hash/duration/16kHz-mono format validated, and normalized speaker/sequence/audio-name/utterance/source identities are disjoint across splits. Per-profile order, IDs, splits and reference texts independently match the manifests. This is the existing selected 100-clip set, not a new untouched generalization test; 70 clips were used in a prior separate fine-tuning experiment. Dataset provenance from the earlier handoff pins VietMed revision `dc1a2ebef3ffa772a955f2c9b4412ad8722c0b74`; this run verified local frozen content, not a remote repository.

JSON SHA-256:

- train: `6b345395a68dbab191aef782279371f087aa5e05cbb6313ba0c9c31605487635`
- dev: `1c582d9a28e20f0ff36148a956f582b9ebfbde172284bd0a8b734dfecd1d9712`
- test: `b1bdb4ddc3362fc7511595344fe3d0f118cd54f13ed525f99512f1f0269bfc80`

CSV hashes are recorded in `report.json`. No label or manifest edits and no FLEURS numeric annotations were used. Both profiles' metrics use `benchmarks.metrics.normalize_vietnamese`: NFC/casefold, punctuation to spaces, whitespace collapse, accents retained, no tone harmonization. Pooled errors are divided by pooled reference lengths, not averaged utterance rates.

| Actual workflow | WER % | CER % | Word S/D/I | Character S/D/I | Processing seconds | RTF |
|---|---:|---:|---|---|---:|---:|
| Original production | **18.117089** | **13.189087** | 243 / 115 / 100 | 437 / 531 / 434 | 109.134958 | **0.174616** |
| Frozen long beam15 | 18.196203 | 13.527752 | 224 / 138 / 98 | 399 / 619 / 420 | 329.014593 | 0.526423 |

Long minus original: **+0.079114 WER percentage points**, **+0.338664 CER percentage points**. Word errors 458 → 460; character errors 1402 → 1438. Paired utterance word-error outcomes for long: **23 wins, 60 ties, 17 losses**; number of wins alone is not pooled WER. Decision was predeclared: retain original if either metric worsens. Both worsened; long remains an explicitly separate opt-in experiment.

| Stage, summed seconds | Original | Long |
|---|---:|---:|
| Audio load/I/O | 15.074443 | 19.219720 |
| Mel preprocessing | 0.792103 | 0.927492 |
| Acoustic decode | 51.479935 | 129.494418 |
| LM scoring/selection | 41.788478 | 179.372962 |

One excluded dev[0] warmup per profile: 1.370184 / 3.088557 seconds. Profile stage wall including warmup and writes: 110.808822 / 332.390403 seconds. Processing RTF includes audio load, mel, decoding and scoring; excludes model load, manifest integrity audit, warmup and JSON output overhead. Models loaded once before both profiles. Sequential batch1, no concurrent model jobs. Each profile stage guarded at 1800 seconds and complete command bounded to 1800 seconds.

Archived `outputs/benchmarks/vietmed-local-20261025T233412Z/report.json`, `base-gpt2-all100`: WER 18.117089%, CER 13.189087%, RTF .188928, 118.079730 seconds. This archived baseline is 100 clips, unlike the separate 15-sample `vietmed_benchmark_results.json`. Fresh baseline reproduces all 100 archived hypotheses exactly. Timing is not directly controlled across dates: archived API used GPU LM/4 threads versus current CPU LM/2 threads, with different caches/warm state. No speed improvement over the archive is claimed. The historical Fast result is distinct: WER 19.620253%, CER 13.668862%, RTF .115390; it was not freshly rerun here.

## Commands, environment and verification

Main actual command:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 1800 .venv-benchmark/bin/python -m tools.personal_vietmed_compare
```

Completed in 472.076 seconds including snapshots/integrity/loading. New invocation creates a fresh UTC directory and refuses existing output/profile/snapshot directories. Audit, no model rerun:

```sh
.venv-benchmark/bin/python -m tools.audit_personal_vietmed_compare
```

NVIDIA GeForce RTX3050 Laptop GPU, CUDA13.0; Python3.14.8, torch2.14.1, Whisper20250625, transformers4.57.6, tokenizers0.22.2, numpy2.5.3. Metadata-only `tools/check_benchmark_environment.py` passed all **138 pins in .416 seconds**. GPT2 printed its standard default causal-loss warning, not a model failure/fallback.

New isolated smoke RED: `python -m unittest discover -s tests -p test_personal_vietmed_compare.py -v` failed with missing module (0.000s). GREEN after implementation: 2 passed (0.000s); final extended tests 3 passed (0.000s), covering exact sample coverage/tail, RTF, conservative CER regression decision, sample identity rejection and profile seal/tampering. Standard unittest `-p` restricts filename collection; no project wrapper expands execution.

Final isolated commands use `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v`:

| Exact FILE | Result | unittest duration |
|---|---|---:|
| test_personal_vietmed_compare.py | 3 passed | .000s |
| test_gpt2_prefix_smoke.py | 2 passed | .000s |
| test_nbest_ranker_smoke.py | 2 passed | .000s |
| test_vietmed_local.py | 2 passed | .001s |
| test_section23_smoke.py | 2 passed | .001s |

Durations are runner-reported, excluding interpreter/import startup. Audit independently recomputed saved WER/CER with jiwer, verified all IDs/references, output hashes, processing stage sums and personal sample coverage, and compared baseline against archived actual hypotheses. Audit standalone elapsed time was not instrumented. `python -m py_compile tools/personal_vietmed_compare.py tools/audit_personal_vietmed_compare.py tests/test_personal_vietmed_compare.py` and `git diff --check` passed (individual wall durations not measured). Full/e2e suites skipped: current scope requests isolated verification; no existing production tools changed. No download/training tests invoked.

New files: two scoped runner/audit modules, one portable smoke file, this report and two versioned profile JSON files. Snapshot directory contains copies of the five named original source files and `snapshot.json`. Existing tools/data, `.gitignore`, unrelated dirty files and all preexisting outputs were left intact. No blockers remain for the requested actual runs; limits are small preselected corpus, no personal reference, deterministic boundary artifacts, and multi-factor workflow comparison rather than a single-factor ablation.
