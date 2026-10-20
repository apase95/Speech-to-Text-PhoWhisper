# Exact 1000 TRAIN / 200 validation pilot

This additive workflow does not change `train.py prepare` (the full byte-audit
default remains available). `pilot.py` reads metadata first, normalizes speaker
IDs with NFC/casefold/whitespace collapse, reserves whole speakers until at least
200 eligible validation rows exist, then round-robins shuffled speaker buckets.
Seed is 42. There is no clip-random fallback and no guarantee that every speaker
is represented. Missing speakers/empty transcripts are excluded and counted;
conflicting duplicate IDs fail. Identical duplicate metadata rows collapse.

Exactly 1200 selected files are opened, decoded and hashed; invalid selected
audio, token length >448 or duplicate/excluded hashes fail rather than silently
replace examples. Supplied benchmark/dev IDs and hashes and all official test
IDs are excluded. Preparation records metadata/manifest/exclusion hashes,
speaker counts and duration totals. This is **subset-only** auditing: no global
audio-byte audit, near-duplicate audit, official-test hash audit, speaker-alias
resolution, or guarantee against model pretraining exposure.

## Build / local validation

```bash
python -m unittest kaggle_training.test_pilot -v
python -m kaggle_training.build_pilot_source
python kaggle_training/pilot1000_source.py --verify-only
```

`pilot1000_source.py` embeds the explicit source allowlist and both complete
exclusion CSVs. It contains no weights, tokens, GitHub dependencies or Kaggle
mutation calls. Regenerate after changing embedded modules/tests. `--verify-only`
extracts to a temporary directory and runs only the two named unittest modules;
it imports no model dependencies and downloads nothing.

## Controller launch (not executed locally)

## Progress and wait diagnosis

The first executed source lines print a flushed UTC `artifact startup start`
marker, before payload evaluation/extraction or model imports. Stage start/done
markers include UTC time and elapsed seconds; failed stages preserve tracebacks
on stderr. Python children use `-u` and `PYTHONUNBUFFERED=1`, without hiding logs.
Runtime status is atomically replaced at
`/kaggle/working/outputs/benchmarks/current-progress.json`, then also at each
run's `progress.json`. These are last-event snapshots, not success guarantees.
Selected-audio audit and baseline/post expose actual processed counts (1200 and
200 respectively); optimizer callbacks report steps out of 125. ETA is null
through five warmup steps, then estimates **remaining optimizer training only**,
not evaluation, checkpoint saving, downloads or total job time. Smoke counts
remain separate. Dependency pip has a 600-second process bound and bounded
retries; HF metadata/read timeouts are 60 seconds, not a total download deadline.
No heartbeat is claimed during a blocking library call: its last start marker
identifies the wait. Kaggle may publish logs/files only later; progress files
may only be downloadable after Kaggle publishes outputs. An absent startup
marker alone cannot distinguish provisioning from delayed log publication.

Create a **new private** Kaggle Python **script** titled `PhoWhisper Pilot1000`,
slug `phowhisper-pilot1000`, language Python, GPU and Internet enabled, dataset
`tuandaodev/viet-bud500-formatted` attached. Inject the complete UTF-8 contents of
`kaggle_training/pilot1000_source.py` as the script source; no checkout/upload
of unpublished code is necessary. Do not edit `hngthiduy/phowhisper2`.

For notebook-type injection, put the source in a single code cell and execute
with a clean argument list (`sys.argv = ['pilot1000_source.py']` before the final
`main()` guard); IPython kernel arguments otherwise confuse argparse. Alternatively
write the embedded script into `/kaggle/working/pilot1000_source.py` and launch
`subprocess.run([sys.executable, path], check=True)` from the notebook.

The confirmed default root is `/kaggle/input/datasets/tuandaodev/viet-bud500-formatted`; if metadata is nested,
provide `--root` with the confirmed directory containing both metadata CSVs and
`wavs/`. No automatic dataset path discovery is performed. The wrapper masks
GPU 0 even for T4 x2; preflight requires one visible CUDA GPU/one process.

The entry installs `jiwer==3.0.5` with the executing interpreter's pip only when
jiwer is missing. Layout and metadata validation precede any HF downloads.
Use the existing image's dependencies first: torch CUDA, Transformers,
accelerate, huggingface_hub, numpy, scipy, soundfile, jiwer. Import/API preflight
records actual versions and fails on missing/incompatible dependencies; it does
not blindly downgrade Python 3.13 or torch. The legacy supported stack is Python
3.10–3.12 / Transformers 4.46.3 / accelerate 1.1.1. Controller must choose/fix
the image if its installed stack fails preflight. Model compatibility, CUDA,
finite gradients/loss and checkpoint advancement are gated by a one-step
two-example smoke and resume to step two before the real pilot. This is an
**advancement-only** check: scheduler horizon changes from 1 to 2, so optimizer
and scheduler continuity are not verified or claimed. Smoke outputs
are separate and never used to initialize the pilot.

HF model `main` resolves once to an immutable commit reused for processor,
baseline, smoke and training. FP32, LR 1e-5, batch 2, accumulation 4, world 1:
1000 examples give 125 optimizer steps (one epoch). Final one-epoch checkpoint
is compared, not selected on benchmark/test results. Validation is frozen for
baseline/post; HF FP32 Vietnamese greedy decoding and benchmark normalization
are shared. Genuine per-clip synchronized timing includes feature preparation
and decoding, not Trainer evaluation runtime. No warmup is applied to either
side. These reports imply no HF/OpenAI `.pt` runtime parity.

Reports and checkpoints persist under
`/kaggle/working/outputs/benchmarks/pilot1000-<UTC>/`: `preflight.json`,
`prepare.json`, train/validation manifests, `smoke_verified.json`, `setup.json`,
baseline/post `summary.json` and `predictions.jsonl`, comparison JSON/Markdown,
and `train-final/` containing HF weights, processor, checkpoint/state,
`run.json`, `metrics.json`, `errors.csv`. `SUCCESS.json` is written only after
completion and zero inference failures; failures write `FAILED.json`. Directory
existence is not success. Baseline checkpoint hash is null with explanation and
immutable HF revision; final model identity hashes artifact names/content.
Fetch authentic reports locally only after the controller's run; do not
manufacture local placeholder metrics. Protect transcript-bearing outputs.

Provenance records the base Git revision separately, with actual Git revision
null for the extracted uncommitted payload, per-file executed-source SHA256,
aggregate SHA256 and implementation ID. RSS is the parent lifetime high-water
mark (including preparation/prior benchmarks), not a per-run comparison.

For source too large for tool text, use the local authenticated Kaggle CLI:
copy the generated artifact to an isolated launch directory as `pilot1000_source.py`,
and create `kernel-metadata.json` there with `id` `hngthiduy/phowhisper-pilot1000`,
`title` `PhoWhisper Pilot1000`, `code_file` `pilot1000_source.py`, `language`
`python`, `kernel_type` `script`, `is_private` true, `enable_gpu` true,
`enable_internet` true, and `dataset_sources`
`["tuandaodev/viet-bud500-formatted"]`. Controller then runs
`kaggle kernels push -p /absolute/path/to/launch-directory` and checks status
with `kaggle kernels status hngthiduy/phowhisper-pilot1000`. Not executed here.

Remaining runtime gates: actual dataset layout/counts/labels, selected audio
validity, dependency/model compatibility, VRAM, finite training and resume,
GPU timing, ASR quality and persistence via Kaggle Save Version. No local model
or full-corpus download is needed. DDP is deliberately unsupported for this pilot.
