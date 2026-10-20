# Isolated Kaggle runtime diagnostic

This standalone script does not install packages, download models, train, or
change pilot parameters. It uses UTC timestamps and independent `perf_counter`
durations. Default stage is `cpu`; default output is
`/kaggle/working/kaggle-diagnostic`. Use a fresh output directory for each run.

Controller must launch **sequentially**, examining authentic logs/results before
advancing. No remote run is claimed by local tests.

| Gate | Private Python script settings | Stage | Stop budget after startup |
| --- | --- | --- | --- |
| 1 | GPU false, Internet false, dataset sources `[]` | cpu | 60 seconds |
| 2 | GPU true, Internet false, dataset sources `[]` | gpu | 180 seconds |
| 3 | GPU true, Internet false, dataset sources `["tuandaodev/viet-bud500-formatted"]` | data | 300 seconds |

CPU deliberately waits 20 seconds, with a heartbeat every 5 seconds. GPU imports
the image's torch in a subprocess and performs a tiny CUDA operation/synchronize;
its hard timeout is 120 seconds. Data repeats GPU preflight, streams both pipe
metadata CSVs, then checks one audio file with soundfile/numpy in another
120-second subprocess. Metadata streaming has no internal deadline; the controller
enforces the stage stop budget. No full audio walk/hash or transcript output.
Speaker summaries contain aggregate counts only. One sample is constrained to
`wavs/audio<ID>.wav`, rejecting traversal and escaping symlinks. This is not a
corpus validity audit. Decode is bounded to 30 seconds / 12 million sample values.

Use a new private slug such as `hngthiduy/phowhisper-diag-cpu-20261015`;
never modify `hngthiduy/phowhisper2` or the pilot notebook. For an isolated launch
directory immediately under `kaggle_training`, metadata can be:

```json
{
  "id": "hngthiduy/phowhisper-diag-cpu-20261015",
  "title": "PhoWhisper diagnostic CPU 20261015",
  "code_file": "../diagnostic.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

MCP script launch can supply the complete standalone `diagnostic.py` text.
For the next gates, prefix source with
`import os; os.environ['KAGGLE_DIAGNOSTIC_STAGE'] = 'gpu'` (then `'data'`)
and adjust the remote settings above; CLI equivalent is `--stage gpu` / `--stage data`.
Notebook injection needs a clean `sys.argv` before invoking `main`, or execution
as a separate Python script. Known root is
`/kaggle/input/datasets/tuandaodev/viet-bud500-formatted`, containing
`metadata_train.csv`, `metadata_test.csv`, and `wavs/`.

First source execution prints a flushed startup marker before third-party
imports. JSONL events are flushed and status/result JSON is atomically replaced.
Heartbeat records include stage, event count and elapsed time; metadata also emits
processed-row counts every 10,000 rows. Child stdout/stderr is captured, printed
when the child exits or times out, and retained as a bounded summary. Parent
heartbeats continue while children block. Exceptions preserve bounded tracebacks.
Threads are joined before the final success/failure event and result file.
An output directory or heartbeat alone is **not success**.

Kaggle may delay publishing both logs and downloadable outputs. An absent startup
marker does not distinguish provisioning from delayed publication. Stop and ask
the controller to inspect/cancel at the above budgets rather than assume a live
GPU hang. Provisioning needs a separate controller wait budget. Do not print
tokens, environment dumps, transcripts, speaker names, or secrets.

Local checks (stdlib only, no local torch needed):

```bash
python -m unittest kaggle_training.test_diagnostic -v
python -m py_compile kaggle_training/diagnostic.py kaggle_training/test_diagnostic.py
python kaggle_training/diagnostic.py --stage cpu --heartbeat-test-seconds 6 --output /tmp/opencode/kaggle-diagnostic-local
git diff --check
```

The explicit unittest module selects only this isolated module using Python's
standard unittest loader, not a project-wide test wrapper. Remote CUDA, installed
soundfile, real metadata layout/counts and Kaggle publication remain remote gates.
