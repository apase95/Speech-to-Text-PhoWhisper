# Human review / cached FLEURS60 benchmark

Reference-only label audit, not independent model improvement. Literal NFC/lower/alphanumeric/space normalization; no dialect collapse; CER includes spaces. No reselection.

Fresh30 and its test:836 parser correction are unchanged; not human-reviewed here.
Only 10 sentences / 11 supplied spans corrected. Other original13 assertion covers 3 cases only. Whole-sentence independent verification remains pending.

| Split | Ref | Method | Words | Characters | S | D | I | Errors | WER % | Char errors | CER % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | A | gpt2_tuned_beam5 | 1267 | 5750 | 187 | 9 | 93 | 289 | 22.8098 | 785 | 13.6522 |
| validation | A | greedy | 1267 | 5750 | 201 | 11 | 78 | 290 | 22.8887 | 975 | 16.9565 |
| validation | A | beam5 | 1267 | 5750 | 201 | 8 | 101 | 310 | 24.4672 | 818 | 14.2261 |
| validation | A | gpt2_original | 1267 | 5750 | 187 | 9 | 96 | 292 | 23.0466 | 790 | 13.7391 |
| validation | A | gpt2_tuned_beam15 | 1267 | 5750 | 183 | 8 | 90 | 281 | 22.1784 | 792 | 13.7739 |
| test | A | gpt2_tuned_beam5 | 1353 | 6085 | 167 | 11 | 57 | 235 | 17.3688 | 654 | 10.7477 |
| test | A | greedy | 1353 | 6085 | 183 | 14 | 66 | 263 | 19.4383 | 688 | 11.3065 |
| test | A | beam5 | 1353 | 6085 | 181 | 11 | 59 | 251 | 18.5514 | 661 | 10.8628 |
| test | A | gpt2_original | 1353 | 6085 | 166 | 11 | 60 | 237 | 17.5166 | 651 | 10.6984 |
| test | A | gpt2_tuned_beam15 | 1353 | 6085 | 161 | 12 | 55 | 228 | 16.8514 | 646 | 10.6163 |
| validation | C | gpt2_tuned_beam5 | 1291 | 5876 | 177 | 9 | 69 | 255 | 19.7521 | 639 | 10.8747 |
| validation | C | greedy | 1291 | 5876 | 191 | 11 | 54 | 256 | 19.8296 | 829 | 14.1082 |
| validation | C | beam5 | 1291 | 5876 | 191 | 8 | 77 | 276 | 21.3788 | 672 | 11.4364 |
| validation | C | gpt2_original | 1291 | 5876 | 177 | 9 | 72 | 258 | 19.9845 | 644 | 10.9598 |
| validation | C | gpt2_tuned_beam15 | 1291 | 5876 | 171 | 9 | 67 | 247 | 19.1325 | 648 | 11.0279 |
| test | C | gpt2_tuned_beam5 | 1387 | 6242 | 160 | 20 | 32 | 212 | 15.2848 | 577 | 9.2438 |
| test | C | greedy | 1387 | 6242 | 178 | 23 | 41 | 242 | 17.4477 | 621 | 9.9487 |
| test | C | beam5 | 1387 | 6242 | 176 | 20 | 34 | 230 | 16.5826 | 593 | 9.5002 |
| test | C | gpt2_original | 1387 | 6242 | 161 | 20 | 35 | 216 | 15.5732 | 583 | 9.3400 |
| test | C | gpt2_tuned_beam15 | 1387 | 6242 | 154 | 21 | 30 | 205 | 14.7801 | 569 | 9.1157 |
| validation | human-v3 | gpt2_tuned_beam5 | 1291 | 5869 | 174 | 9 | 69 | 252 | 19.5198 | 632 | 10.7684 |
| validation | human-v3 | greedy | 1291 | 5869 | 188 | 11 | 54 | 253 | 19.5972 | 822 | 14.0058 |
| validation | human-v3 | beam5 | 1291 | 5869 | 188 | 8 | 77 | 273 | 21.1464 | 665 | 11.3307 |
| validation | human-v3 | gpt2_original | 1291 | 5869 | 174 | 9 | 72 | 255 | 19.7521 | 637 | 10.8536 |
| validation | human-v3 | gpt2_tuned_beam15 | 1291 | 5869 | 168 | 9 | 67 | 244 | 18.9001 | 641 | 10.9218 |
| test | human-v3 | gpt2_tuned_beam5 | 1381 | 6207 | 159 | 14 | 32 | 205 | 14.8443 | 543 | 8.7482 |
| test | human-v3 | greedy | 1381 | 6207 | 177 | 17 | 41 | 235 | 17.0167 | 587 | 9.4571 |
| test | human-v3 | beam5 | 1381 | 6207 | 175 | 14 | 34 | 223 | 16.1477 | 559 | 9.0060 |
| test | human-v3 | gpt2_original | 1381 | 6207 | 160 | 14 | 35 | 209 | 15.1340 | 549 | 8.8449 |
| test | human-v3 | gpt2_tuned_beam15 | 1381 | 6207 | 153 | 15 | 30 | 198 | 14.3374 | 535 | 8.6193 |

S/D/I use existing deterministic backtrace; independent jiwer counts and all per-sample edits are in metrics.json. Equivalent optimal jiwer alignments may have different S/D/I, but word/character errors and denominators are independently checked.
The sealed fixed-hypothesis artifact and every bound input hash were verified against reconstructed old-v2 manifests. Historical raw split hashes/references and frozen beam15 winner were checked. No scoring or reselection.

## Execution

```json
{
  "changed_sentences": 10,
  "corrected_spans": 11,
  "actual_reapply_changed": 0,
  "samples": 60,
  "mismatches": 59,
  "char_only": 0,
  "backup": "data/benchmark/long-audio/reference-backups/human-v3-1793761569290419152",
  "validation": {
    "samples": 60,
    "counts": {
      "capleaf/viVoice": {
        "validation": 0,
        "test": 0
      },
      "google/fleurs": {
        "validation": 30,
        "test": 30
      },
      "NhutP/VietSpeech": {
        "validation": 0,
        "test": 0
      }
    },
    "duration_min": 15.24,
    "duration_max": 29.94,
    "duration_total": 1236.24
  },
  "packages": [
    {
      "path": "/home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/review-packages/fleurs13-human-reviewed-v3",
      "zip": "/home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/review-packages/fleurs13-human-reviewed-v3.zip",
      "zip_bytes": 14059863,
      "samples": 13,
      "zip_entries": 109,
      "audio_hashes_verified": 13,
      "files_hashed": 108
    },
    {
      "path": "/home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/review-packages/fleurs60-mismatches-human-v3",
      "zip": "/home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/review-packages/fleurs60-mismatches-human-v3.zip",
      "zip_bytes": 62890920,
      "samples": 59,
      "zip_entries": 477,
      "audio_hashes_verified": 59,
      "files_hashed": 476
    }
  ],
  "seconds": 30.064342011999543
}
```

## Tests and reproduction evidence

- RED: each new isolated smoke file initially failed with the expected missing `human_reference_review` import (0.000s each).
- GREEN: `test_human_reference_review.py`: 3 passed, 0.016s; `test_review_package.py`: 2 passed, 0.022s. Covers literal supplied variants, missing/multiple spans, source guards, old-v2 preservation, idempotence, no candidate reselection, relative copied WAV/ZIP, HTML escaping, edit alignment and refusal to overwrite later human edits.
- Existing filename-limited tests: reference normalization 4 passed (0.002s), numeric metrics 8 (0.021s), optimization 6 passed / 1 opt-in historical audit skipped (0.019s), long audio 15 (0.066s), ranker 2 (0.000s), prefix 2 (0.000s), error audit 3 (0.001s), holdout 2 (0.017s). Total: 47 passed, 1 skipped. No full suite executed.
- Exact isolated runner: `.venv-benchmark/bin/python -m unittest discover -s tests -p FILENAME -v`. Standard unittest discovery's exact filename pattern limits execution; no custom suite-wide runner.
- Existing evaluator rerun: `.venv-benchmark/bin/python tools/reevaluate_numeric_metrics.py --artifact-only --output outputs/benchmarks/long/human-review-v3-cache-verification-20261104` completed in 13.027s before activating human-v3. Required because the legacy audit's code fingerprint predated the already-present official TSV parser correction. It repeated strict original-cache/frozen checks with current code; all 60 records' candidate IDs and raw texts were exactly equal to the sealed legacy fixed artifact. Old fingerprints were not rewritten. `--artifact-only` preserves the earlier project audit document.
- Actual application/execution: `.venv-benchmark/bin/python tools/human_reference_review.py`. Publication was resumed using `--backup data/benchmark/long-audio/reference-backups/human-v3-1793761569290419152` after stdlib `wave` rejected IEEE-float WAV format 3; the implementation now uses full ffmpeg decoding for those WAVs. No audio transcoding or source audio edits occurred.
- Successful repeat execution: same tool without arguments, 30.064s, zero reference changes on reapply. It rechecked all 60 audio hashes/decoded PCM/durations, split partitions, fixed identities, metrics and both packages. Subsequent package-only verification after tightening portable-link/ZIP-member checks: 11.640s, both passed, 100% file/audio hashes. Each ZIP's exact member set, CRC, manifest data and hash inventory match the package; all HTML media/link targets are local relative files.
- `git diff --check` passed. No inference, LM scoring, tuning, downloads, server launch, commits or pushes. PhoBERT and optional symmetric D were omitted; the frozen five methods and A/C/human-v3 were evaluated. Fresh30 was not mixed into these packages.

Changed implementation files: `tools/human_reference_review.py`, `tools/reevaluate_numeric_metrics.py` (artifact-only report switch), `tests/test_human_reference_review.py`, `tests/test_review_package.py`. Added v3 annotation and this report; active manifest and its two split files now retain `reference_raw`, `reference_v2` and current `reference`. Prior active bytes are preserved in the named backup. Other pre-existing dirty project files were left intact.
