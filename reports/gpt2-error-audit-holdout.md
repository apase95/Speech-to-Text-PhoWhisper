# Corrected source-v2 headline — all original30 holdout samples retained

**Completed source fix and verification:** production ingestion now uses one shared `parse_fleurs_tsv` with literal `csv.QUOTE_NONE`, exactly seven fields, numeric native IDs/sample counts, unique safe WAV filenames, nonempty transcription/character fields and valid gender. The archive reader additionally verifies decoded audio duration against source sample count. All named ingestion paths were inspected via an AST inventory saved in the sealed overlay: the only default-quote CSV reader left is explicitly labeled historical-bug reproduction in `physical_tsv_audit`, not ingestion. The new diagnostic `verify_tsv` delegates to the corrected shared parser.

**Exact test:836 diagnosis after correction:** retained46-word source reference, frozen GPT2 S/D/I **5/1/0** (six word errors); top1 and word oracle **4/0/0**. Frozen GPT2 corpus D deletions **163→16**: precisely147 artificial character-field tokens removed, not128. This is a source-parser correction, not a newly recognized-word improvement or a human listening approval.

**Verification this turn:** RED first (missing parser/correction tool), GREEN final **52 passed / one optional historical audit skipped**, all via exact filename-limited standard `unittest discover`. New `test_fleurs_source_correction.py`:5 passed (0.011s), including malformed quote retention, bad schema/path/duplicate filename rejection, native-ID/filename swaps, immutable reference-only bridge and tampering rejection, preserved old numeric annotations, and viewer-versus-physical ordinal mapping. Existing scoped files: long_audio_smoke15 (0.089s), holdout_deletion_audit5 (0.002s), numeric_metrics8 (0.026s), reference_normalization4 (0.002s), gpt2_optimization6+one skipped (0.014s), gpt2_prefix2 (0.000s), nbest_ranker2 (0.000s), fleurs_holdout2 (0.032s), gpt2_error_audit3 (0.000s). Full suites skipped; the sole brief requires these isolated files and actual audit, not full integration. Environment checker passed138 effective pins (0.424s). Tracked and new-file whitespace checks had no diagnostics.

**Actual commands:** `.venv-benchmark/bin/python tools/correct_fleurs_sources.py` initially stopped safely before publication because a historical viewer ordinal was not a physical TSV ordinal; native-ID/exact-sample mapping fixed that assumption. `.venv-benchmark/bin/python tools/correct_fleurs_sources.py --resume-metadata` completed the all90 audit/validation, source overlay and cached reevaluation in25.759s. `.venv-benchmark/bin/python tools/correct_fleurs_sources.py --verify` independently reconstructed the published source overlay, corrected metrics/IDs, strict old-cache joins and active-manifest binding in6.495s; evidence is `postpublication-verification.json`. Corrected evaluation independently audits8 corpus method/policy rows and240 sample/method rows with jiwer. No model imports/inference in this source-correction tool.

**Network disclosure:** the failed prepublication attempt fetched only the pinned public dev TSV:247,001 response-body bytes, one request; its wall time was not instrumented. The resumed correction reused that file and the previously hash-verified official test TSV (zero additional requests). Both attempts were under the configured4MiB/60s public-metadata budget; no credentials, archive/audio acquisition or model downloads this turn. Cumulative data transfer across all stages is534,593,433 bytes (<1GiB). The `network` field below describes the successful resumed process; the initial attempt is explicitly recorded in `postpublication-verification.json`.

**Files changed this turn:** `tools/long_audio_benchmark.py`, `tools/audit_holdout_deletions.py`, new `tools/correct_fleurs_sources.py`, `tests/test_long_audio_smoke.py`, `tests/test_holdout_deletion_audit.py`, new `tests/test_fleurs_source_correction.py`, and this report. Generated versioned source overlays, corrected results, metadata and backup are ignored. Existing unrelated changes were preserved. No commits, inference, LM scoring, tuning, training, dependency changes, sample exclusions or replacements.

This supersedes the contaminated-reference headline only. No audio/sample exclusions or replacements, model inference, LM rescoring, tuning, normalization changes or listening claims. Frozen greedy/top1/GPT2 texts and selected IDs are unchanged; the word oracle is reference-dependent and remains diagnostic.

| Policy | Method | Words | S | D | I | Errors | WER % | CER % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | greedy | 1302 | 146 | 9 | 54 | 209 | 16.0522 | 9.6588 |
| A | beam15_top1 | 1302 | 125 | 9 | 54 | 188 | 14.4393 | 8.8364 |
| A | gpt2_frozen | 1302 | 108 | 14 | 50 | 172 | 13.2104 | 8.7839 |
| A | oracle | 1302 | 97 | 9 | 46 | 152 | 11.6743 | 7.7340 |
| D | greedy | 1315 | 146 | 11 | 43 | 200 | 15.2091 | 8.7212 |
| D | beam15_top1 | 1315 | 125 | 11 | 43 | 179 | 13.6122 | 7.8906 |
| D | gpt2_frozen | 1315 | 107 | 16 | 39 | 162 | 12.3194 | 7.8387 |
| D | oracle | 1315 | 95 | 11 | 35 | 141 | 10.7224 | 6.7486 |

A uses corrected literal source transcription; D applies the unchanged conservative numeric grammar to refs and hyps symmetrically. Both remain separate policies; source-schema validation is not human audio validation.

## All current90 source audit
- current60: 60 rows audited; 0 exact raw-source text differences; 0 normalized-source differences. IDs with normalized differences: [].
- fresh30: 30 rows audited; 1 exact raw-source text differences; 1 normalized-source differences. IDs with normalized differences: ['test:836'].
All rows checked against pinned official dev/test literal seven-field TSV, native ID, filenames and source sample count. Archive-backed rows require their persisted archive-member filename and physical ordinal to match. Ten historical viewer rows have viewer ordinals that are not physical TSV ordinals: they each match uniquely by native split/native ID/exact sample count. The physical filename candidates are recorded, but their audio-filename identity is not newly independently verified in this metadata-only pass. No filename ambiguity remained in these ten mappings.
Current60 active annotations/reference_raw and partitions were left untouched; the full enriched source_reference_v2 overlay preserves every prior field. Any old-source difference is explicitly represented there without forging original cache hashes. The only activated references are in the corrected fresh30 manifest, with complete previous-row snapshots and evidence.
- Frozen GPT2 A, all30 before→after: errors 319→172, words 1449→1302, S/D/I 108/161/50→108/14/50, WER 22.0152%→13.2104%, CER 13.1802%→8.7839%.
- A comparisons versus beam15: `{"greedy": {"win": 1, "tie": 20, "loss": 9}, "gpt2_frozen": {"win": 17, "tie": 10, "loss": 3}}`.
- Frozen GPT2 D, all30 before→after: errors 309→162, words 1462→1315, S/D/I 107/163/39→107/16/39, WER 21.1354%→12.3194%, CER 12.2345%→7.8387%.
- D comparisons versus beam15: `{"greedy": {"win": 1, "tie": 20, "loss": 9}, "gpt2_frozen": {"win": 17, "tie": 10, "loss": 3}}`.

The exact diagnosis is a147-token character-spelled field appended to test:836 by CSV quote handling (193 stored words versus46 source transcription words), not128 words. Quotes remain literal source data; ordinary metric punctuation normalization removes them, not a per-sample manual override. The all30 corrected result is not the earlier29-sample sensitivity diagnostic. The old 148-deletion outlier is re-aligned against its real physical transcription here; per-sample counts for every method/policy are saved.

## Integrity, publication and artifacts
The original results.json/N-best/GPT2/frozen files are unchanged. The exact original manifest is backed up byte-for-byte and still passes strict o.checked joins against the old caches. New results bind both old and corrected manifest hashes, source-overlay hashes and original cache hashes. A verified reference-only identity bridge prohibits audio, IDs, split, source provenance or candidate changes. Every corrected sample/corpus word and character total was independently checked with jiwer. No protocol check was disabled, and no old hash was rewritten.
Versioned directory: `outputs/benchmarks/long/fleurs-holdout30-20261104/source-correction-v2/`; corrected-results.json, independent-audit.json, source-reference-v2-overlay.json, current60-source-reference-v2.json, corrected-holdout30-manifest.json, official-dev.tsv, official-test.tsv. Active manifest: `data/benchmark/fleurs-holdout30/manifest.json`; historical original: `data/benchmark/fleurs-holdout30/source-correction-backups/fleurs-source-tsv-v2/manifest.json`.
Wall time 25.759s; metadata-only network `{"bytes": 0, "requests": 0, "seconds": 24.186944348000907, "max_bytes": 4194304, "max_seconds": 60, "test_tsv": "reused independently fetched pinned/hash-verified file"}`. All90 full audio/hash/duration validations passed before publication. Human listening remains pending for numeric/dialect and lexical readings; no audio was certified.

---

# Historical execution and diagnostic record (superseded headline)

# GPT2 error audit and fresh holdout status

> **Deletion follow-up limitation:** the original fresh30 headline below uses a contaminated stored reference for `test:836`: CSV quote handling merged the official transcription and character-spelled fields. This single row contributes 148/163 frozen-GPT2 D deletions. Original artifacts/headlines are preserved for provenance, not endorsed as clean-reference performance. See the targeted deletion follow-up and explicitly labeled leave-one-out sensitivity below. No row was replaced, no reference corrected, no model rerun and no listening certification performed.

Targeted follow-up final checks: `.venv-benchmark/bin/python tools/long_audio_benchmark.py validate --manifest data/benchmark/fleurs-holdout30/manifest.json` passed all30 complete file/PCM/duration checks in **4.149s process wall**. `test_holdout_deletion_audit.py` finished **5 passed, 0.001s**, with missing-module/helper RED runs beforehand. `git diff --check` and no-index whitespace checks for the new tool, new test and report had no whitespace diagnostics. Follow-up files changed only: `tools/audit_holdout_deletions.py`, `tests/test_holdout_deletion_audit.py`, this report; ignored diagnostic artifacts were added without altering the original manifest/caches/results. Independent metadata/archive verification added132,279,648 data bytes, bringing cumulative transfer to534,346,432 bytes, below1GiB. Official builder-script retrieval was unavailable; physical TSV identity/field/sample-count checks and direct official archive byte hashes are the actual source evidence, not an imagined builder inspection.



## Executed cached audit

Wall time 2.577s. 13 modified references queued; zero human approvals. No inference/network/training/tuning.

A original standard remains reported historically; C annotated-v2 and D symmetric-conservative are provisional, not definitive main metrics.

| Split | Metric | S/D/I | WER % | Oracle WER % | Whole-candidate gap errors | Local opportunity | Missing | Ambiguous |

|---|---|---|---:|---:|---:|---:|---:|---:|

| validation | C | 171/9/67 | 19.1325 | 18.3579 | 10 | 14 | 60 | 97 |

| validation | D | 171/9/67 | 19.1325 | 18.3579 | 10 | 14 | 60 | 97 |

| test | C | 154/21/30 | 14.7801 | 14.0591 | 10 | 9 | 62 | 83 |

| test | D | 154/16/30 | 14.4718 | 13.8929 | 8 | 9 | 62 | 83 |



All candidate alignments use an optimal Levenshtein DAG. A location is ambiguous if ANY candidate has multiple optimal mappings there. The chosen deterministic backtrace matches the existing metric. Independent jiwer validates total word/character errors and denominators (its equally optimal S/D/I may differ).

Local opportunities are not additive attainable improvements: correct tokens can occur in mutually incompatible candidates. Missing means absent at that aligned location, NOT necessarily absent anywhere in the text. Neither category establishes acoustic versus LM causality. Whole-candidate word oracle uses the same convention and decoder-order tie-break; no deployed choice is changed.

Concrete saved example (C, validation:151): reference index15 `bơi` → selected `hơn`, context `các nội dung bơi một trăm mét`; index36 `khối` → `khói`, context `hội thể thao khối thịnh vượng chung`. Both are missing from the aligned candidate set, with zero correct candidate IDs. Selected ID: `49587e8d593ff0a8f1fa7e1df6afa9f2b0dd6ecd144e33f802bd30bfe7f4004c`. These are text diagnostics, not listening conclusions. All local opportunities and their matching candidate IDs, including ambiguity cases, are retained in `substitutions.csv`.

## Human action required

Open `outputs/benchmarks/long/error-audit-20261104/review.html` locally. All 13 modified references are prioritized, citation test:266 first. Listen to the complete audio; resolve actual citation omission/presence, year readings, nghìn/ngàn, linh/lẻ and abbreviations. Cached hypotheses are UNTRUSTED. Leave uncertainty pending. Record accepted/override, reviewer, notes, exact transcript and human_verified only after real listening.

Guarded application is deliberately not automatic: review-queue.json binds audio SHA and raw/v2 text. Before any later override application, require human_verified=true, named reviewer, nonempty notes and exact expected current text/audio hash; validate the entire proposed manifest and partitions and back up originals. Existing text-only annotation transformer rejects audio_validated=true; do not misuse it as a listening approval importer.

## Fresh30 status

Completed in the subsequent stage below. The cached60 results above are not fresh-holdout results.

## Artifacts

`outputs/benchmarks/long/error-audit-20261104`: sealed analysis.json with source hashes, substitutions.csv with candidate IDs/context, pending review-queue.json, escaped local review.html. No old artifacts or references changed.

## Fresh30 — actual frozen evaluation

30 additional native official FLEURS TEST clips, 15–30 seconds, no concatenation; 2 at least20s, 28 explicit15–20 fallback. First eligible archive order, seed42 recorded (no randomized-order claim). Strict disjoint ID/native ID/audio/PCM/raw normalized transcript against current60 and all three known replacement backups, and within fresh30. No speaker/training independence claim. No holdout tuning or manual reference edits. All listening pending.
| Convention | Method | S | D | I | WER % | CER % |
|---|---|---:|---:|---:|---:|---:|
| A | greedy | 146 | 156 | 54 | 24.5687 | 13.9457 |
| A | beam15_top1 | 125 | 156 | 54 | 23.1194 | 13.2302 |
| A | gpt2_frozen | 108 | 161 | 50 | 22.0152 | 13.1802 |
| A | oracle | 97 | 156 | 46 | 20.6349 | 12.1817 |

A per-sample comparisons versus beam15: `{"greedy": {"win": 1, "tie": 20, "loss": 9}, "gpt2_frozen": {"win": 17, "tie": 10, "loss": 3}}`.

| Convention | Method | S | D | I | WER % | CER % |
|---|---|---:|---:|---:|---:|---:|
| D | greedy | 146 | 158 | 43 | 23.7346 | 13.0084 |
| D | beam15_top1 | 125 | 158 | 43 | 22.2982 | 12.2839 |
| D | gpt2_frozen | 107 | 163 | 39 | 21.1354 | 12.2345 |
| D | oracle | 95 | 158 | 35 | 19.6990 | 11.1971 |

D per-sample comparisons versus beam15: `{"greedy": {"win": 1, "tie": 20, "loss": 9}, "gpt2_frozen": {"win": 17, "tie": 10, "loss": 3}}`.

Network: `{"bytes": 133631328, "requests": 2, "seconds": 23.015257889999702, "max_bytes": 805306368, "max_seconds": 240, "status": {"test_archive_members_scanned": 206}, "previous_attempt": {"bytes": 268435456, "requests": 2, "seconds": 46.26679617500122, "max_bytes": 268435456, "max_seconds": 240, "status": {"test_archive_members_scanned": 419, "blocker": "RuntimeError"}}, "total_bytes": 402066784, "expansion_reason": "256MiB prefix yielded only2 disjoint20+; resume with explicit15\u201320 fallback within1GiB total"}`. Stage seconds: `{"decode": 95.0943766599994, "score": 75.30549393699948}`. Total 217.282s.
Word-oracle diagnostic uses convention-specific minimum whole-candidate word errors, decoder-order ties. CER is that word-oracle candidate, not a character oracle. Candidate diversity per sample and chosen IDs in results.json. Independent jiwer word/character totals checked. No new model downloads, training, PhoBERT or configuration search.
Artifacts: `outputs/benchmarks/long/fleurs-holdout30-20261104`. Protocol sealed before acquisition/inference, manifest/cache hashes sealed after. Generic D verbalizer applies identically to raw refs and hypotheses. A and D remain separate provisional analyses; human transcript confirmation pending.

Fresh audio validation: 30 samples, 15.06–28.38s, 516.90s total. Complete old60+fresh30 file/PCM/duration/cross-split validation passed (90 samples, 1753.14s). Old60 has 58 unique normalized raw transcripts; the two existing within-split repetitions were preserved, not misreported as globally text-unique. Every fresh transcript/native ID is disjoint from both active and known superseded rows. Native TEST means official TEST archive, not speaker independence.

The preferred initial prefix hit 256MiB after 419 members, not the end of the archive. The fallback does not establish global exhaustion of 20–30s clips. The resumed 217.282s includes acquisition/validation/model loading/evaluation in that process but excludes the failed initial process (46.267s network stage). Total measured acquisition time: 69.282s; transfer: 402,066,784 response-body bytes (383.441MiB), four requests. Byte counts exclude headers/TLS. Both network attempts stayed below 240s. No credentials were accessed.

Frozen GPT2 improves fresh A over same-beam top1 by 16 errors / 1.1042 WER percentage points, and D by 17 errors / 1.1628 points. Whole-candidate oracle headroom is 20 errors (A) / 21 errors (D); these are diagnostics, not independently attainable token repairs. Fresh deletions are numerous and require human audio/reference review; their counts do not establish a cause. No significance or broad generalization claim.

## Reproduction and verification (this execution)

Standard `unittest discover -s tests -p FILE -v` is the filename-limited runner, not a custom suite wrapper. New files demonstrated RED (missing production module ImportError, 0.000s each), then GREEN: `test_gpt2_error_audit.py` 3 passed in 0.000s; `test_fleurs_holdout.py` 2 passed in 0.018s. Synthetic tests use production alignment/selection/normalization, no ignored-data dependency: repeated-token ambiguity, insertion/deletion edges, candidate swaps, missing aligned token, HTML escaping, each disjointness key, and frozen choice distinct from reference oracle.

Exact commands:

```sh
.venv-benchmark/bin/python tools/audit_gpt2_errors.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/run_fleurs_holdout.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/run_fleurs_holdout.py --resume-acquisition
.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v
.venv-benchmark/bin/python tools/check_benchmark_environment.py
git diff --check
```

First holdout command stopped before inference at the byte cap, preserved partial data and pre-inference seal; the explicit resume verified that seal. Completed output refuses a new decode/resume. Decode30 once (greedy plus beam15), score once. Actual evaluation verifies all source/protocol/candidate hashes and exact validation-frozen acoustic/LM protocol, including runtime, prefix/vocabulary and GPT2 revision. Alpha .3 / beta -.02 never tuned here. Existing `choose` determines selections without reading holdout reference costs. Independent jiwer checked corpus word/character errors and denominators for every reported method/convention.

Mandatory existing filename results: numeric_metrics 8 passed (0.020s), reference_normalization 4 (0.002s), gpt2_optimization 6 passed / 1 optional historical audit skipped (0.012s), long_audio_smoke 15 (0.053s), gpt2_prefix_smoke 2 (0.000s), nbest_ranker_smoke 2 (0.000s). Including new files: **42 passed, one skipped**. Environment checker: **138 effective pins passed**, 0.488s. `git diff --check` passed. Actual current/fresh/combined validation is in `validation.json`. Full suites skipped: not requested; actual acquisition/inference and these validations were expressly mandated by the sole brief. All invoked processes exited; no background workers launched.

Changed files for this task only: `tools/audit_gpt2_errors.py`, `tools/run_fleurs_holdout.py`, `tests/test_gpt2_error_audit.py`, `tests/test_fleurs_holdout.py`, this report. Generated audio/HTML/cache/results remain ignored. Existing unrelated working-tree changes were preserved. No commits/pushes, dependency changes, model downloads, tuning or training. Remaining blocker is real human confirmation, not compute execution; guarded override application is intentionally not automatic.

Final new-file whitespace inspection used `git diff --no-index --check /dev/null PATH` for each of these five initially untracked files, since ordinary `git diff --check` omits them. It found and removed one trailing space in the HTML string concatenation; no production semantics changed, and the new audit smoke was rerun successfully (3 tests, 0.000s). The sealed cached analysis source hash refers to the execution-time file before that whitespace-only cleanup; artifact results were not rewritten. All five new-file whitespace checks were then clean (no-index can return1 solely because files differ from /dev/null).

## Targeted deletion follow-up — cached only, original headline unchanged



Fresh frozen GPT2 D has 163 deletions across 8/30 samples. Top1 contributes 148 (90.7975%); top3 contribute 157 (96.3190%). Old test D has 16 deletions across 12/30, top1 2 / top3 6.



| Sample | Duration s | Ref words D | GPT2 S/D/I | Word-oracle D | Selected / oracle tokens | Candidate token range |

|---|---:|---:|---|---:|---|---|

| test:836 | 17.40 | 193 | 5/148/0 | 147 | 79 / 80 | 78–82 |

| test:822 | 15.48 | 53 | 8/5/0 | 5 | 98 / 98 | 96–99 |

| test:32 | 15.42 | 37 | 2/4/0 | 0 | 71 / 81 | 71–81 |



### Confirmed reference-ingestion defect, not a listening conclusion

The top outlier test:836 (native ID1898) has a sentence followed by a character-spelled duplicate. This is not merely an unusually long transcript: the official physical TSV line has seven tab-separated fields, while the existing `csv.reader(..., delimiter="\t")` quote handling returns six, merging transcription field3 with adjacent character field4 because field3 has an unmatched opening quote. The local manifest exactly reproduces this legacy parser output. The diagnostic first reproduced that mapping, then independently inspected literal tab boundaries; reproducing the legacy parser alone was insufficient validation.

Across all30, 1 normalized stored references differ from physical transcription column3. No manifest, reference, selected candidate, score or original result has been edited. No cleaned-reference score is published here.

- test:836: native ID 1898, filename `14117020288073941718.wav`, source samples 278400; physical TSV fields 7 versus CSV fields 6; stored ref 193 words versus physical transcription 46 words, adjacent character field 147 normalized tokens.

The 148-deletion result is therefore dominated by contaminated reference text, not evidence of 148 missing spoken words. The exact true spoken transcript, punctuation/quote readings and remaining lexical mismatches still require human listening. Fixing the shared acquisition parser, auditing its historical output and publishing a separately versioned reference correction would require a new explicit change; this diagnostic does not silently perform that change or replace any holdout row.



### Top3 saved text (D; no listening performed)

- **test:836 reference:** `anh ta wales từ đầu đã nói dối chúng tôi đầu tiên bằng việc làm như thể vì những lý do hợp pháp thứ nữa bằng việc giả bộ như đang lắng nghe chúng tôi anh ta đã có màn xóa dấu vết đầy ngoạn mục a n h t a w a l e s t ừ đ ầ u đ ã n ó i d ố i c h ú n g t ô i đ ầ u t i ê n b ằ n g v i ệ c l à m n h ư t h ể v ì n h ữ n g l ý d o h ợ p p h á p t h ứ n ữ a b ằ n g v i ệ c g i ả b ộ n h ư đ a n g l ắ n g n g h e c h ú n g t ô i a n h t a đ ã c ó m à n x ó a d ấ u v ế t đ ầ y n g o ạ n m ụ c`

  **Frozen GPT2:** `anh ta wales từ đầu đã nói dối chúng tôi đầu tiên bằng việc làm như thể vì những lý do hợp pháp thứ nữa bằng việc dã bộ như đang lắng nghe chúng tôi anh ta đã có màn xó gió vết đàn lợn`

  **Word oracle:** `anh ta wales từ đầu đã nói dối chúng tôi đầu tiên bằng việc làm như thể vì những lý do hợp pháp thứ nữa bằng việc dã bộ như đang lắng nghe chúng tôi anh ta đã có màn xó gió vết đằng ngoạn mục`

- **test:822 reference:** `loài người đã biết đến các nguyên tố hoá học cơ bản như vàng bạc và đồng từ thời cổ đại bởi tất cả các nguyên tố đó đều có thể được phát hiện trong tự nhiên ở dạng nguyên thể và tương đối dễ khai thác bằng các công cụ nguyên thuỷ`

  **Frozen GPT2:** `lời người đã biết đến các quân tố học cơ bản như dàn bạc đồng từ thời có đại bởi tất cả quân tố đó đều có thể phát hiện trong tự nhiên ở dạng quân thể tương đối và khai thác bằng các công cụ nguyên thủy`

  **Word oracle:** `lời người đã biết đến các quân tố học cơ bản như dàn bạc đồng từ thời có đại bởi tất cả quân tố đó đều có thể phát hiện trong tự nhiên ở dạng quân thể tương đối và khai thác bằng các công cụ nguyên thủy`

- **test:32 reference:** `các đề cử khác bao gồm phim đạo diễn quay phim thiết kế trang phục biên tập phim nhạc phim gốc thiết kế sản xuất biên tập âm thanh hòa âm và kịch bản gốc hay nhất`

  **Frozen GPT2:** `các đề cử khác bao gồm phim đạo diễn với phim thiết kế trang phục biên tập thiết kế sản xuất biên tập âm thanh hoà âm và kịch bản gốc hay nhất`

  **Word oracle:** `các đề cử khác bao gồm phim đạo diễn quay phim thiết kế trang phục biên tập phim nhạc phim gốc thiết kế sản xuất biên tập âm thanh hoà âm và kịch bản gốc hay nhất`



### Source and cache identity

All30 sample/native IDs and archive filenames match the independently fetched pinned official TEST TSV. All30 physical rows have seven fields, and their official sample-count column5 equals the decoded duration ×16000. Column2 is raw transcription, column3 transcription, column4 the character-spelled field; the preserved TSV makes this distinction directly inspectable. The attempted pinned `fleurs.py` builder fetch returned HTTPError and is not claimed as schema evidence. No missing builder content was inferred.

All three top-deletion local WAV files independently match their named members in the pinned official TEST archive byte-for-byte (SHA256), not just filename. This rules out an observed top3 file swap; it does not certify the speech/transcript match by listening.

Strict integrity passed: original results/frozen seals, result→manifest/N-best/GPT2/protocol SHA bindings; full strict N-best/scored schemas; sample/source/split joins; exact frozen validation protocol/revision; unchanged old/backups and normalizer hashes; candidate IDs recomputed from source+sample+index+token arrays; candidate text independently decoded from stored tokens; greedy/candidate source equality between N-best and scored; and exact cached reconstruction of every saved metric/result row. Fresh-vs-old/superseded disjointness was rechecked. Fresh30 full FFmpeg/file/PCM/duration validation passed again.



### Preprocessing and stopping evidence

All30 waveforms were decoded and actual production mel preprocessing was recomputed without a model forward. Content frames 1506–2838 (10ms/frame), retained content frames exactly equal the corresponding full spectrogram prefix; zero content frames trimmed. All output mels are [(80, 3000)], right-padded to3000. Thus the 15.06–28.38s source audio fits the30s window; there is no observed audio-content truncation in this preprocessing path.

Metadata-only checkpoint read used `torch.load(map_location='meta', weights_only=True)` (no model instantiation, materialized weights, or encoder/decoder inference), after matching its frozen SHA. n_text_ctx=448, default sample_len=224 and4 initial tokens. Installed Whisper DecodingTask uses sample_len or n_text_ctx//2; main loop stops on completion/context or sample iteration cap. Across450 cached candidates, generated lengths are 59–141; none reaches224. Even max141+4 is below448. Top outlier has78–82 generated tokens, selected79, oracle80, far below either hard length cap.

The cache strips EOT and does not store generation-loop stop reasons. Beam finalize can append EOT to unfinished sequences, so actual per-candidate natural-EOT events cannot be independently reconstructed from cache alone. No candidate shows a token-length-cap signature, and the dominant deletion anomaly has direct reference-parser evidence; normal/early EOT for remaining errors is unconfirmed. Greedy lengths are only text re-encodings, not retained greedy token sequences. No silent LM truncation is used: the scorer checks context and fails on overflow.



### Leave-one-out sensitivity ONLY — not a revised headline or exclusion rule

Omitting test:836 solely to quantify concentration gives the following fixed-GPT2 diagnostic. The original30-sample A/D headline results remain unchanged; this is not tuning, a replacement holdout, reference cleaning or an adopted exclusion.

- A, remaining29: S/D/I=103/13/50, 166/1256 word errors, WER 13.2166%, CER 8.7815%.

- D, remaining29: S/D/I=102/15/39, 156/1269 word errors, WER 12.2931%, CER 7.8038%.



### Commands, tests and artifacts

```sh

.venv-benchmark/bin/python -m unittest discover -s tests -p test_holdout_deletion_audit.py -v

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/audit_holdout_deletions.py

.venv-benchmark/bin/python tools/audit_holdout_deletions.py --schema-detail

```

Actual first diagnostic wall time 49.667s including metadata-only checkpoint read, pure audio preprocessing, validation and bounded network. Independent source transfer `{"bytes": 132279648, "requests": 3, "seconds": 32.84077421800066, "max_bytes": 268435456, "max_seconds": 240}`; total data bytes including original acquisition 534,346,432 (<1GiB). No credentials/signed URLs accessed/output. Existing source artifacts were checked unchanged at end.

New isolated tests demonstrated RED (missing module/helper), then GREEN: five portable synthetic tests cover concentration/zero denominator, swapped official filename/transcription, candidate-token tampering, sealed manifest binding/tampering, and the actual malformed-quote contamination pattern. No model reruns, decoding, rescoring, tuning, downloads of weights, reference changes or holdout replacements. No full-suite rerun; unrelated passing checks are not repeated.

Artifact directory: `outputs/benchmarks/long/fleurs-holdout30-20261104/deletion-diagnostic/`: sealed `diagnostic.json`, `schema-evidence.json`, `per-sample.csv`, independently fetched `official-test.tsv`, and `report.md`. Original frozen evaluation remains at `outputs/benchmarks/long/fleurs-holdout30-20261104/results.json`; manifest `data/benchmark/fleurs-holdout30/manifest.json`.
