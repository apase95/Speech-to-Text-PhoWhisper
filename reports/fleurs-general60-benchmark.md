# FLEURS general-content — feasible23 frozen benchmark; requested60 INCOMPLETE

**Not a completed native30/30 general60.** Sealed policy yields12 validation/11 test,23 distinct texts at15–30s after excluding fresh30. No silent relaxation, duplicated texts, split reassignment or post-inference replacement. Original datasets remain active and byte-identical.

## Reference-only policy and screening

Policy persisted before screening; selection and source eligibility persisted before acquisition and decoding. Full pinned TSV:1218 recordings;278 at15–30s;175 automatic-clear records manually read. Exclude explicit digits/spoken numeric constructs and foreign proper entities/acronyms/transliterated terms; indefinite một and generic loanwords such as protein/blog allowed. No capitalization-only NER. Exact matched spans/reasons for every pool/current60 row are in screening.json. No hypotheses were inputs to screening. Accent unknown; no accent guesses or restrictions.
Requested30/30 is blocked under these conservative constraints. Broader10–15s sampling was not silently activated. These feasible counts are policy-specific, not an assertion about every imaginable eligibility policy or global maximum. A future wider-duration dataset requires its own pre-inference selection.

Retained11/23 (47.83%) from current60;12 replacement recordings. Not fresh/independent. Superseded-pool usage of replacements was not exhaustively audited and is explicitly unknown. All fresh30 native/text/audio/PCM excluded. No speaker/training independence claim.

| Original ID | Decision | Reference-only reason/spans |
|---|---|---|
| validation:151 | excluded from supplement | digits: 100; digits: 200; digits: 100; digits: 200; foreign_entity_or_term: olympic; foreign_entity_or_term: của anh |
| validation:240 | excluded from supplement | foreign_entity_or_term: kruger |
| validation:99 | excluded from supplement | spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười |
| validation:37 | retained | ordinary content |
| validation:178 | excluded from supplement | spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười |
| validation:280 | excluded from supplement | foreign_entity_or_term: kruger |
| validation:60 | excluded from supplement | spoken_numeric: mười; spoken_numeric: mười |
| validation:231 | excluded from supplement | digits: 4892; digits: 4892 |
| validation:254 | excluded from supplement | foreign_entity_or_term: java |
| validation:126 | excluded from supplement | digits: 400; digits: 000; digits: 400; digits: 000 |
| validation:237 | excluded from supplement | digits: 6; digits: 10; digits: 1789; digits: 11; digits: 4; digits: 6; digits: 10; digits: 1789; digits: 11; digits: 4; foreign_entity_or_term: paris |
| validation:121 | excluded from supplement | foreign_entity_or_term: npws |
| validation:256 | excluded from supplement | spoken_numeric: mười; spoken_numeric: mười |
| validation:339 | excluded from supplement | foreign_entity_or_term: iraq |
| validation:73 | excluded from supplement | digits: 108; digits: 56; digits: 108; digits: 56 |
| validation:257 | excluded from supplement | foreign_entity_or_term: bush |
| validation:260 | excluded from supplement | digits: 1; digits: 000; digits: 2000; digits: 1; digits: 000; digits: 2000; foreign_entity_or_term: của anh |
| validation:79 | retained | ordinary content |
| validation:83 | excluded from supplement | foreign_entity_or_term: của anh |
| validation:67 | retained | ordinary content |
| validation:303 | excluded from supplement | foreign_entity_or_term: romania |
| validation:169 | retained | ordinary content |
| validation:337 | excluded from supplement | foreign_entity_or_term: newt gingrich |
| validation:58 | retained | ordinary content |
| validation:191 | excluded from supplement | foreign_entity_or_term: bishkek |
| validation:136 | excluded from supplement | foreign_entity_or_term: armand versace |
| validation:350 | excluded from supplement | foreign_entity_or_term: ski bums |
| validation:298 | excluded from supplement | explicit_numeric_construct: một năm |
| validation:64 | excluded from supplement | foreign_entity_or_term: point marion |
| validation:176 | retained | ordinary content |
| test:524 | retained | ordinary content |
| test:153 | excluded from supplement | digits: 45; digits: 117; digits: 45; digits: 117 |
| test:767 | excluded from supplement | foreign_entity_or_term: huhne |
| test:574 | retained | ordinary content |
| test:375 | excluded from supplement | foreign_entity_or_term: tupua |
| test:39 | excluded from supplement | foreign_entity_or_term: catalan; foreign_entity_or_term: barcelona; explicit_numeric_construct: một nửa |
| test:49 | excluded from supplement | foreign_entity_or_term: catalunya |
| test:266 | excluded from supplement | digits: 2002; digits: 2002 |
| test:161 | excluded from supplement | explicit_numeric_construct: thứ bảy |
| test:124 | excluded from supplement | foreign_entity_or_term: trung hoa |
| test:670 | excluded from supplement | foreign_entity_or_term: usoc |
| test:435 | excluded from supplement | explicit_numeric_construct: hai thế kỷ |
| test:760 | excluded from supplement | foreign_entity_or_term: martelly |
| test:322 | excluded from supplement | spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười; spoken_numeric: mười |
| test:621 | excluded from supplement | explicit_numeric_construct: hai ý tưởng |
| test:483 | excluded from supplement | foreign_entity_or_term: usoc |
| test:115 | retained | ordinary content |
| test:121 | excluded from supplement | digits: 50; digits: 31; digits: 50; digits: 31 |
| test:557 | excluded from supplement | foreign_entity_or_term: pamela |
| test:242 | excluded from supplement | spoken_numeric: triệu; spoken_numeric: triệu; spoken_numeric: triệu; spoken_numeric: triệu |
| test:145 | retained | ordinary content |
| test:653 | excluded from supplement | digits: 70; digits: 70 |
| test:6 | excluded from supplement | foreign_entity_or_term: plitvice |
| test:813 | excluded from supplement | digits: 1988; digits: 1988 |
| test:472 | excluded from supplement | digits: 2002; digits: 2002; foreign_entity_or_term: nyiragongo; foreign_entity_or_term: goma |
| test:668 | excluded from supplement | foreign_entity_or_term: falkland |
| test:177 | excluded from supplement | foreign_entity_or_term: toginet |
| test:73 | retained | ordinary content |
| test:661 | excluded from supplement | digits: 29; digits: 15; digits: 29; digits: 15 |
| test:289 | excluded from supplement | foreign_entity_or_term: saint petersburg |

## Frozen results

Beam15, alpha0.3/beta−0.02, GPT2 revision `65818d14816b42be09e2201933bf07106d9a2647`, invariant v3 protocol exactly matched to the existing winner. No tuning, downloads of models, training or new dependencies. All selected23 decoded once and scored once. All errors remain in the dataset. A=source transcription; D=unchanged symmetric conservative numeric rules on refs/hyps; current=preserved active labels for retained rows and official source for new rows. No new label rewrites. CER includes normalized spaces.

| Split | Policy | Method | Words | Chars | S | D | I | Errors | WER % | CER % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | A | greedy | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | A | beam15_top1 | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | A | gpt2_frozen | 531 | 2299 | 22 | 2 | 1 | 25 | 4.7081 | 2.1314 |
| validation | A | oracle | 531 | 2299 | 19 | 2 | 1 | 22 | 4.1431 | 2.0879 |

validation/A win/tie/loss versus beam15: `{"greedy": {"win": 0, "tie": 12, "loss": 0}, "gpt2_frozen": {"win": 8, "tie": 4, "loss": 0}}`.
| validation | D | greedy | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | D | beam15_top1 | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | D | gpt2_frozen | 531 | 2299 | 22 | 2 | 1 | 25 | 4.7081 | 2.1314 |
| validation | D | oracle | 531 | 2299 | 19 | 2 | 1 | 22 | 4.1431 | 2.0879 |

validation/D win/tie/loss versus beam15: `{"greedy": {"win": 0, "tie": 12, "loss": 0}, "gpt2_frozen": {"win": 8, "tie": 4, "loss": 0}}`.
| validation | current | greedy | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | current | beam15_top1 | 531 | 2299 | 35 | 2 | 1 | 38 | 7.1563 | 3.1318 |
| validation | current | gpt2_frozen | 531 | 2299 | 22 | 2 | 1 | 25 | 4.7081 | 2.1314 |
| validation | current | oracle | 531 | 2299 | 19 | 2 | 1 | 22 | 4.1431 | 2.0879 |

validation/current win/tie/loss versus beam15: `{"greedy": {"win": 0, "tie": 12, "loss": 0}, "gpt2_frozen": {"win": 8, "tie": 4, "loss": 0}}`.
| test | A | greedy | 484 | 2116 | 45 | 4 | 5 | 54 | 11.1570 | 5.9074 |
| test | A | beam15_top1 | 484 | 2116 | 46 | 15 | 2 | 63 | 13.0165 | 8.0340 |
| test | A | gpt2_frozen | 484 | 2116 | 39 | 15 | 2 | 56 | 11.5702 | 7.6560 |
| test | A | oracle | 484 | 2116 | 39 | 3 | 2 | 44 | 9.0909 | 5.1040 |

test/A win/tie/loss versus beam15: `{"greedy": {"win": 3, "tie": 7, "loss": 1}, "gpt2_frozen": {"win": 6, "tie": 5, "loss": 0}}`.
| test | D | greedy | 484 | 2116 | 45 | 4 | 5 | 54 | 11.1570 | 5.9074 |
| test | D | beam15_top1 | 484 | 2116 | 46 | 15 | 2 | 63 | 13.0165 | 8.0340 |
| test | D | gpt2_frozen | 484 | 2116 | 39 | 15 | 2 | 56 | 11.5702 | 7.6560 |
| test | D | oracle | 484 | 2116 | 39 | 3 | 2 | 44 | 9.0909 | 5.1040 |

test/D win/tie/loss versus beam15: `{"greedy": {"win": 3, "tie": 7, "loss": 1}, "gpt2_frozen": {"win": 6, "tie": 5, "loss": 0}}`.
| test | current | greedy | 484 | 2116 | 45 | 4 | 5 | 54 | 11.1570 | 5.9074 |
| test | current | beam15_top1 | 484 | 2116 | 46 | 15 | 2 | 63 | 13.0165 | 8.0340 |
| test | current | gpt2_frozen | 484 | 2116 | 39 | 15 | 2 | 56 | 11.5702 | 7.6560 |
| test | current | oracle | 484 | 2116 | 39 | 3 | 2 | 44 | 9.0909 | 5.1040 |

test/current win/tie/loss versus beam15: `{"greedy": {"win": 3, "tie": 7, "loss": 1}, "gpt2_frozen": {"win": 6, "tie": 5, "loss": 0}}`.

## Original60 comparison — SAME raw/symmetric policy

| Split | Policy | Original GPT2 WER % | Curated GPT2 WER % |
|---|---|---:|---:|
| validation | A | 22.1784 | 4.7081 |
| validation | D | 19.1325 | 4.7081 |
| test | A | 16.8514 | 11.5702 |
| test | D | 14.4718 | 11.5702 |

This is different content/duration/denominator sampling, NOT model improvement or a generalization estimate. Original raw references retain historical label differences; current human-v3 is separate, not compared to raw as if identical. Human corrections are span-limited; whole-sentence listening remains pending. Word oracle is diagnostic whole-candidate minimum word errors, decoder-order ties; its CER is not a character oracle.

## Audits, timings and artifacts

Independent jiwer checks word/character totals and denominators for every corpus method/policy, original and curated. Production deterministic S/D/I and independent alternative backtraces are both retained. Full real WAV byte/decoded-PCM/duration validation passed for original60 and curated23. Strict candidate IDs/source caches/protocols/fingerprints checked. Copied WAV hashes, portable HTML links, ZIP exact member set, CRC and SHA inventory verified.

```json
{
  "durations": {
    "curated": {
      "validation": {
        "samples": 12,
        "under15": 0,
        "from15to20": 11,
        "at_least20": 1,
        "min": 15.0,
        "max": 29.94,
        "mean": 17.625,
        "total": 211.5
      },
      "test": {
        "samples": 11,
        "under15": 0,
        "from15to20": 6,
        "at_least20": 5,
        "min": 15.06,
        "max": 22.92,
        "mean": 19.047272727272727,
        "total": 209.52
      }
    },
    "original": {
      "validation": {
        "samples": 30,
        "under15": 0,
        "from15to20": 16,
        "at_least20": 14,
        "min": 15.24,
        "max": 29.94,
        "mean": 20.378,
        "total": 611.34
      },
      "test": {
        "samples": 30,
        "under15": 0,
        "from15to20": 7,
        "at_least20": 23,
        "min": 15.3,
        "max": 25.32,
        "mean": 20.83,
        "total": 624.9
      }
    }
  },
  "timings": {
    "validation_decode": 36.78866636699968,
    "validation_score": 24.213248857999133,
    "test_decode": 29.582750183002645,
    "test_score": 21.100797426999634
  },
  "network": [
    {
      "split": "validation",
      "bytes": 214500592,
      "seconds": 25.05274904399994,
      "max_seconds": 360
    },
    {
      "split": "test",
      "bytes": 429445120,
      "seconds": 59.93365523100147,
      "max_seconds": 360
    }
  ],
  "candidate_diversity": {
    "validation": {
      "candidates": 180,
      "unique": 180,
      "mean_pairwise": 0.05947604998541334
    },
    "test": {
      "candidates": 165,
      "unique": 165,
      "mean_pairwise": 0.07848834015486121
    }
  },
  "package": {
    "path": "outputs/review-packages/fleurs-general60-feasible23",
    "zip": "outputs/review-packages/fleurs-general60-feasible23.zip",
    "bytes": 21763175,
    "samples": 23,
    "entries": 188,
    "verified": true
  },
  "seconds": 129.94413982900005
}
```

Artifacts: `data/benchmark/fleurs-general60/manifest.json` and native split files (directory retains requested target name, but manifest has23). `outputs/benchmarks/long/fleurs-general60-20261104/`: policy.json, full-pool.json, screening.json, selection.json, acquired-seal.json, inference-freeze.json, split N-best/GPT2, results.json, timings.json. Review package includes every23 copied WAV plus raw/current/GPT2 text, highlights and pending decisions. Open review.html from the ZIP; no file:// or remote dependencies.

Reproduce stages separately with `.venv-benchmark/bin/python tools/curate_fleurs_general.py inventory|screen|acquire|benchmark` (one stage argument, existing sealed outputs refused). Offline benchmark invocation: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/curate_fleurs_general.py benchmark`.

Scope caveats: native30/30 target NOT met; conservative lexical screening is not exhaustive NER or human audio certification; very small selected read-speech subset; overlap and difficulty bias; unknown speaker/training overlap; no accent verification. No commitments/pushes. Full suites skipped; exact isolated smoke and required real artifact checks only.

## Final execution and verification evidence

- New isolated smoke RED: `.venv-benchmark/bin/python -m unittest discover -s tests -p test_fleurs_general.py -v` failed with expected missing production module (0.000s). Initial GREEN2 tests0.024s; final GREEN3 tests0.029s, including ordinary-content false-positive guards, explicit numeric/foreign spans, refusal of hypothesis inputs, native split/duplicate identity rejection, literal QUOTE_NONE, fingerprint tampering, independent metric empty edge, copied-WAV/escaped-relative-HTML/ZIP and existing-package refusal. Filesystem constants alone were redirected to a temporary directory; production screening/metrics/package behavior was not mocked.
- Runner inspected: standard-library `unittest discover -s tests -p EXACT_FILENAME -v`, which limits discovery to that filename. Required existing isolated files: numeric_metrics8 (0.021s), reference_normalization4 (0.004s), gpt2_optimization6 passed/1 optional historical audit skipped (0.011s), long_audio_smoke15 (0.045s), gpt2_prefix_smoke2 (0.000s), nbest_ranker_smoke2 (0.000s), human_reference_review3 (0.013s), review_package2 (0.018s). Including the new file: **45 passed / one skipped**. No suite-wide command.
- `.venv-benchmark/bin/python tools/check_benchmark_environment.py`: PASS138 effective pins,0.456s; no dependency/model downloads.
- Actual staged commands: `tools/curate_fleurs_general.py inventory`, `screen`, `acquire`, and the offline/thread-limited `benchmark` command above, all with `.venv-benchmark/bin/python`. The first screen attempt safely stopped before selection publication because the cached source overlay predates human-v3. It now verifies that overlay against the guarded reconstructed v2 rows while preserving the current v3 rows; no originals or historic hashes were rewritten. The successful acquisition transferred **643,945,712 response-body bytes** in **84.986s** across two archive stages, each under360s and combined below1GiB. Metadata reused pinned/hash-verified TSVs; no credentials or signed URLs logged.
- Actual benchmark wall129.944s, including validation/model loading/audit/package, excluding acquisition. Decode66.371s and GPT2 scoring45.314s in total. No inference was rerun after this successful execution.
- `.venv-benchmark/bin/python tools/curate_fleurs_general.py verify`: PASS3.591s. Reconstructed all1218 rows from literal pinned TSVs; checked complete screening reconstruction, policy→screen→selection→acquisition→inference seal ordering and bindings, both split caches/frozen protocols, all saved corpus metrics, originals unchanged and published package ZIP hashes. Evidence: `postpublication-verification.json`. Verification-only code was added after inference; execution-time and final source hashes are both preserved, not rewritten.
- Known superseded usage was subsequently audited against all three exact backup manifests named by the earlier holdout runner. Per-sample native/audio/PCM matches are recorded in `postpublication-verification.json`; the pre-inference manifest's conservative prior-usage-unknown statement remains historical, not overwritten. This still does not establish freshness against every possible earlier use.
- Historical viewer-backed retained audio uses the existing native-ID/exact-sample-count physical-TSV mapping; its archive filename was not newly independently byte-matched in this run. All retained audio byte/PCM hashes and source sample counts were validated. Independent listening and accent verification remain pending.
- Screening is intentionally conservative and lexical: e.g. the literal `của anh` span was used for the source's English saddle provenance, so the rule would also overexclude that pronoun phrase in a different text. No exhaustive/context-perfect NER claim is made, and no rules or selected rows were changed after results. The shortage is therefore policy-specific, not proof that every reasonable broader content policy has only23 feasible clips.
- `git diff --check` passed. Full integration/e2e suites and optional historical test audit were skipped; the mandatory real acquisition/benchmark/audio/metric/package audits were executed. No commits, pushes, training, model downloads or unrelated edits.

Changed implementation files for this task: `tools/curate_fleurs_general.py`, `tests/test_fleurs_general.py`, and this report. New supplementary audio/manifests and seals/results are generated ignored artifacts; the copied-WAV review package is local and untracked. Prior packages and active original datasets remain unchanged.

## Targeted cached-only deletion and shortage diagnosis

**One sample explains the positive deletion excess:** `test:10188` (physical TSV TEST index188;22.92s). It contributes GPT2/beam15-top1 **12 deletions**, versus greedy **0**. The remaining10 clips contribute3 GPT2 deletions versus4 greedy deletions: `test:115` offsets the excess by one. Thus corpus excess is `12 - 1 = 11` (15 versus4), not eleven separate failed clips. This is a cached text-alignment finding, not an acoustic explanation.

| Test sample | Duration s | GPT2 D | Greedy D | Beam top1 D | GPT2 / greedy word errors |
|---|---:|---:|---:|---:|---:|
| test:524 | 22.20 | 0 | 0 | 0 | 10 / 10 |
| test:574 | 20.76 | 1 | 1 | 1 | 8 / 10 |
| test:115 | 20.88 | 0 | 1 | 0 | 6 / 6 |
| test:145 | 19.50 | 2 | 2 | 2 | 7 / 8 |
| test:73 | 20.16 | 0 | 0 | 0 | 2 / 2 |
| test:10188 | 22.92 | 12 | 0 | 12 | 16 / 9 |
| test:10097 | 18.60 | 0 | 0 | 0 | 3 / 3 |
| test:10215 | 15.06 | 0 | 0 | 0 | 0 / 1 |
| test:10330 | 17.04 | 0 | 0 | 0 | 1 / 1 |
| test:10429 | 16.56 | 0 | 0 | 0 | 2 / 2 |
| test:10764 | 15.84 | 0 | 0 | 0 | 1 / 2 |
| **Total** | | **15** | **4** | **15** | **56 / 54** |

Normalized source snippets for `test:10188`:

- Reference: `... civilis trong tiếng latinh nghĩa là thuộc nhân dân liên quan đến civis trong tiếng latinh nghĩa là công dân và civitas nghĩa là thành phố hay thành bang ...`
- Frozen GPT2: `... civic trong tiếng latin nghĩa là công dân và civic tác nghĩa là thành phố hay thành bang ...`
- Greedy: `... siri phelead trong tiếng latin nghĩa là thuộc nhân dân liên quan đến siri pitt trong tiếng latin nghĩa là công dân và siri pitt nghĩa là thành phố hay thành mang ...`

**The Vietnamese clause is NOT absent from all beam15 candidates.** Zero-based candidate indices **1,2,3,4,8,9,10** all contain `thuộc nhân dân liên quan đến`; they have48–50 normalized words and **D=0**. GPT2 selected index14 (36 words, S/D/I3/12/1). Beam top1 index0 is already short (36 words,4/12/1); rescoring did not originate this omission, but failed to select the available longer alternatives. Index8 has48 words and4/0/1 (five errors), versus selected16 errors. Greedy's exact whole string is absent from all15 candidates, but its correctly preserved Vietnamese clause is available in seven. The exact full12-token reference deletion span is absent because the alternatives also substitute `civis` and/or `latinh`; exact-span absence must not be misreported as absence of the correct Vietnamese clause.

Frozen score decomposition (`avg_logprob + .3 * LM - .02 * words`, higher wins):

| Candidate | Words | avg_logprob | LM | Length term | Total |
|---|---:|---:|---:|---:|---:|
| Selected14 | 36 | −0.093874262 | −3.907516956 | −0.72 | **−1.986129349** |
| Longer8 | 48 | −0.081708512 | −4.060126305 | −0.96 | −2.259746404 |

Against longer8, selected14 has a worse acoustic score, a better mean-LM contribution, and a0.24 smaller length penalty; its combined score wins by0.273617055. This is evidence about the frozen scoring components, not a claim that LM alone caused the omission, or that a different configuration would universally help. No score/configuration was changed or retuned.

**Per-clip word-error W/T/L on11 native TEST clips:** GPT2 versus beam15-top1 **6/5/0**; GPT2 versus greedy **4/6/1**. Four small wins over greedy are outweighed by the seven-error loss on `test:10188`: total56 versus54 errors. A/D/current policies coincide on these selected clips.

| Split | Clips | Duration min s | Max s | Mean s | Raw-reference words |
|---|---:|---:|---:|---:|---:|
| Validation | 12 | 15.00 | 29.94 | 17.625 | 531 |
| Test | 11 | 15.06 | 22.92 | 209.52 / 11 = 19.0472727272727… | 484 |

### Why30/30 was not reached — saved screening counts

Counts below are recordings, not necessarily distinct sentences. The exclusive waterfall assigns the first applicable reason in order digits→spoken-number scale→foreign term→explicit numeric construct. The separate overlapping counts must not be added.

| Stage / exclusive reason | Validation | Test |
|---|---:|---:|
| Full pinned TSV pool | 361 | 857 |
| Under15s | 291 | 640 |
| At least30s | 2 | 7 |
| Duration-eligible15–30s | 68 | 210 |
| Excluded: digits | 16 | 63 |
| Excluded next: spoken numeric scale | 6 | 18 |
| Excluded next: foreign entity/term | 25 | 83 |
| Excluded next: other explicit numeric construct | 1 | 7 |
| Content-clear recordings | 20 | 39 |
| Content-clear unique native sentence IDs | 12 | 21 |
| Selection skipped: already-selected native ID/text | 14 | 14 |
| Selection skipped: protected fresh-holdout native ID/text | 0 | 19 |
| Selection skipped: quota | 0 | 0 |
| Retained original clips + new recordings | 6 + 6 | 5 + 6 |
| Final clips / shortfall to30 | 12 / **18** | 11 / **19** |

Already-selected skips include the pool counterpart of an already retained original clip, not just extra recordings. Current60 inventory separately retained6/30 validation and5/30 test, excluding24 and25 respectively. Overlapping duration-eligible reason counts: validation digits16/spoken6/foreign31/other-number1; test digits63/spoken22/foreign91/other-number9. The shortage was not an archive/network cap: every preselected recording was acquired; no sub15 fallback was activated.

**Overrestriction and screening incompleteness, now explicitly evidenced (no retroactive changes):**

- The15s floor removes931 recordings before content eligibility. Strict text uniqueness removes repeat recordings; protecting fresh30 removes19 otherwise content-clear TEST recordings. These are declared constraints, not evidence of bad audio or accents.
- `của anh` is context-insensitive: it matches7 duration-eligible validation recordings (some have other flags), even though it is also an ordinary Vietnamese pronoun phrase. Its intended source here was English saddle provenance. Common translated geographic/cultural references such as `trung hoa` (5 TEST recordings) and `tây âu` (1) were also excluded; this is stricter than merely avoiding unfamiliar foreign-name-heavy content.
- Simple conceptual quantities were rejected too: `hai ý tưởng` (1 TEST recording), `ba phần` (1), `hai thế kỷ` (2), and `một năm` (1 validation). This is stricter than a preference for *limited*, rather than zero explicit numeric constructs. Counts overlap with other reasons and are not automatic estimates of recoverable additional clips.
- Conversely the lexicon is **underinclusive**: retained `test:115` contains `Pa amb Pli`, `test:145` contains `Daru l-Badya`/`Casablanca`, and `test:10188` contains `civilis`/`civis`/`civitas`. These were missed by the sealed finite lexical rules, despite the intended foreign/transliteration-heavy exclusion. Therefore this23-clip set is **not fully compliant with the intended general-content category**, in addition to failing60. The earlier manual screening was not exhaustive. These cases are disclosed, not removed or relabeled after seeing errors; the dataset, policy and benchmark results remain unchanged.

Evidence: `outputs/benchmarks/long/fleurs-general60-20261104/cached-deletion-diagnosis.json` contains every TEST sample's metrics, all165 candidate texts/score components, exact selected IDs, deletion-span checks, reason counts and protected-input hashes. Command `.venv-benchmark/bin/python tools/diagnose_general_cached.py` completed in1.055s; `--summary` reads that saved audit without reevaluation. Strict cache joins/frozen identity checks and independent per-sample jiwer totals passed; source/policy/selection/results/caches were checked unchanged. New filename-isolated smoke RED missing-module0.000s→GREEN2 tests0.020s exercises available-longer versus missing-all-candidates cases. No inference, audio operations, downloads, retuning, sample/policy/result changes, full suites or commits in this follow-up.
