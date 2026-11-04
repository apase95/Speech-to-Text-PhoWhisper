# GPT2 optimization — FLEURS60 (2026-11-04)

> **HISTORICAL: original reference orthography.** Active manifests now use
> `fleurs60-spoken-v1`. All numbers and cached artifacts below remain historical,
> not scores for the new references. See [normalization](reference-normalization-fleurs60.md).

> **Portable tests update:** default optimization unit tests now use synthetic
> temporary fixtures, with no ignored manifest/output dependency. The historical
> independent audit is opt-in via `FLEURS60_PERSISTED_AUDIT=1`; prerequisites and
> exact command are in the linked normalization document. Prior test timings below
> describe the historical artifact-dependent tests, not current checkout defaults.



Executed offline with the original v3 scorer/decoder; original artifacts untouched. 30 validation clips (1267 words), 30 test clips (1353 words). NFC/lowercase/alphanumeric normalization; CER includes spaces.



## Protocol and selection



`score = avg_logprob + alpha * mean_LM + beta * normalized_word_count`. Alpha grid: [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]; beta grid: [-0.02, -0.01, 0, 0.01, 0.02]. Beta is log-score units per whitespace-delimited normalized word (not LM token). No inferred token counts or sum-LM variants. All 50 configurations per beam are persisted. Tie-break: word errors, character errors, beta zero, smaller absolute beta, smaller alpha, beta. Beam selection: validation word errors, character errors, smaller beam. Beam15 won before test access. Test comparisons are exploratory on an already-seen test set.



Freeze files bind exact protocol, grid, scoring formula, revision, validation manifest/score/N-best hashes. Candidate source identity and finite scores are checked against hashed N-best; test requires identical invariant protocol. Fingerprints detect corruption, not cryptographic authenticity against an adversary who can rewrite all artifacts.



GPT2 revision `65818d14816b42be09e2201933bf07106d9a2647`; checkpoint `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d`. Runtime and tokenizer vocabulary fingerprints are preserved in each freeze file.



| Split | Beam | Method | WER % | CER % | S | D | I |

|---|---:|---|---:|---:|---:|---:|---:|

| validation | 5 | beam | 24.4672 | 14.2261 | 201 | 8 | 101 |

| validation | 5 | baseline | 23.0466 | 13.7391 | 187 | 9 | 96 |

| validation | 5 | tuned | 22.8098 | 13.6522 | 187 | 9 | 93 |

| validation | 5 | word_oracle | 22.2573 | 13.3391 | 181 | 8 | 93 |

| validation | 5 | char_oracle | 22.6519 | 13.2174 | 181 | 8 | 98 |

| validation | 10 | beam | 24.4672 | 14.4000 | 202 | 6 | 102 |

| validation | 10 | baseline | 22.6519 | 14.1391 | 188 | 7 | 92 |

| validation | 10 | tuned | 22.5730 | 14.1913 | 188 | 8 | 90 |

| validation | 10 | word_oracle | 21.7837 | 13.7217 | 178 | 9 | 89 |

| validation | 10 | char_oracle | 22.4152 | 13.2696 | 177 | 8 | 99 |

| validation | 15 | beam | 24.4672 | 14.3130 | 202 | 6 | 102 |

| validation | 15 | baseline | 22.4152 | 13.7913 | 185 | 6 | 93 |

| validation | 15 | tuned | 22.1784 | 13.7739 | 183 | 8 | 90 |

| validation | 15 | word_oracle | 21.3891 | 13.3739 | 174 | 8 | 89 |

| validation | 15 | char_oracle | 22.2573 | 13.0609 | 176 | 8 | 98 |

| test | 5 | beam | 18.5514 | 10.8628 | 181 | 11 | 59 |

| test | 5 | baseline | 17.5166 | 10.6984 | 166 | 11 | 60 |

| test | 5 | tuned | 17.3688 | 10.7477 | 167 | 11 | 57 |

| test | 5 | word_oracle | 16.8514 | 10.1726 | 159 | 12 | 57 |

| test | 5 | char_oracle | 16.9254 | 10.0575 | 160 | 12 | 57 |

| test | 15 | beam | 18.7731 | 10.9449 | 182 | 11 | 61 |

| test | 15 | baseline | 17.0732 | 10.6327 | 159 | 12 | 60 |

| test | 15 | tuned | 16.8514 | 10.6163 | 161 | 12 | 55 |

| test | 15 | word_oracle | 16.1123 | 10.1726 | 158 | 10 | 50 |

| test | 15 | char_oracle | 16.6297 | 9.4659 | 158 | 11 | 56 |



## Diversity and per-sample comparisons



- validation beam5: 150 candidates, 150 unique normalized strings (summed within samples), mean pairwise normalized word edit distance 0.06161.

- validation beam5 baseline versus beam: {'win': 17, 'tie': 10, 'loss': 3}.

- validation beam5 tuned versus beam: {'win': 17, 'tie': 11, 'loss': 2}.

- validation beam10: 300 candidates, 300 unique normalized strings (summed within samples), mean pairwise normalized word edit distance 0.07318.

- validation beam10 baseline versus beam: {'win': 19, 'tie': 7, 'loss': 4}.

- validation beam10 tuned versus beam: {'win': 18, 'tie': 9, 'loss': 3}.

- validation beam15: 450 candidates, 449 unique normalized strings (summed within samples), mean pairwise normalized word edit distance 0.07994.

- validation beam15 baseline versus beam: {'win': 18, 'tie': 9, 'loss': 3}.

- validation beam15 tuned versus beam: {'win': 18, 'tie': 11, 'loss': 1}.

- test beam5: 150 candidates, 149 unique normalized strings (summed within samples), mean pairwise normalized word edit distance 0.05730.

- test beam5 baseline versus beam: {'win': 9, 'tie': 21, 'loss': 0}.

- test beam5 tuned versus beam: {'win': 10, 'tie': 20, 'loss': 0}.

- test beam15: 450 candidates, 449 unique normalized strings (summed within samples), mean pairwise normalized word edit distance 0.07430.

- test beam15 baseline versus beam: {'win': 13, 'tie': 15, 'loss': 2}.

- test beam15 tuned versus beam: {'win': 15, 'tie': 13, 'loss': 2}.



## Attribution and findings



1. Candidate-set limitations dominate: beam5 validation 310 → baseline GPT2 292 → oracle 282 errors; test 251 → 237 → 228. The oracle residual is not correctable by whole-candidate selection in those sets; it is not evidence of exclusively acoustic errors. For tuned beam15, validation 281 → oracle 271; test 228 → oracle 218. Substitutions dominate all these totals, followed by insertions, not deletions. Counts alone do not establish cause.

2. Larger beam improves word oracle: validation 282 / 276 / 271 errors for beams5/10/15. Tuned validation errors 289 / 286 / 281. However beam top1 stays at 310 and test beam15 top1 regresses from 251 to 254. Word-oracle CER also regresses across validation beams: word and character oracles are separately labeled, with ties resolved by original decoder order.

3. Length/alpha tuning gives small gains: beam5 validation 292 → 289, test 237 → 235; beam15 validation 284 → 281, test 231 → 228. Beam5 test CER worsens despite WER improvement. Selected configs: beam5 (alpha .3, beta -.02), beam10 (.4, -.02), beam15 (.3, -.02). Beta hits the predefined lower boundary; expanding the grid would be a new validation experiment, not grounds to retune test.



Illustrative validation: reference “bơi 100m và 200m” versus beam “hơn một trăm mét và hai trăm mét”; reference “khối thịnh vượng” versus “khói thịnh vượng”. Test: “lông vũ” versus “lâm vũ”. These illustrate numeric formatting and lexical differences, not verified causal diagnoses. Full normalized references/hypotheses and per-sample S/D/I are in diagnostic JSON.



Next: manually audit transcripts and number-format conventions, then evaluate candidate-generation changes on fresh held-out speech. Remaining oracle errors outweigh selection headroom; another rescoring-only grid cannot fix missing candidates.



## Reproduction and timing



Run `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/gpt2_optimization.py` (refuses an existing output directory). Report/CSV generation: same interpreter and script with `--report`. No training, PhoBERT, credentials, packages or downloads requested. CPU GPT2, existing GPU checkpoint for acoustic decode. Each run used a 1800-second outer bound; sequential model stages only.



Measured successful stage wall seconds: `{"decode-10-validation": 65.39086970599965, "score-10-validation": 44.5255425560008, "decode-15-validation": 88.04744708500039, "score-15-validation": 70.5063064650003, "decode-15-test": 108.19669957399856, "score-15-test": 76.39416453300146}`.



## Caveats



Human transcript audit pending; training overlap unknown; small read-speech clips under30 seconds; no significance/generalization or long-context claim. Test was already seen historically. Beam10 test was intentionally skipped: only cached-tuned beam5 and the validation-selected winner were evaluated. No sum-LM/token-normalization grid because sufficient token statistics were not stored. Original smoke fixtures are stale relative to v3: the unchanged test_long_audio_smoke.py fixture declares v2 and omits prefix/vocabulary fields.



## Verification



All tests used `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v`; standard unittest discovery restricts execution to that filename. New test RED: missing module ImportError (0.000s). GREEN: 3 passed (0.060s), covering alignment, empty-reference/hypothesis edges, freeze tampering and cached-source missing/forged/protocol/source-hash rejection. test_gpt2_prefix_smoke.py: 2 passed (0.000s); test_nbest_ranker_smoke.py: 2 passed (0.000s unittest timing, excluding imports). test_long_audio_smoke.py: 12 passed, 3 errors (0.037s), all at the unchanged v2 fixture versus current v3 schema; not fixed outside authorized new files. Environment checker passed 138 effective pins (0.441s). Both split manifest validations passed full audio/hash checks; validation 611.34s audio, test 624.9s audio. Individual manifest-command wall durations were not instrumented. `git diff --check` passed. Full suites skipped; only explicitly requested isolated files run. No commits. Known coverage limitation: no dedicated malformed test-freeze end-to-end unit case; actual frozen test execution checked protocol equality successfully.


## Final independent verification (supersedes historical test status above)



All saved word/character error totals and WER/CER independently reproduced with jiwer. Persisted S/D/I independently aggregated from sample records. Jiwer may choose a different equally optimal backtrace, so its individual S/D/I can differ; both are retained in the audit artifact. Reference totals: validation **1267 words / 5750 characters**; test **1353 words / 6085 characters**. Characters include normalized spaces. Evidence: `independent-audit.json`.



Original GPT2 means cached beam5 alpha .4; selected means validation-selected beam15 alpha .3, beta -.02. Validation: 292 → 281 errors (−11), WER 23.0466 → 22.1784 (−0.8682 percentage points), CER 13.7391 → 13.7739 (+0.0348 points). Test: 237 → 228 (−9), WER 17.5166 → 16.8514 (−0.6652 points), CER 10.6984 → 10.6163 (−0.0822 points). Versus cached beam5 top1: validation 310 → 281 (−29; −2.2889 WER points), test 251 → 228 (−23; −1.6999 WER points). Versus same-beam15 top1: validation 310 → 281 (−29), test 254 → 228 (−26; −1.9217 WER points).



Concrete normalized snippets, directly checked against saved hypotheses:



- validation:151, beam5 top1: reference `bơi 100m và 200m` → hypothesis `hơn một trăm mét và hai trăm mét`; isolated-span alignment S=3, D=0, I=4.

- validation:151, beam5 top1: reference `khối thịnh vượng` → hypothesis `khói thịnh vượng`; isolated-span alignment S=1, D=0, I=0.

- test:524, beam5 top1: reference `lông vũ` → hypothesis `lâm vũ`; isolated-span alignment S=1, D=0, I=0.



The first snippet exhibits numeric orthography differences and additional hypothesis words; it does not prove number formatting caused the error. Isolated-span alignments need not equal the full-utterance alignment attribution.



Only stale v2 fixture fields in test_long_audio_smoke.py were migrated to strict v3. Production schema/checks unchanged. Added tests exercise validation-only selection/tie-break, frozen protocol mismatch, reuse of validation manifest rejection, persisted S/D/I and reference totals, and winner/freeze persistence before test access. The orchestration-order test isolates model/I/O boundaries; the saved-metric audit uses independent jiwer rather than the production backtrace.



No model rerun. Final scoped commands use `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v` for the four exact filenames; no full suite. Final results recorded below.

Final GREEN: `test_long_audio_smoke.py` 15/15 (0.059s), `test_gpt2_prefix_smoke.py` 2/2 (0.000s), `test_nbest_ranker_smoke.py` 2/2 (0.000s), `test_gpt2_optimization.py` 6/6 (3.055s). These are unittest-reported durations, not total shell wall time. Optimization tests exceed the <2s target with independent jiwer audit and orchestration checks. New guard test RED: missing-helper AttributeError; GREEN followed extraction of the unchanged selection/protocol checks into tested helpers. Final `.venv-benchmark/bin/python tools/gpt2_optimization.py --audit` passed; wall duration not instrumented. `git diff --check` passed. All 25 scoped tests pass; no lingering scoped regression. No model or full-suite rerun. The stale scoring-name mutation now correctly expects rejection at strict schema validation, rather than expecting it to survive until protocol comparison.
