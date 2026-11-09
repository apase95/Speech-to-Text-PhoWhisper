# FLEURS60 numeric metric audit — fixed cached predictions

No model inference, training, retuning or downloads. A reproduces historical original-reference metrics before any new results are published. CER includes normalized spaces.

A: original references. B: v1 ref-only. C: v2 ref-only. D: the same conservative numeric verbalizer on ORIGINAL raw references and raw hypotheses, before NFC/lower/alphanumeric/space normalization.

| Split | Convention | Method | Words | Chars | S | D | I | Errors | WER % | Char errors | CER % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | A | gpt2_tuned_beam5 | 1267 | 5750 | 187 | 9 | 93 | 289 | 22.8098 | 785 | 13.6522 |
| validation | A | greedy | 1267 | 5750 | 201 | 11 | 78 | 290 | 22.8887 | 975 | 16.9565 |
| validation | A | beam5 | 1267 | 5750 | 201 | 8 | 101 | 310 | 24.4672 | 818 | 14.2261 |
| validation | A | gpt2_original | 1267 | 5750 | 187 | 9 | 96 | 292 | 23.0466 | 790 | 13.7391 |
| validation | A | gpt2_tuned_beam15 | 1267 | 5750 | 183 | 8 | 90 | 281 | 22.1784 | 792 | 13.7739 |
| validation | B | gpt2_tuned_beam5 | 1277 | 5796 | 184 | 9 | 83 | 276 | 21.6132 | 732 | 12.6294 |
| validation | B | greedy | 1277 | 5796 | 198 | 11 | 68 | 277 | 21.6915 | 922 | 15.9075 |
| validation | B | beam5 | 1277 | 5796 | 198 | 8 | 91 | 297 | 23.2576 | 765 | 13.1988 |
| validation | B | gpt2_original | 1277 | 5796 | 184 | 9 | 86 | 279 | 21.8481 | 737 | 12.7157 |
| validation | B | gpt2_tuned_beam15 | 1277 | 5796 | 180 | 8 | 80 | 268 | 20.9867 | 739 | 12.7502 |
| validation | C | gpt2_tuned_beam5 | 1291 | 5876 | 177 | 9 | 69 | 255 | 19.7521 | 639 | 10.8747 |
| validation | C | greedy | 1291 | 5876 | 191 | 11 | 54 | 256 | 19.8296 | 829 | 14.1082 |
| validation | C | beam5 | 1291 | 5876 | 191 | 8 | 77 | 276 | 21.3788 | 672 | 11.4364 |
| validation | C | gpt2_original | 1291 | 5876 | 177 | 9 | 72 | 258 | 19.9845 | 644 | 10.9598 |
| validation | C | gpt2_tuned_beam15 | 1291 | 5876 | 171 | 9 | 67 | 247 | 19.1325 | 648 | 11.0279 |
| validation | D | gpt2_tuned_beam5 | 1291 | 5876 | 177 | 9 | 69 | 255 | 19.7521 | 639 | 10.8747 |
| validation | D | greedy | 1291 | 5876 | 191 | 11 | 54 | 256 | 19.8296 | 829 | 14.1082 |
| validation | D | beam5 | 1291 | 5876 | 191 | 8 | 77 | 276 | 21.3788 | 672 | 11.4364 |
| validation | D | gpt2_original | 1291 | 5876 | 177 | 9 | 72 | 258 | 19.9845 | 644 | 10.9598 |
| validation | D | gpt2_tuned_beam15 | 1291 | 5876 | 171 | 9 | 67 | 247 | 19.1325 | 648 | 11.0279 |
| test | A | gpt2_tuned_beam5 | 1353 | 6085 | 167 | 11 | 57 | 235 | 17.3688 | 654 | 10.7477 |
| test | A | greedy | 1353 | 6085 | 183 | 14 | 66 | 263 | 19.4383 | 688 | 11.3065 |
| test | A | beam5 | 1353 | 6085 | 181 | 11 | 59 | 251 | 18.5514 | 661 | 10.8628 |
| test | A | gpt2_original | 1353 | 6085 | 166 | 11 | 60 | 237 | 17.5166 | 651 | 10.6984 |
| test | A | gpt2_tuned_beam15 | 1353 | 6085 | 161 | 12 | 55 | 228 | 16.8514 | 646 | 10.6163 |
| test | B | gpt2_tuned_beam5 | 1364 | 6129 | 162 | 11 | 46 | 219 | 16.0557 | 601 | 9.8058 |
| test | B | greedy | 1364 | 6129 | 178 | 14 | 55 | 247 | 18.1085 | 635 | 10.3606 |
| test | B | beam5 | 1364 | 6129 | 176 | 11 | 48 | 235 | 17.2287 | 608 | 9.9201 |
| test | B | gpt2_original | 1364 | 6129 | 161 | 11 | 49 | 221 | 16.2023 | 598 | 9.7569 |
| test | B | gpt2_tuned_beam15 | 1364 | 6129 | 156 | 12 | 44 | 212 | 15.5425 | 594 | 9.6916 |
| test | C | gpt2_tuned_beam5 | 1387 | 6242 | 160 | 20 | 32 | 212 | 15.2848 | 577 | 9.2438 |
| test | C | greedy | 1387 | 6242 | 178 | 23 | 41 | 242 | 17.4477 | 621 | 9.9487 |
| test | C | beam5 | 1387 | 6242 | 176 | 20 | 34 | 230 | 16.5826 | 593 | 9.5002 |
| test | C | gpt2_original | 1387 | 6242 | 161 | 20 | 35 | 216 | 15.5732 | 583 | 9.3400 |
| test | C | gpt2_tuned_beam15 | 1387 | 6242 | 154 | 21 | 30 | 205 | 14.7801 | 569 | 9.1157 |
| test | D | gpt2_tuned_beam5 | 1382 | 6217 | 160 | 15 | 32 | 207 | 14.9783 | 552 | 8.8789 |
| test | D | greedy | 1382 | 6217 | 178 | 18 | 41 | 237 | 17.1491 | 596 | 9.5866 |
| test | D | beam5 | 1382 | 6217 | 176 | 15 | 34 | 225 | 16.2808 | 568 | 9.1362 |
| test | D | gpt2_original | 1382 | 6217 | 161 | 15 | 35 | 211 | 15.2677 | 558 | 8.9754 |
| test | D | gpt2_tuned_beam15 | 1382 | 6217 | 154 | 16 | 30 | 200 | 14.4718 | 544 | 8.7502 |

## Changes versus A

| Split | Convention | Method | Error delta | Word delta | Character delta |
|---|---|---|---:|---:|---:|
| validation | B | gpt2_tuned_beam5 | -13 | +10 | +46 |
| validation | B | greedy | -13 | +10 | +46 |
| validation | B | beam5 | -13 | +10 | +46 |
| validation | B | gpt2_original | -13 | +10 | +46 |
| validation | B | gpt2_tuned_beam15 | -13 | +10 | +46 |
| validation | C | gpt2_tuned_beam5 | -34 | +24 | +126 |
| validation | C | greedy | -34 | +24 | +126 |
| validation | C | beam5 | -34 | +24 | +126 |
| validation | C | gpt2_original | -34 | +24 | +126 |
| validation | C | gpt2_tuned_beam15 | -34 | +24 | +126 |
| validation | D | gpt2_tuned_beam5 | -34 | +24 | +126 |
| validation | D | greedy | -34 | +24 | +126 |
| validation | D | beam5 | -34 | +24 | +126 |
| validation | D | gpt2_original | -34 | +24 | +126 |
| validation | D | gpt2_tuned_beam15 | -34 | +24 | +126 |
| test | B | gpt2_tuned_beam5 | -16 | +11 | +44 |
| test | B | greedy | -16 | +11 | +44 |
| test | B | beam5 | -16 | +11 | +44 |
| test | B | gpt2_original | -16 | +11 | +44 |
| test | B | gpt2_tuned_beam15 | -16 | +11 | +44 |
| test | C | gpt2_tuned_beam5 | -23 | +34 | +157 |
| test | C | greedy | -21 | +34 | +157 |
| test | C | beam5 | -21 | +34 | +157 |
| test | C | gpt2_original | -21 | +34 | +157 |
| test | C | gpt2_tuned_beam15 | -23 | +34 | +157 |
| test | D | gpt2_tuned_beam5 | -28 | +29 | +132 |
| test | D | greedy | -26 | +29 | +132 |
| test | D | beam5 | -26 | +29 | +132 |
| test | D | gpt2_original | -26 | +29 | +132 |
| test | D | gpt2_tuned_beam15 | -28 | +29 | +132 |

## Local changes and remaining mismatches

All sample-level metrics, normalized texts, unresolved digits and independent edit spans are in `metrics.json`. Below: original GPT2 only; every changed sample is listed (A → B → C → D word errors).
- test:121: 18 → 10 → 10 → 10
  Remaining D edits: `[{"type": "insert", "reference": "", "hypothesis": "croatia phát lên"}, {"type": "substitute", "reference": "cristina fernandez de kirchner", "hypothesis": "disney đề khích nơi"}, {"type": "substitute", "reference": "plata", "hypothesis": "lata"}, {"type": "substitute", "reference": "buenos aires", "hypothesis": "ronaldo eric"}]`
- test:153: 19 → 19 → 15 → 15
  Remaining D edits: `[{"type": "substitute", "reference": "trượt tuyết", "hypothesis": "trực tiếp"}, {"type": "insert", "reference": "", "hypothesis": "toàn"}, {"type": "substitute", "reference": "được", "hypothesis": "cừu"}, {"type": "delete", "reference": "cuộc", "hypothesis": ""}, {"type": "substitute", "reference": "mươi", "hypothesis": "lâm"}, {"type": "delete", "reference": "lăm", "hypothesis": ""}, {"type": "substitute", "reference": "trượt tuyết", "hypothesis": "trực tiếp"}, {"type": "insert", "reference": "", "hypothesis": "gan"}, {"type": "substitute", "reference": "giant slalom", "hypothesis": "la lam"}, {"type": "substitute", "reference": "giành", "hypothesis": "nhìn"}, {"type": "substitute", "reference": "trong bảng", "hypothesis": "trên bãi"}]`
- test:266: 2 → 2 → 7 → 2
  Remaining D edits: `[{"type": "delete", "reference": "oravec 2002", "hypothesis": ""}]`
- test:472: 13 → 13 → 13 → 13
  Remaining D edits: `[{"type": "substitute", "reference": "nghìn không", "hypothesis": "ngàn lẻ"}, {"type": "delete", "reference": "trăm linh", "hypothesis": ""}, {"type": "substitute", "reference": "goma", "hypothesis": "gomez"}, {"type": "substitute", "reference": "san bằng bởi", "hypothesis": "sang bàn bến"}, {"type": "insert", "reference": "", "hypothesis": "nigeron"}, {"type": "substitute", "reference": "nyiragongo", "hypothesis": "google"}, {"type": "substitute", "reference": "phun", "hypothesis": "phong"}, {"type": "substitute", "reference": "chôn vùi", "hypothesis": "trong bồi"}]`
- test:653: 10 → 3 → 3 → 3
  Remaining D edits: `[{"type": "substitute", "reference": "cường độ", "hypothesis": "quần đổ"}, {"type": "substitute", "reference": "dẫm", "hypothesis": "giảm"}]`
- test:661: 18 → 17 → 16 → 16
  Remaining D edits: `[{"type": "insert", "reference": "", "hypothesis": "ma la ba la chu ra"}, {"type": "substitute", "reference": "malar balasubramanian", "hypothesis": "mi an"}, {"type": "delete", "reference": "mươi", "hypothesis": ""}, {"type": "substitute", "reference": "blue", "hypothesis": "lúa"}, {"type": "delete", "reference": "ash", "hypothesis": ""}, {"type": "substitute", "reference": "vùng", "hypothesis": "quần"}, {"type": "insert", "reference": "", "hypothesis": "shin shin"}, {"type": "substitute", "reference": "cincinnati", "hypothesis": "natin"}, {"type": "substitute", "reference": "lăm", "hypothesis": "lâm"}]`
- test:813: 7 → 7 → 2 → 2
  Remaining D edits: `[{"type": "delete", "reference": "mươi", "hypothesis": ""}, {"type": "substitute", "reference": "ngoại", "hypothesis": "và"}]`
- validation:126: 8 → 8 → 6 → 6
  Remaining D edits: `[{"type": "substitute", "reference": "nghìn", "hypothesis": "ngàn"}, {"type": "substitute", "reference": "xơ", "hypothesis": "sơ"}, {"type": "insert", "reference": "", "hypothesis": "emm"}, {"type": "substitute", "reference": "ms", "hypothesis": "ớt"}, {"type": "substitute", "reference": "xơ", "hypothesis": "sơ"}, {"type": "substitute", "reference": "kinh", "hypothesis": "khinh"}]`
- validation:151: 8 → 2 → 2 → 2
  Remaining D edits: `[{"type": "substitute", "reference": "bơi", "hypothesis": "hơn"}, {"type": "substitute", "reference": "khối", "hypothesis": "khói"}]`
- validation:231: 12 → 5 → 5 → 5
  Remaining D edits: `[{"type": "substitute", "reference": "sentinel", "hypothesis": "arsenal"}, {"type": "substitute", "reference": "vinson massif", "hypothesis": "vincent messi"}, {"type": "substitute", "reference": "vinson", "hypothesis": "vincent"}, {"type": "substitute", "reference": "nghìn", "hypothesis": "ngàn"}]`
- validation:237: 24 → 24 → 13 → 13
  Remaining D edits: `[{"type": "substitute", "reference": "nghìn", "hypothesis": "ngàn"}, {"type": "insert", "reference": "", "hypothesis": "với sa ly"}, {"type": "substitute", "reference": "versaille", "hypothesis": "e"}, {"type": "insert", "reference": "", "hypothesis": "mười"}, {"type": "substitute", "reference": "xvi", "hypothesis": "sáu"}, {"type": "substitute", "reference": "antoinette", "hypothesis": "anthony"}, {"type": "substitute", "reference": "therese", "hypothesis": "terry"}, {"type": "substitute", "reference": "madam elizabeth", "hypothesis": "manon alibaba"}, {"type": "substitute", "reference": "hoảng sợ", "hypothesis": "khoảng sở"}]`
- validation:260: 9 → 9 → 5 → 5
  Remaining D edits: `[{"type": "substitute", "reference": "thụy", "hypothesis": "thủy"}, {"type": "insert", "reference": "", "hypothesis": "luke"}, {"type": "substitute", "reference": "klöcker ehrenstrahl", "hypothesis": "a remton"}, {"type": "substitute", "reference": "guinness", "hypothesis": "reed"}]`
- validation:73: 23 → 23 → 19 → 19
  Remaining D edits: `[{"type": "substitute", "reference": "trăm linh", "hypothesis": "trong lẻ"}, {"type": "insert", "reference": "", "hypothesis": "sẽ hấp panh"}, {"type": "substitute", "reference": "chhappan bhog trong ấn độ", "hypothesis": "và hấp cho món đỗ"}, {"type": "substitute", "reference": "sáu", "hypothesis": "sá"}, {"type": "substitute", "reference": "dâng", "hypothesis": "dân"}, {"type": "substitute", "reference": "thần", "hypothesis": "thành"}, {"type": "substitute", "reference": "được", "hypothesis": "để"}, {"type": "substitute", "reference": "vụ", "hypothesis": "giữ"}, {"type": "insert", "reference": "", "hypothesis": "ba ba"}, {"type": "substitute", "reference": "baba shyam", "hypothesis": "xi am"}]`

Confirmed unchanged lexical snippets (beam5, D):
- validation:151: `khối thịnh vượng` versus `khói thịnh vượng`.
- test:524: `lông vũ` versus `lâm vũ`.

## Grammar, provenance and limitations

C and D agree on validation. On test, D has five fewer errors than C for every method solely because the unsupported bare citation year in test:266 stays as 2002 instead of becoming six spoken words. The hypothesis omits that citation. This is a denominator/annotation-scope difference, NOT five additional recognized words or evidence that symmetric normalization repaired a prediction.

Only explicit valid ngày/tháng/năm dates, năm years, thứ ordinals, age/count noun contexts, total/subset counts and numeric physical units are expanded. Dot-separated groups of three mean thousands; comma decimals require units. Leading zeros, malformed notation, slash dates, bare citation years, Roman numerals, abbreviations and alphanumeric identifiers are not guessed. No nghìn/ngàn, linh/lẻ, một/mốt or omitted-hundred equivalences are collapsed: D is a conservative lower-bound normalization, not full numeric invariance. No sample-ID rules.

Original manifests are reconstructed by restoring reference_raw and removing only the two annotation fields; their exact serialized hashes must equal old caches. Strict existing artifact/config/source checks remain intact. Original freezes, validation source hashes, winner fingerprint, protocol/checkpoint identities and historical final hypotheses are checked. The fixed-hypotheses artifact seals chosen candidate IDs, RAW texts and input hashes. No reference enters candidate scoring. These fingerprints detect corruption, not adversarial rewriting of all evidence.

S/D/I use deterministic diagonal-before-deletion-before-insertion backtrace. Independent jiwer word/character total errors match every sample and corpus result. Equally optimal jiwer S/D/I can differ; both are saved. Jiwer edit spans are illustrative, not causal attribution.

Percentage changes mix numerator and denominator changes. A→B→C→D differences are not additive causal error categories and do not improve the model. Text-only references have not been audio audited. Near-homophone substitutions such as khối/khói and lông/lâm are lexical examples, not proof of acoustic origin. Small, previously seen test set; no generalization claim.

Verified inventory: `{"samples": 60, "original_digit_refs": 13, "original_numeric_expressions": 23, "v1_changed_refs": 5, "v2_additional_changed_refs": 9, "v2_remaining_digit_refs": 0}`. V1 covers seven measurement spans; v2 additionally changes 16 numeric expressions (the three date components form one span). Roman/abbreviation ambiguities remain. V1 backup: `data/benchmark/long-audio/reference-backups/1793755881486055602/`. Previous actual apply fully validated 60 audio hashes/PCM/durations/leak checks, 1236.24s total; repeated apply changed zero files.

Unresolved D forms: `[{"id": "test:266", "method": "gpt2_tuned_beam5", "reference": ["2002"], "hypothesis": []}, {"id": "test:266", "method": "greedy", "reference": ["2002"], "hypothesis": []}, {"id": "test:266", "method": "beam5", "reference": ["2002"], "hypothesis": []}, {"id": "test:266", "method": "gpt2_original", "reference": ["2002"], "hypothesis": []}, {"id": "test:266", "method": "gpt2_tuned_beam15", "reference": ["2002"], "hypothesis": []}]`

Artifacts: `outputs/benchmarks/long/numeric-metric-v2-20261104-final` (`fixed-hypotheses.json`, `metrics.json`, `independent-audit.json`, `inventory.json`, `report.md`). Reevaluation wall time 11.625s; no audio/model stage. PhoBERT and new-reference candidate oracles intentionally omitted; fixed greedy, beam5, original GPT2, tuned beam5 and selected beam15 covered.

Next priorities: human audio/transcript audit of number readings and unresolved citation/abbreviation forms, then lexical/proper-name and candidate-generation errors on fresh held-out speech. Further whole-candidate rescoring cannot recover words absent from candidates.

## Reproduction and verification

Run `.venv-benchmark/bin/python tools/reevaluate_numeric_metrics.py --output NEW_DIRECTORY` (refuses existing output). Relative output paths are resolved; original artifacts are read-only. The final invocation used `outputs/benchmarks/long/numeric-metric-v2-20261104-final`.

New tests demonstrated RED (missing module ImportError), then GREEN: 7 tests, 0.021s. Synthetic fixtures exercise integer/unit/date/year/count/ordinal/decimal rules, invalid-date and identifier preservation, symmetric equality/idempotence, S/D/I and empty edges, original-manifest reconstruction/hash rejection, swapped reference rejection and finite fixed-candidate scoring. No ignored-data dependency.

Exact filename-limited command: `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v`. Existing scoped results: test_reference_normalization.py 4 passed (0.005s), test_long_audio_smoke.py 15 (0.063s), test_gpt2_prefix_smoke.py 2 (0.000s), test_nbest_ranker_smoke.py 2 (0.000s), test_gpt2_optimization.py 6 passed / 1 optional historical audit skipped (0.013s). New reevaluation independently audits every evaluated sample and corpus row with jiwer; no full suite run.

An intermediate relative-output run completed metrics/audits but failed report rendering; its partial directory is retained, not the final result. The path-resolution bug was fixed before the final run. All original input hashes are checked again before publication. No commits or pushes; unrelated working-tree files preserved.

Final successful command: `.venv-benchmark/bin/python tools/reevaluate_numeric_metrics.py --output outputs/benchmarks/long/numeric-metric-v2-20261104-final`; 11.625s internal / 11.720s process wall, 40 corpus rows and 1200 per-sample rows independently audited. A subsequent additional synthetic no-reselection test passed: final `test_numeric_metrics.py` result **8 passed (0.019s)**. All scoped files total **37 passed, one optional historical audit skipped**. `git diff --check` passed. No full-suite/integration/model run was needed; actual cached evaluation is explicitly required by the handoff and exceeds the <2s isolated-test target.
