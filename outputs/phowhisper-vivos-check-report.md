# PhoWhisper VIVOS Check Report

**Evaluation date:** 2026-10-04  
**Audio directory:** `data/raw/check`  
**Model:** `vinai/PhoWhisper-base`  
**Model revision:** `7ebdb9e88f5cc5271fb88f4d642c82ff9388650e`  
**Device:** NVIDIA GeForce RTX 3050 Laptop GPU, FP16  
**Software:** Transformers 5.17.0, PyTorch 2.11.0+cu128, JiWER 4.0.0

## Executive summary

| Metric | Result |
|---|---:|
| Evaluated files | 10 |
| Total audio duration | 37.283 s |
| Reference words | 93 |
| Word errors | 11 substitutions, 0 deletions, 0 insertions |
| Corpus WER (micro) | **11.83%** |
| Corpus word match rate (`100% - WER`) | **88.17%** |
| Mean sample WER (macro) | **9.50%** |
| Exact sentence matches | **4/10 (40.00%)** |
| WER agreement with Image 1 | **10/10 (100.00%)** |
| Error-excerpt agreement with Image 2 | **5/5 (100.00%)** |

The reproduced WER percentage matches Image 1 for every sample. The five prediction excerpts visible in Image 2 also match this run. Some duration, reference-word-count, and error-count fields in Image 1 do not match the actual WAV metadata or official VIVOS transcripts; these discrepancies are documented below.

## Method

Ground-truth text was taken from the official VIVOS `prompts-test.txt.gz` file in [`AILAB-VNUHCM/vivos`](https://huggingface.co/datasets/AILAB-VNUHCM/vivos). Inference used the repository's `src.inference.transcribe()` function with Vietnamese transcription forced through the Hugging Face ASR pipeline.

Before scoring, reference and predicted text were:

1. Unicode-normalized to NFC
2. converted to lowercase
3. stripped of punctuation
4. collapsed to single spaces

WER was calculated with JiWER:

```text
WER = (substitutions + deletions + insertions) / reference words
```

## Per-file results

| Sample | Actual duration | Ground truth | PhoWhisper prediction | Ref words | Errors (S/D/I) | WER | Word match | Image WER | WER matches image |
|---|---:|---|---|---:|---:|---:|---:|---:|:---:|
| R002 | 4.062 s | tiếng cọc cạch khựng lại của những khớp sắt | tiếng cộc cực khuẩn lại của những khớp sắt | 9 | 3/0/0 | 33.33% | 66.67% | 33.33% | Yes |
| R003 | 3.688 s | cũng lên tiếng ủng hộ các kiến nghị này | cũng lên tiếng ủng hộ các ký nghị này | 9 | 1/0/0 | 11.11% | 88.89% | 11.11% | Yes |
| R012 | 4.188 s | những cơn gió mạnh và mưa đóng băng gây trơn trợt | những cơn gió mạnh và mưa đóng băng ghi trơn trợt | 11 | 1/0/0 | 9.09% | 90.91% | 9.09% | Yes |
| R027 | 4.719 s | di tích có bảy phỗng ông và hai phỗng bà được làm bằng gỗ mít | di tích có bảy phỏng ông và hai phỏng bài được làm bằng gỗ mít | 15 | 3/0/0 | 20.00% | 80.00% | 20.00% | Yes |
| R028 | 4.562 s | các chú cũng thoải mái phóng uế giữa chốn đông người mà không sợ | các chú cũng thoải mái phóng ới giữa chốn đông người mà không sợ | 14 | 1/0/0 | 7.14% | 92.86% | 7.14% | Yes |
| R034 | 1.938 s | người ta nói | người ta nói | 3 | 0/0/0 | 0.00% | 100.00% | 0.00% | Yes |
| R043 | 3.375 s | anh phú chưa bú được sữa mẹ lần nào | anh phú chưa bú được sữa mẹ lần nào | 9 | 0/0/0 | 0.00% | 100.00% | 0.00% | Yes |
| R044 | 5.219 s | ngọn lửa bạo động ở trung đông và phá vỡ lộ trình hòa bình | ngọn lửa bạo động ở trung đông và phá vỡ loại trình quà bình | 14 | 2/0/0 | 14.29% | 85.71% | 14.29% | Yes |
| R055 | 2.594 s | và rất sáng tạo | và rất sáng tạo | 4 | 0/0/0 | 0.00% | 100.00% | 0.00% | Yes |
| R057 | 2.938 s | nhưng giấc ngủ chập chờn | nhưng giấc ngủ chập chờn | 5 | 0/0/0 | 0.00% | 100.00% | 0.00% | Yes |

## Error analysis

All 11 word errors were substitutions. No words were deleted or inserted.

| Sample | Reference | Prediction | Substitutions |
|---|---|---|---:|
| R002 | cọc cạch khựng | cộc cực khuẩn | 3 |
| R003 | kiến | ký | 1 |
| R012 | gây | ghi | 1 |
| R027 | phỗng ông / phỗng bà | phỏng ông / phỏng bài | 3 |
| R028 | uế | ới | 1 |
| R044 | lộ trình hòa bình | loại trình quà bình | 2 |

The errors are concentrated in uncommon words and acoustically similar Vietnamese vowels or tones. R002 is the hardest sample, while R034, R043, R055, and R057 are exact matches after normalization.

## Comparison with the supplied images

### Matching results

- All 10 reproduced WER values equal the values shown in Image 1: **100.00% WER agreement**
- All 5 error excerpts shown in Image 2 reproduce exactly after normalization: **100.00% excerpt agreement**
- 9 of 10 displayed error counts agree. R027 is the exception

### Metadata discrepancies

| Sample | Image ref words | Official VIVOS ref words | Image errors | JiWER errors | Image duration | WAV duration |
|---|---:|---:|---:|---:|---:|---:|
| R002 | 9 | 9 | 3 | 3 | 3.2 s | 4.062 s |
| R003 | 9 | 9 | 1 | 1 | 3.8 s | 3.688 s |
| R012 | 11 | 11 | 1 | 1 | 4.1 s | 4.188 s |
| R027 | 10 | 15 | 2 | 3 | 4.5 s | 4.719 s |
| R028 | 14 | 14 | 1 | 1 | 5.0 s | 4.562 s |
| R034 | 8 | 3 | 0 | 0 | 3.1 s | 1.938 s |
| R043 | 7 | 9 | 0 | 0 | 2.9 s | 3.375 s |
| R044 | 14 | 14 | 2 | 2 | 4.6 s | 5.219 s |
| R055 | 8 | 4 | 0 | 0 | 3.5 s | 2.594 s |
| R057 | 7 | 5 | 0 | 0 | 3.0 s | 2.938 s |

For R027, standard word-level alignment finds three substitutions: both occurrences of `phỗng` become `phỏng`, and `bà` becomes `bài`. Using the official 15-word reference still gives the same WER shown in the image: `3 / 15 = 20.00%`.

## Runtime

- End-to-end runtime for 10 clips: **10.002 s**, including initial model loading
- First clip, including model startup: **7.276 s**
- Remaining 9 clips: **2.726 s** total
- End-to-end real-time factor: **0.268x**
- Warm-model real-time factor for the remaining clips: **0.082x**

Runtime was measured once and should not be treated as a benchmark distribution.

## Reproducibility notes

- The model revision is recorded because `MODEL_NAME` currently points to a moving Hugging Face repository rather than a pinned revision
- Results use one inference run; greedy ASR inference is expected to be deterministic on this setup, but no repeated-run variance study was performed
- “Word match rate” in this report is `100% - WER`; it is not a semantic-similarity score
- Image 2 contains excerpts of erroneous phrases rather than complete transcripts
