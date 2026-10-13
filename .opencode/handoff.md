# PhoWhisper Kaggle training setup implementation

## Selection and task lock
- Project: /home/hodangthaiduy/Workspace/AI/phowhisper-pytorch
- Current revision: d5c4ac772a8a45dba121ef91be930941e8b06aa3; git status clean before this artifact.
- Benchmark recorded revision: f0a0019f7dcd95dcdf36af1d2439adfb77a9f3af.
- OpenArms selection: python-ai, huggingface-expert; already loaded by controller. No skill file access authorized through this handoff.
- Provider/model: openai/gpt-6.1-sol. Scout agent explore, ID ses_ecadebefaffehdnZbEMO2xAQKL. Implementer/scoped reviewer general, ID ses_ecadd571cffeC03xswy79grnJT.
- Task lock: read code/reports and advise on large-scale VietBUD500 fine-tuning on Kaggle. No source edits, training, downloads, dependency changes, or spawned tasks.

## Scope for fresh scoped reviewer
Read this brief and named source files only as needed. Provide an evidence-grounded training recommendation in Vietnamese or English; do not implement yet. Focus on compatibility between existing converted OpenAI Whisper inference and Hugging Face fine-tuning, dataset leakage/normalization, Kaggle feasibility and resumability, and realistic expected improvements. No promises of target WER. Distinguish hypotheses from measured results. Return recommendations, risks, necessary user questions, and a staged validation plan. Do not undertake open-ended discovery.

## Progress and findings
- Scout completed bounded read-only project/report inspection. Controller verified current git revision and clean working tree.
- Current runtime is OpenAI Whisper/PyTorch using converted models/phowhisper-base.pt, CUDA FP16 when available. Defaults: greedy, without_timestamps=True, condition_on_previous_text=False, no silence padding.
- No existing fine-tuning entry point found. requirements.txt is inference-oriented, no Transformers training pipeline.
- Project downloader identifies tuandaodev/viet-bud500-formatted, pipe-delimited audio_file|text|speaker_name. Actual user's intended Kaggle dataset version/schema remains unknown.
- 1000 test clips / 2544.346 seconds: greedy no timestamps WER 10.49%, CER 6.10%, RTF .253; beam5 WER 9.57%, CER 5.55%, RTF .558. Beam5 exact sentences 451/1000, zero inference failures.
- Same test greedy +200ms silence padding: WER 11.99%, CER 7.62%, RTF .247. Timestamp baseline: WER 41.78%, CER 34.14%, RTF .360. Do not recommend these as demonstrated improvements.
- Separate 250 dev clips: greedy WER 11.11%, CER 6.42%, RTF .245; beam3 10.36%, 6.03%, .426; beam5 10.32%, 6.01%, .578.
- Reports measured RTX3050 Laptop, not Kaggle hardware. Short pre-segmented clips do not establish long-audio or punctuation quality. Metrics normalize punctuation away, retaining Vietnamese tone marks.
- Existing tests passing in user history are software correctness, not training/ASR quality evidence. No checks run during review.
- Scoped review complete: recommend bounded HF training pilot, split/provenance audit, persistent resume test, and HF-to-OpenAI export parity before scaling. Named tools/convert_weights.py absent; README points to tools/convert_phowhisper.py, which was not reviewed. Downloader merges train/test IDs in one dictionary; ID collisions could overwrite entries, but actual collision/leakage is not established. Prior PhoWhisper exposure to VietBUD500 remains unverified.

## Named inputs
- src/inference.py
- tools/convert_weights.py (if path exists; do not search broadly if absent)
- tools/download_viet_bud500_subset.py
- requirements.txt
- README.md
- benchmarks/run.py
- benchmarks/README.md
- outputs/viet-bud500-pytorch-benchmark-check.md
- outputs/benchmarks/viet-bud500-beam5-20261006T010052Z/summary.json
- data/benchmark/viet-bud500/metadata_train.csv (read header only if needed)

## Mandatory project checks
- For future implementation: python -m unittest discover -s tests -v
- Fixed tiny training/validation smoke test before large-scale training; no existing training command yet.
- Compare exported checkpoint with HF checkpoint on fixed audio before trusting existing benchmark runtime.
- Evaluate held-out manifests with same decode and normalization settings and report WER/CER/deletions plus runtime.
- This review is read-only: do not run expensive checks or claim verification not performed.

## Open questions
Kaggle GPU/VRAM and session budget; exact dataset/version and splits; whether PhoWhisper's prior training overlaps this benchmark; user domain and fast-speech examples; accuracy vs latency goal; LoRA vs full training preference and checkpoint persistence.

## Current user task (supersedes previous read-only scope)
User requests setup and testing on HF/Kaggle for dataset https://www.kaggle.com/datasets/tuandaodev/viet-bud500-formatted, T4 x2, 5–10h or less. Error type: full transcript but substitutions in fast/coarticulated and similar-sounding speech. Implement additive training artifacts and Vietnamese instructions now; no training/downloads here. Preserve existing inference and benchmarks. Provider/model remains openai/gpt-6.1-sol; python-ai/huggingface-expert task lock now Kaggle setup/test implementation. Current scout explore ID ses_eca9d7841ffeFjdKYQjgRL6PLj; fresh implementer general ID ses_eca9c3b2dffetc0xAH0d4BIUou; scoped reviewer general ID ses_eca97384fffenuj8Qr2QCAGcFA.

## Refined evidence
Official HF model card and official https://raw.githubusercontent.com/VinAIResearch/PhoWhisper/main/README.md describe 844h multilingual-Whisper Vietnamese fine-tuning; they do NOT establish user's claim that original training is VIVOS-only or that Viet-BUD500 was unseen. State uncertainty clearly. README reports VIVOS evaluation, not exclusive provenance.
tools/convert_phowhisper.py exists and converts vinai/PhoWhisper-base HF weights to OpenAI names. Don't modify conversion/runtime in this task; document post-training conversion validation as separate work.
Downloader assumes metadata_train.csv and metadata_test.csv at root, pipe fields audio_file/text/speaker_name, wavs/audio<ID>.wav; actual Kaggle mount must be checked. Benchmark test audio IDs 1–1000 and dev IDs 1001–1250 are protected; README forbids training on benchmark samples. WER normalization NFC/lowercase/punctuation stripping/whitespace, tones retained.

## Implementation deliverables
Create an additive self-contained kaggle_training directory: pinned training dependencies (do not replace Kaggle torch), runnable Python script and Vietnamese setup/test/training/evaluation guide. Use HF WhisperForConditionalGeneration/WhisperProcessor, seq2seq Trainer and accelerate/torchrun genuine 2-GPU DDP (no device_map=auto). Full fine-tune base is reasonable measured starting point; no unneeded LoRA complexity. Single-GPU smoke first; fp16 for T4, gradient checkpointing, use_cache false, explicit language/task/no timestamps and greedy validation. Reproducible seed and HF commit revision resolved and saved. No tokens in source; HF optional for public download, secrets only for optional uploads.
Robust bounded-memory on-demand audio loading and resampling to 16k mono, reject invalid/empty/nonfinite or >30s data explicitly (no silent truncate). Correct decoder-start label stripping and -100 padding. Dataset preparation must be inspectable: validate schema/layout, reject duplicates/conflicting labels; train from official train only, test untouched; speaker-group validation when possible; exclude protected test/dev IDs and existing benchmark manifest paths/hashes if practicable. Be transparent about exclusions/sample caps and residual near-duplicate/source leakage.
Provide inspect/prepare, smoke/tiny training, baseline and post-training evaluation, train/resume modes or subcommands with exact working Kaggle cell commands. Avoid all-test evaluation initially: frozen validation subset for baseline/model choice, official test only final. Report normalized WER/CER and substitutions/deletions/insertions, errors CSV and run metadata. Save processor/tokenizer, model, optimizer scheduler state, reproducible manifests. Checkpoints every bounded steps/save_total_limit, single rank side effects, explicit output persisted via Kaggle Save Version / separate checkpoint dataset; /kaggle/working not guaranteed after session, describe optimizer resume vs warm start.
Avoid precomputing entire dataset to RAM/disk mel features. No 5–10h runtime promise; guide measures warmed-up optimizer step times and validation overhead and computes max_steps budget, includes limited pilot caps. Explain two 16GB GPUs do not pool into 32GB single model VRAM; effective batch = per-device batch x 2 x grad accumulation. CLI max_steps, caps, workers etc configurable. Mention matching domain needed to reduce substitutions and no guaranteed gain.
Pin exact compatible HF packages and API assumptions, use eval_strategy or evaluation_strategy appropriate to pinned version, check generation behavior. Unit tests for metadata/path/exclusions/collator/normalization if feasible without downloading weights. Run existing mandatory python -m unittest discover -s tests -v, py_compile, and new light tests. No GPU/model execution available => report not run. Do not claim Kaggle runtime verified.
No skills filesystem access, no spawned tasks, no raw history needed. Scope source reads to named files above and existing tests as required; changes only additive training artifacts/tests and handoff progress. Return changed files, commands/check outcomes, known limitations and important guidance for controller final.

## Implementation progress (2026-10-13)
- Profile retained: openai/gpt-6.1-sol, general implementation; current implementer session ID was not supplied. Controller scout ID remains ses_eca9d7841ffeFjdKYQjgRL6PLj. No delegation, skills access, downloads, installs, commits or training.
- Added kaggle_training/train.py, requirements.txt, README.md (Vietnamese), test_smoke.py. Existing inference/benchmarks unchanged.
- RED: `python -m unittest kaggle_training.test_smoke -v` failed missing train module, wall 0.118s. GREEN final: same command, 4 tests pass, wall 0.128s; unittest execution 0.005s. Tests cover metadata happy path, conflicting duplicate, unsafe path and protected-ID/speaker split.
- `python -m py_compile kaggle_training/train.py kaggle_training/test_smoke.py` passed, final wall 0.074s. `python kaggle_training/train.py --help` passed, wall 0.062s.
- Mandatory `python -m unittest discover -s tests -v` attempted (handoff requires full suite): four import errors, missing gradio/soundfile/torch in system Python 3.14, wall 0.397s. No dependency changes made.
- GPU/model/DDP, pinned environment, collator tensor operations, actual Kaggle schema, resume and ASR metrics remain unverified. Smoke unit runner is standard unittest named module (not suite discovery). No conversion/parity or full-test ASR evaluation run.
- Known limits: exact hashes only within retained training and explicit benchmark exclusion manifests; no official-test content hash audit or near-duplicate audit. Audio validation happens lazily during use. Missing speaker IDs fail closed. Actual source pretraining overlap unknown; no improvement/runtime promises.

## Scoped review follow-up (2026-10-13)
- Authority: current user task plus controller's five-file scoped review. Provider/model openai/gpt-6.1-sol, general reviewer; reviewer ID supplied later by controller (not inferred from historical IDs). Scout remains explore ses_eca9d7841ffeFjdKYQjgRL6PLj; no delegation or skills.
- Fixed DDP checkpointing risk by setting gradient_checkpointing_kwargs={use_reentrant: False}; retained ddp_find_unused_parameters=False and use_cache=False. Added explicit empty-label rejection after per-example decoder-start stripping.
- Added dependency-free fake collator happy/critical-edge tests. RED named unittest module: 6 tests, empty-label cases IndexError/missing ValueError, wall 0.064s. GREEN same module: 6 pass, execution 0.003s, wall 0.124s. Runner inspected: standard unittest, named module limits execution.
- py_compile train.py/test_smoke.py passed, wall 0.113s; train.py --help passed, wall 0.091s. Existing runtime unchanged. Full existing suite not repeated: prior mandatory attempt recorded above failed on unavailable dependencies, no changed runtime inputs.
- README now explains additive ZIP/Dataset transfer without invented git URL, restart after installs, nvidia-smi plus torch GPU count, separate pilot prepare/baseline/manifests, baseline SHA pin for pilot, schema limitations, and unexecuted GPU/training/resume smoke.
- Static API assessment: eval_strategy, processing_class and gradient_checkpointing_kwargs are expected 4.46.3 interfaces; vi/transcribe/return_timestamps=False with forced_decoder_ids cleared is appropriate explicit Whisper generation configuration. Separate feature_extractor.pad/tokenizer.pad is the expected collator pattern; fake tests do not validate real tensors.
- BLOCKER to actual pinned-source/runtime verification: `python -c "import importlib.metadata as m; print(m.version('transformers'))"` fails PackageNotFoundError (Python 3.14 environment). No installs/downloads permitted. Controller must supply exact 4.46.3 relevant source/API evidence or run pinned environment verification; do not describe this as actual API execution verification.
- Resume passes checkpoint path to Trainer, while max_steps remains total; optimizer/RNG/DDP restoration unexecuted. Fairness is frozen speaker-disjoint validation and matching greedy normalization; official-test byte overlap/near duplicates/pretraining overlap remain unaudited. Processor loading/saving, actual Kaggle schema, GPU/DDP and ASR metrics remain unverified.
