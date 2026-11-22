# LLM solutions

LLM-side development work for LifeMate, kept apart from the submitted solution in the repo root. **None of this
runs in the Docker image**: the API agents call external models, which the rules allow only during development
(`RULES.md` R-RES-2, F-1); the 27B model is over the 5B LLM limit (R-SIZE-2); the student notebooks are
experimental versions of the team's training and eval notebooks.

## Results

Public leaderboard (API or reference runs, not eligible as a final solution):

| Run | LLM | Final-request text | Public LB |
|---|---|---|---|
| `api_agent/solve.py` | Nemotron 3 Ultra 550B-A55B (NVIDIA NIM) | Gemini | 68.42 |
| same agent, Colab run (`colab/zola_api_pipeline.ipynb`) | Nemotron 3 Ultra | Gemini + fine-tuned Gipformer | **76.08** |
| `qwen27b_reference/colab_qwen38_27b.ipynb` (v1) | Qwen3.8-27B FP8, vLLM, zero-shot | Gemini | 64.55 |
| `qwen27b_reference/colab_qwen38_27b_v2.ipynb` | same | Gemini | 70.88 |

On train (gold labels, official scoring formula):

| Check | Score |
|---|---|
| `solve.py`, Nemotron, gold text vs two transcripts (same 96 rows) | 98.44 vs 92.33: transcripts cost about 6 points |
| `solve.py`, Nemotron, rows of 7 apps kept out of examples and habits (gold text, 92 rows) | 81.14 (about 85 on seen apps) |
| `solve.py --why`, DeepSeek V4.1 Flash, all train rows (1541 of 1545 finished) | 83.06 |

Not recorded: the v3 27B notebook on the full public set (a local llama.cpp run reached 480 of 634 samples), the
DeepSeek and K2 public runs, and any score of the student notebooks.

## What is where

| Path | What |
|---|---|
| `api_agent/solve.py` | The API agent, one file (called `gemi + inkling/` during development). The builders and notebooks use this path, so a data bundle must be rebuilt from this folder before a Colab run. |
| `colab/` | Colab notebooks for the agent: `zola_api_pipeline.ipynb` (Nemotron, ASR on a T4 or CPU, resumable on Drive), `zola_deepseek.ipynb` (DeepSeek with a USD budget stop); builders `make_notebook.py`, `make_deepseek.py`, `make_bundle.py`. |
| `kaggle/` | `kaggle_nemotron_train.ipynb` (agent over all 1545 train rows in 10 folds), `kaggle_deepseek.ipynb`, `kaggle_k2.ipynb` (IFM K2 Horizon, rotates keys, resumes after the daily token cap); builders `make_kaggle.py`, `make_k2.py`. |
| `qwen27b_reference/` | Qwen3.8-27B runs v1 to v3 (Colab A100, vLLM) and `local_qwen38_27b_v3.ipynb`, the port to a rented RTX card (llama.cpp, quantized GGUF). |
| `student/` | Work toward the ≤4B model: SFT data builder, held-out eval, reason converter, SFT and RL versions of the team's notebooks, a rented-GPU eval script. |
| `tools/check_rules.py`, `RULES.md` | The competition rules with IDs, and a static checker for them. |

## 1. API agent (`api_agent/solve.py`)

One sample per conversation, native tool calling in a loop on one `ToolEnv` per sample:

- `env.validate` before every `env.call`; an invalid call goes back to the model with the error and is never logged.
- The 5 nearest train-gold examples in the prompt, picked by request similarity.
- **Argument habits from train gold**: optional arguments gold fills in ≤ 5 % of calls are dropped from reads and
  sent back once on writes; arguments gold fills in ≥ 95 % are asked for once; per-tool hints in the prompt.
- A read tool runs at most 3 times per sample.
- Two transcripts of the final request (Gemini and Gipformer) with a note on typical ASR errors.
- Backends: OpenRouter, NIM, DeepSeek (paid, budget stop), a gateway that speaks the Anthropic Messages format, IFM K2
  (key rotation). Resumable cache, one thought trace per sample, empty API replies re-asked instead of cached, unsolved
  samples filled from a base submission. `--self-check` runs offline asserts.

What the runs taught:

- A Colab run scored 14.46 because empty HTTP-200 replies were cached as "no action" (532 of 633 samples). After the
  fix the same agent scored 76.08.
- The remaining loss against train gold is mostly conventions on optional arguments: `alarm.create` without a label,
  `bill_payment.list_bills` and `calendar.list_events` without filters, `banking.transfer` without a note,
  `calendar.create_event` ending one hour after the start. That is what the habits and hints target.

## 2. Qwen3.8-27B reference (`qwen27b_reference/`)

- **v1** (64.55): 33 turns ended inside the thought, so the plan text became the answer; asks in 19 % of samples
  against 14 % in train gold.
- **v2** (70.88): unfinished thoughts are closed and answered, generic extra rules, medium effort, and a train A/B
  mode that compares settings in one session with a sign test.
- **v3**: 5 train-gold examples per sample (TF-IDF, one per likely tool), per-tool argument habits, a second transcript
  when the two disagree, and the train check on the same loaded model.
- **local v3**: same agent on `llama-server` with a GGUF quantization chosen from GPU memory, parallel slots, Google
  Drive sync of checkpoints, Hugging Face token from the environment.

## 3. Student, ≤ 4B (`student/`)

- `build_sft.py`: SFT data from train gold, clean and stock-Gipformer text (1217 train, 218 unseen-app and 110 seen-app
  rows). Every target is checked to parse back to its gold call.
- `eval_student.py`: held-out eval with the same agent loop as the notebooks (vLLM, transformers or stub backend).
- `why_from_traces.py`: turns DeepSeek `--why` train traces into the per-call reasons file the LLM notebook reads.
  It keeps a reason only when the row scored 1.0 with one call per step in gold order (913 rows).
- `make_sft_notebooks.py` → `notebooks/{thbased01,bowen,neweval}_sft.ipynb`: SFT versions of the team's ASR, LLM and
  eval notebooks (plan and checks in `SFT_PLAN.md`). Digits spelled out so digit clips can train the ASR, 5-fold
  cross-fitted train transcripts as the LLM's noisy input, reasons only on final-turn targets, an eval that refuses
  transcripts from a model that heard the held-out clips, an offline public run.
- `make_rl_notebook.py` → `notebooks/qwen_k2_rl.ipynb` and `neweval_rl.ipynb`: one round of reward-filtered imitation.
  The old checkpoint acts on train rows with the official score as reward, K2 writes reasons, failed rows count twice,
  a fresh LoRA trains on top, and a gate compares old and new on 66 held-out rows neither saw.
- `eval_vast.py` (and `eval_vast_onefile.py`, built by `--pack`): the eval notebook as one script for a rented 1-GPU
  Linux box. It downloads its inputs from Kaggle.
- `lifemate.py`: a copy of the team's prompt/scorer module at the version these scripts were written against. Not my
  code; it is here because the scripts import it.

The submitted model (root README) was trained with template reasons and no API data, so it does not use the DeepSeek
or K2 parts of this work.

## Rules

- External APIs only during development (F-1, R-RES-2). Nothing here is part of the offline Docker run.
- Prompts, examples, habits and training data come from **train gold only** (R-RES-3). No public-test transcript,
  trace or submission is used for training or prompts, and none is in this folder.
- No competition data or anything derived from it is in git: no transcripts, caches, traces, submissions, bundles or
  `rationales.json` (R-RES-4, same as the root README). The builders write data bundles locally; upload them only as
  private Kaggle datasets or private Drive files.
- Keys come from environment variables or Colab/Kaggle secrets (`NVIDIA_API_KEY`, `OPENROUTER_API_KEY`,
  `DEEPSEEK_API_KEY`, `GWAI_API_KEY`, `IFM_API_KEY`), or a local `api_keys.txt` that git ignores.
- `python tools/check_rules.py <path>` checks code against the rules; add `--final` for code that goes into the Docker image.

## Running

The scripts expect the organisers' package at `llm_solutions/train+public_test/` (git ignores it). The folders
must stay side by side: the builders and notebooks refer to each other by relative path.

```bash
cd llm_solutions
python "api_agent/solve.py" --self-check
NVIDIA_API_KEY=... python "api_agent/solve.py" --split train --backend nim \
  --model nvidia/nemotron-3-ultra-550b-a55b --sample 8 --workers 4
```

Public runs also need transcript files from the ASR step (`--transcripts`), which are not in this folder.
`student/build_sft.py` and `eval_student.py --texts asr` read `asr_zipformer/asr_base_train.json`, written by the
team's ASR notebook.
