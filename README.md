# LifeMate — Zalo AI Challenge 2026

Offline voice agent for the LifeMate track: a spoken Vietnamese request, the
user's context and the JSON schemas of the installed apps go in; an ordered
list of tool calls on one `ToolEnv` per sample comes out. Everything runs in
one Docker image on one GPU, no API. Weights live outside git (see
`saved_models/README.md`).

| Part | Model | Params | Licence | Runs on |
|---|---|---|---|---|
| ASR | `g-group-ai-lab/gipformer1.5-68M-rnnt`, fine-tuned on the train clips | 68.6M | MIT | CPU, int8 ONNX, `sherpa-onnx` 2.0.1 |
| LLM | `Qwen/Qwen3.5-4B`, LoRA r 32, merged, text only | 4.21B | Apache-2.0 | GPU, vLLM 0.32.0, fp16 |
| Total | | 4.27B | | under the 6B limit |

Public leaderboard: **60.29** (held-out train split, gold text: 71.77).

## Pipeline

**ASR.** The final clip and every user turn of the history are transcribed
one clip at a time (greedy search, 4 threads, fixed so re-runs match), then
lowercased and put through `vi_itn.py`, a rule-based inverse text normaliser:
spoken money and long numbers become digits (`hai triệu tám` → `2800000`),
times stay in words, digit runs join (phone numbers). Clips whose transcript holds digits were skipped in
fine-tuning, so the model keeps numbers spoken and ITN converts them.

**LLM.** The prompt (`lifemate.py`, shipped with the weights it was trained
with) holds the rules, profile, current time, a 14-day date table labelled
`tuần này` / `tuần sau`, the installed tools' compact schemas, the history and
the request. Each turn the model writes one JSON line, a short reason then
`{"name", "arguments"}`, or `{"done": true}`. Tool results come back as a
`KẾT QUẢ:` turn. Greedy, seed 42, context limit 16,384 (prompts are under 9K).

**Loop** (`predict.py`):
- `env.validate` (not logged) before every `env.call`, so invalid calls cost
  nothing. A call missing a required parameter becomes
  `sys.ask_missing_info` for it; any other invalid call gets one retry with
  the error shown.
- After a search with a `query` that returns nothing, the tool is called once
  without filters so the model can pick from the full list.
- Stop on `done`, after `sys.ask_missing_info`, on the same call a third
  time, before a third call of one read tool, after 8 steps, and no new step
  20 s after the sample started (ASR included). The calls made are always
  written.

## Training

Kaggle, `GPU T4 x2`, seed 42. Only the organisers' train split (1,545
samples), no public-test data, no API or teacher model. Details, base-model
commits and settings: `training_code/README.md`.

1. `training_code/notebooks/01c_asr_gipformer_ft.ipynb` — icefall fine-tune
   (ScaledAdam 0.0045, SpecAugment-style masks, 8 epochs, best epoch by
   held-out WER), int8 ONNX export to `asr_model/`.
2. `training_code/notebooks/02_llm_qwen_ver2.ipynb` — LoRA on 1,217
   conversations (7 apps held out for validation), 3 epochs, 459 updates.
   - Targets are the gold calls of the final turn **and** of the history.
   - Each target starts with a short template reason
     (`training_code/adahome/gen_rationales_short.py`, no model): the ID and
     the result it came from, or the missing parameter.
   - In 30 % of conversations every app gets a made-up name, so the model has
     to read schemas: 65 % of public samples install an app train never has.
   - Half the conversations see stock Gipformer transcripts instead of clean
     text.

T4 notes, since none of this is in the Qwen3.5 docs: `head_dim` 256 makes
`transformers` take the grouped-key SDPA route, which a T4 runs in the
float32 math kernel and runs out of memory, so keys are repeated and the math
kernel is off; the linear-attention reference kernel runs in checkpointed
512-token parts; fused AdamW does not skip an overflowed fp16 step when both
GPUs overflow, so `adamw_torch` is used.

## Files

| Path | What |
|---|---|
| `Dockerfile` | `vllm/vllm-openai:v0.32.0-cu129` + `sherpa-onnx`; uninstalls the base image's CUDA-13 `torchcodec`, which breaks `from vllm import LLM` |
| `predict.sh`, `predict.py` | `/data` → `/result/submission.jsonl` (+ `details.jsonl`: stop reason, raw output, transcripts, time) |
| `predict_notebook.ipynb`, `start_jupyter.sh` | Timing notebook, Jupyter on port 9777 |
| `test_agent.py` | No-GPU check: scripted model and a gold-replaying oracle on real envs, plus the organisers' validator |
| `training_code/` | The two training notebooks, the reason generator, the LLM training log |
| `eval/03c_eval_repair.ipynb` | Kaggle eval: held-out scores on gold and ASR text, public submission |
| `eval/04_private_submit.ipynb` | Same loop on `private_test`, with the Docker's ASR code |

## Run

The folder holding `public_test/` (or `private_test/`) is mounted at `/data`.

```bash
docker build -t zac2026-zola:v1 .
docker run --rm --gpus "device=0" --network none \
  -v /path/to/data:/data:ro -v /path/to/result:/result \
  zac2026-zola:v1 /bin/bash /code/predict.sh
```

Without a GPU, inside the image: `python3 test_agent.py /data /code/saved_models public_test`.

## Data

The organisers' package is not redistributable and is not in this repo, nor
is anything derived from it (`rationales.json` is regenerated by the script
above).

## Licences

- Qwen3.5-4B weights: Apache-2.0.
- Gipformer 1.5 weights: MIT (per the model card). icefall: Apache-2.0.
