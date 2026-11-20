# Training code

Both models were trained on Kaggle (`GPU T4 x2`, internet on). Seed 42 everywhere.

## Data

The organisers' package `train+public_test` v3.2.0 (built 2026-10-24). Its README forbids redistribution, so it is not re-uploaded. Add it as a **private** Kaggle dataset. Only `train/` is used for training. No public-test data and no synthetic or API-generated data.

`adahome/rationales.json` holds one short reason per gold step. It is derived from train labels only, with no model, so it is not in git (the data licence forbids redistribution). Regenerate it (the output is byte-identical to the one used for training):

Set `ROOT` at the top of `rationales/gen_rationales.py` to the folder holding `train/`, then:

```bash
python adahome/gen_rationales_short.py
```

It imports `rationales/gen_rationales.py`, so keep the two folders side by side.

## Base models (downloaded at training time, not in the image)

| Model | URL | Commit |
|---|---|---|
| Qwen3.5-4B | https://huggingface.co/Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Gipformer 1.5 | https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt | `dd9227dcd8705c13f33bdbe59728d546ab94480f` |
| icefall (zipformer code) | https://github.com/k2-fsa/icefall | `3f848bb6d0acc970c9b294a30ca0a04a7c9c78d1` |

## Steps

1. **ASR**: `notebooks/01c_asr_gipformer_ft.ipynb`, `FINAL = True`, `EPOCHS = 8`. It fine-tunes Gipformer on the train clips (clips with digits in the transcript are skipped) and exports int8 ONNX to `asr_model/` (= `saved_models/asr_model/`). It also writes `vi_itn.py` and `asr_base_train.json` (stock-model transcripts of train clips, used as ASR noise for the LLM).
2. **LLM**: `notebooks/02_llm_qwen_ver2.ipynb` with the step 1 output and `adahome/rationales.json` attached as inputs. Settings as shipped: `FINAL = False` (1,217 conversations; the 328 samples of the apps in `HOLDOUT_APPS` are held out for validation), `EPOCHS = 3`, `LR = 2e-4`, LoRA r 32, `GRAD_ACCUM = 8`, `MAX_LEN = 9000`, `NOISE_RATE = 0.5`, `HISTORY_CALLS = True`, `WHY = True`, `MASK_RATE = 0.3`, fp16. The run took two sessions (`RESUME = True` from the saved checkpoint for the second one). The 459-update log is `train_llm.log`. Output: `llm_model/` (merged) and `lifemate.py` (= `saved_models/`).

Each notebook runs with `SMOKE = True` first, then `SMOKE = False` and Save & Run All.
