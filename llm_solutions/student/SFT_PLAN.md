# SFT plan across the three Kaggle notebooks (01c ASR → 02 LLM → 03 eval)

Status (2026-11-14, second pass): **A1, A2, B1, B2, B4 (RESUME, MODELS.md), C2, C3 implemented** in new copies
`student/notebooks/{thbased01,bowen,neweval}_sft.ipynb`, built by `student/make_sft_notebooks.py` from the originals
(which stay untouched). Not implemented: week labels, fla/causal-conv1d kernels, DeepSeek reasons (B3), C1/C4 need no
code (settings comments only). Nothing has run on Kaggle yet; local checks are in the CLAUDE.md change log.

Every "verified" fact below was checked on 2026-11-14 against the
executed notebooks in `D:\Study\nlp\jigsaw dataset\` (`thbased01.ipynb`, `bowen.ipynb`, `neweval.ipynb`), the files
next to them, and `train+public_test/train`. Anything marked *estimate* or *to verify* was not measured.

"The SFT technique" here = the student recipe already in `student/`: supervised fine-tuning on **train gold
trajectories only**, with realistic ASR noise on the input side, short written reasons before each action (`WHY`),
app-name masking, and scoring on lifemate's held-out train rows (218 unseen-app + 110 seen-app). Each notebook gets
the part of that recipe it owns:

| Notebook | Role in SFT | What changes |
|---|---|---|
| 01 `thbased01` (01c) | ASR SFT + the *noisy inputs* for LLM SFT | digit clips into ASR training; cross-fitted train transcripts |
| 02 `bowen` | LLM SFT | noise source, reasons source, a train/inference mismatch, run hygiene |
| 03 `neweval` | measures the SFT model | uncontaminated held-out eval, offline-only run, latency |

**Deadline**: public round and Docker both end **2026-11-16** (R-SUB-4, R-DOCK-2). One full 02 run takes ~7.8 h on
2×T4 (verified: `train_runtime` 2.818e4 s). So at most one validation run and one final run fit. Section 5 orders the
work for that.

---

## 1. Verified starting point

### 01 `thbased01.ipynb` (Gipformer 1.5 fine-tune)
- Run shown is `FINAL = True`, `EPOCHS = 8`, `SMOKE = False`: 4,555 training clips (6.1 h), **0 held-out clips**,
  so no per-epoch WER was printed. Loss/frame 0.245 → 0.084; 71–73 s per epoch after the first (308 s, feature
  extraction); peak 5.8 GB on one T4.
- 481 of 5,036 train clips are skipped for having digits in gold text (verified: 481 clips with `\d`). The digit
  tokens are almost all plain integers (604 of 607 tokens; 3 are 7+ digit strings), plus codes such as `VTV6`,
  `SCTV9`, `RON95`.
- `asr_transcripts.json` (fine-tuned) has train WER 0.0254 on the 1,089 held-out clips **because FINAL heard them**;
  `asr_base_train.json` (stock model, what 02 trains on) has WER 0.1644 / 0.1609 after ITN.
- `vi_itn.py` in 01 and in 03 are byte-identical (verified).

### 02 `bowen.ipynb` (Qwen3.5-4B LoRA)
- Settings in the saved notebook: `FINAL = False`, `WHY = True`, `MASK_RATE = 0.3`, `HISTORY_CALLS = True`,
  `NOISE_RATE = 0.5`, `MAX_LEN = 9000`, **`RESUME = True`** (the log starts at update 250, so this run continued from
  a checkpoint). 459 updates, final train loss 0.146, merged model 4,205,751,296 params.
- `rationales.json` next to the notebooks = output of `gen_rationales.py` (rule-based from train gold, reads only
  `train/train.jsonl` and the train catalog). Verified: 1,545/1,545 train ids, every list has history calls + final
  calls + 1 entries, no non-train key; 8,747 reasons, Qwen3.5 tokens mean 35, p90 53, max 119. `rationales (1).json`
  is an older, shorter version (mean 20 tokens).
- Largest labelled action with its reason: 210 tokens (p99 174), under 03's `MAX_NEW_TOKENS = 320`. Reasons add
  ~115 generated tokens per sample on the final turn (mean; max 339).
- **Mismatch found** (code read in both notebooks): with `WHY` and `HISTORY_CALLS`, training rewrites the *history*
  tool calls as `{"why": ..., "name": ...}` (they are labelled targets). At inference, `build_messages` renders
  history calls without `why`. So every final-turn target was learnt after a history that never looks like test
  history. Samples with history: 419/634 on public (from the project memory).
- `lifemate.py` in bowen differs from `student/lifemate.py` only by the week labels in the 14-day table
  (`student/` adds "tuần này / tuần sau"). Prompt change ⇒ needs a fresh training run, not a resume.
- MODELS.md has no Qwen3.5-4B row and Gipformer is all TODO (R-SIZE-3).

### 03 `neweval.ipynb`
- LLM attached is `nquoong/qwenupdatefabe/llm_model`, **not** `png0806/bowen`. Which training config produced it is
  not recorded anywhere here (R-RES-5 gap).
- ASR text: 645 public final-request clips from `raw_asr_public.json` (Gemini, per the notebook's own text) + the
  other 1,288 clips from 01c FINAL's `asr_transcripts.json`. `EVAL = False`, so no held-out score.
- Public run: stops done 562, ask 66, repeat 4, max_steps 1, repaired 1; 0 invalid; **92/634 samples over 30 s** on
  T4 (slowest 56.7 s); validator 0 schema errors. No LB score in the notebook.
- With `FILL_FROM_01C` and 01c FINAL attached, an `EVAL = True` run would score held-out rows on transcripts made by
  a model that trained on those clips (the notebook itself labels its 66.06 "inflated").

### Data hazards next to the notebooks
- `public_test_rationales.json` (634 public ids) sits in the same folder. It must never be attached to 02 or any
  training run (R-RES-3, F-2). 02's glob looks for the exact basename `rationales.json`, so it is safe only as long
  as nobody renames it.
- The DeepSeek `--why` train run (for better reasons) has **not** produced traces yet: the latest
  `deepseek-api-zola (1).log` is a public run failing with `unknown provider for model deepseek-v4.1-flash-1017`.

---

## 2. Notebook 01 — ASR SFT and noisy inputs for the LLM

### A1. Put the 481 digit clips into ASR training (number verbaliser)
- Add `vi_verbalise.py`: integer → Vietnamese words in the model's style (lowercase, old tone spelling), e.g.
  `25 → hai mươi lăm`, `500 → năm trăm`, `2000000 → hai triệu`; letter+digit codes spelled per character only if a
  rule exists for them, else the clip stays skipped.
- Keep a clip only if the round trip holds: `itn(norm(verbalise(text))) == itn(norm(text))`. Clips that fail stay
  skipped exactly as now (the existing `re.search(r"\d", text)` guard becomes "digits that did not round-trip").
- Self-check in `__main__` (as `vi_itn.py` has): a table of cases + a count of how many of the 481 clips pass.
- Risk: speakers say short forms (`hai lăm`, `trăm hai`) the verbaliser will not choose → label noise on exactly
  those clips. Check: WER on the digit clips of the held-out set, before vs after.
- Why it matters: numbers are arguments (amounts, times, channels); today the ASR never trained on them.

### A2. Cross-fitted train transcripts `asr_xfit_train.json` (the main SFT-data change)
- Today 02 learns from the **stock** model's errors (WER 0.16), while test input comes from the **fine-tuned** model
  (memory: 11.7% on unseen digit clips). Different error profile from test.
- K = 5 folds over train **rows** (all clips of a row in one fold, so history and final clip share a fold). For each
  fold: fine-tune on the other 4 folds for the best epoch count, export int8, transcribe that fold with sherpa-onnx.
  Concatenate → every train clip transcribed by a model that never heard it. Same key format as
  `asr_base_train.json` (audio path, ITN applied).
- Cost *estimate*: ~57 s/epoch on 4/5 of the data × 8 epochs + export + CPU transcription of ~1,000 clips ≈ 15 min
  per fold, ~75 min total on one T4. Fits one Kaggle session with 01's other steps.
- Implementation: wrap the existing `train_asr.py` with a `FOLD` setting (rows with `i % 5 == FOLD` held out) and
  loop it in a cell; `asr_transcribe.py` gets a `xfit` mode. No new dependencies.

### A3. Run order and outputs
1. `FINAL = False` run with A1 → per-epoch held-out WER, best epoch (also the eval output for 03).
2. Same session or next: A2 with that epoch count → `asr_xfit_train.json`.
3. `FINAL = True` with A1 → `asr_model/` for the Docker and `asr_transcripts.json` for public.

### Verify (01)
- Held-out WER on 1,089 clips (FINAL=False): with A1 vs the old run; digit-clip subset separately; after ITN both
  sides (the notebook already prints this).
- Per-fold xfit WER ≈ FINAL=False held-out WER (within ~1 point); a much lower number means a fold leak.
- `asr_xfit_train.json` covers all 5,036 train clips; keys equal `asr_base_train.json`'s.
- Rules: F-4 (train audio + gold only), R-REP-1/2 (greedy, seeds), R-SIZE-1 (68.6M params), R-RES-5 (log provenance).

---

## 3. Notebook 02 — LLM SFT

### B1. Noise source
- `NOISY_ASR`: take `asr_xfit_train.json` when attached, else `asr_base_train.json` (current). One-line glob change.
- Keep `NOISE_RATE = 0.5`. Optional third bucket (stock text for some rows) only if B1 alone loses on 03.

### B2. Fix the history-`why` mismatch
- Proposed: add the reason only to the **final-turn** targets; history calls stay labelled (`HISTORY_CALLS`) but are
  rendered without `why`, exactly as inference renders them. In `conversation()` that is: zip the reasons with
  `range(first, len(msgs))` targets only and skip the history entries of `whys[id]` (they are the first
  `len(labelled) - (final calls + 1)` items).
- Side effect: the model sees some targets with reasons and some without. `parse_action` accepts both, so this cannot
  break parsing; the eval stop counts will show if reasons disappear on the final turn.
- This is an A/B against the current run (bowen as saved) on 03 EVAL — the only measurement that decides it.

### B3. Reasons source
- Now: `rationales.json` (template, train-only, verified above). Keep it for the next run.
- Later (only if the DeepSeek train run finishes): `student/why_from_traces.py --traces <traces> --base
  rationales.json --out rationales_ds.json`, then attach as `rationales.json` (the glob needs that exact name).
  Blocker first: the gateway model name in the failed log (`deepseek-v4.1-flash-1017`) is rejected.
- Keep reasons short: `MAX_CHARS = 300` in the converter; check the p99 action+reason tokens stay < 320
  (`MAX_NEW_TOKENS` in 03).

### B4. Run hygiene
- `RESUME = False` for any run whose data or prompt changed (the saved notebook has `True`; resuming a checkpoint
  trained on other text silently mixes the two).
- Week labels from `student/lifemate.py`: optional, only together with B2 in a fresh run, never in a resume.
- Speed (*to verify*): the log shows the reference kernels for `chunk_gated_delta_rule` and `causal_conv1d`. Try
  `flash-linear-attention` + `causal-conv1d` in the smoke run; keep only if the smoke loss matches and seconds/step
  drop. Same packages then go into the Docker (inference speed, R-RUN-2). Triton on T4 (sm75) may not support it.
- Log provenance of every attached file (transcripts, rationales) in CLAUDE.md (R-RES-5); add Qwen3.5-4B to
  MODELS.md (R-SIZE-3).

### Verify (02)
- `python lifemate.py <root>` self-check passes (already a cell).
- Smoke run (`SMOKE = True`): finite loss, peak GB per GPU ≤ the current 9.5 / 6.6.
- New assert in `conversation()` for B2: no message before `first` contains `"why"`; every final-turn target does.
- Full run (`FINAL = False`): loss curve comparable to 0.146; then 03 EVAL (section 4).

---

## 4. Notebook 03 — measuring the SFT model

### C1. Attach the model under test
- Point at the 02 output being measured (bowen or its successor), and write its version name into the result line.
  The saved run tested `qwenupdatefabe`, whose config is unknown.

### C2. Uncontaminated held-out eval
- `EVAL = True` with 01's **FINAL=False** `asr_transcripts.json` (or `asr_xfit_train.json` mapped to `train/<path>`
  keys), never a FINAL=True one. The cell already asserts all 1,089 held-out clips are present.
- `GOLD = True` once per new model (gold-text upper bound). Report unseen / seen / per stop reason (already printed).

### C3. Two public runs, only one of them counts for the final solution
- (i) Offline: 01c FINAL transcripts for all 1,933 clips (no `raw_asr*.json`) — this is what the Docker can score
  (R-RUN-1, R-RES-2).
- (ii) Gemini final requests — ASR-upper-bound measurement only, never in the Docker, never used to train or tune
  (R-RES-3, F-1). The notebook's current setup is (ii).
- `raw_asr*.json` is required by the settings cell today (`assert RAW_JSON`); run (i) needs that assert relaxed to
  allow zero raw files when `FILL_FROM_01C` is on.

### C4. Latency
- T4 run: 92/634 over 30 s. Run once with `TIME_BUDGET = 25` to see the score cost of a hard budget (R-RUN-2 says
  timeouts score 0). Compare WHY vs no-WHY generated tokens per sample (reasons ≈ +115 tokens mean).
- A30 numbers are not measured; the Docker test must time it there.

### Verify (03)
- Validator: 634 lines, 0 schema errors (already a cell).
- Held-out score table: bowen-as-saved vs the new run, gold and asr, same transcripts file for both.
- Baselines from the notebook: bowen + 01c — public 60.29, held-out gold 71.77, held-out asr 66.06 (inflated).

---

## 5. Order of work under the 2026-11-16 deadline

| Step | Notebook | GPU time (estimate) | Gate to continue |
|---|---|---|---|
| 1 | 01 FINAL=False + A1, then A2 | ~2 h (1×T4) | held-out WER not worse than before; xfit WER near held-out WER |
| 2 | 02 smoke with B1 + B2, `RESUME = False` | ~20 min | finite loss, memory as before, B2 assert |
| 3 | 02 full `FINAL = False` | ~8 h | — |
| 4 | 03 EVAL (gold + asr) on step 3 vs bowen-as-saved | ~1.5 h | new ≥ old on held-out asr; else keep bowen's config |
| 5 | 01 FINAL=True + A1 (Docker ASR) | ~20 min | can run in parallel with step 3 |
| 6 | 02 `FINAL = True` with the winning config | ~9 h | — |
| 7 | 03 offline public run (C3 i), validator, submit | ~2 h | ≤ 5 submissions/day |

Steps 3 and 6 do not both fit before 2026-11-16 in sequence. If only one LLM run fits: skip step 3–4, run step 6
with B1 only (lowest risk: same text format, better-matched noise) and leave B2 for after the round. B2 changes what
the model learns and should not ship unmeasured.

## 6. Out of scope here
- DeepSeek reasons (B3 later) until the gateway run works.
- Any use of public transcripts, public rationales or public traces for training (R-RES-3).
- Docker packaging and A30 timing (separate task).
