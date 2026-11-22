# ZAC 2026 LifeMate — Competition Ruleset (binding)

Source of truth: `Description.docx` + `train+public_test/README.md` (data package v3.2.0, built 2026-10-24).
If this file and the official documents ever disagree, **the official documents win** — fix this file.
Every rule has an ID. Cite the ID when a change touches it (in code comments, commit messages and the
CLAUDE.md change log). `tools/check_rules.py` statically enforces the rules marked **[auto]**.

Severity: **MUST** = disqualification / zero score if broken. **SHOULD** = strong guidance, deviation
needs a written reason in the change log.

---

## 1. Data access & the execution environment

| ID | Rule |
|----|------|
| **R-ENV-1** [auto] | MUST NOT read, copy, open, glob, list or parse anything under `<split>_environment/` (incl. `state.json`) from solution code. All app data is reached only through `zaic_tools` tool calls. |
| **R-ENV-2** [auto] | Exactly **one** `ToolEnv` per sample, created **once**. Never re-create, `copy`/`deepcopy`/pickle, snapshot or reset an env to "try and redo". |
| **R-ENV-3** [auto] | Every call made on the env must appear in the submission: submit `env.get_tool_calls()` **unfiltered and unedited**, and `env.get_state_hash()` as `state_hash`. Never build `tool_calls` by hand. |
| **R-ENV-4** [auto] | Agent logic must not peek at app data outside tool calls: no `env.get_state()`, no private attributes (`env._…`), no reading the catalog's per-sample `_meta`. `env.get_state()` may only be used in offline debugging/eval scripts, never in the agent. Public API for the agent: `call`, `validate`, `get_tool_calls`, `get_state_hash`; the metadata attributes `allowed`, `sample_id`, `split` are tolerated (same information as the jsonl context), `env.states` and anything else are not. |
| **R-ENV-5** | Only call tools of the sample's `context.installed_apps` plus `sys.ask_missing_info`. `env.validate()` (no execution, not logged) is allowed and encouraged before `env.call()` to avoid rejected calls. |
| **R-ENV-6** | Use the split's own `zaic_tools` (`sys.path.insert(0, <split>)`). Each split has its own version; never vendor or patch `zaic_tools`. |

## 2. Model & system limits

| ID | Rule |
|----|------|
| **R-SIZE-1** | Whole system (ASR + LLM + any other model, incl. rerankers, classifiers, embedders, VAD) **< 6B parameters total**. |
| **R-SIZE-2** | The LLM **< 5B parameters** and context length **≤ 16,384 tokens** (configure `max_model_len`/`n_ctx` ≤ 16384; prompts + history + tool results must fit). |
| **R-SIZE-3** | Keep `MODELS.md` (model registry) up to date: name, exact param count, source URL, licence, **public release date**. Required before a model enters the final solution. |

## 3. Allowed data & resources

| ID | Rule |
|----|------|
| **R-RES-1** | Only public data and open-source models **released before 2026-10-26**, with licences that allow this use. |
| **R-RES-2** | External services/models (Gemini, NIM, OpenRouter, TTS, …) may be used **only during development** to generate synthetic data. They MUST NOT be used at inference. |
| **R-RES-3** [auto: warns] | MUST NOT label `public_test` or `private_test` — manually or with external models/services — **to train or tune** a solution. This covers transcripts, rationales, predicted tool calls, pseudo-labels, few-shot examples, prompt examples and any derived dataset. Training/tuning data comes from `train` (gold labels) and synthetic/public data only. |
| **R-RES-4** | Do not share or redistribute the competition dataset. |
| **R-RES-5** | Every training/tuning dataset file must have a recorded provenance (which split / generator / script) in the change log. |

## 4. Inference runtime

| ID | Rule |
|----|------|
| **R-RUN-1** [auto] | Fully **offline** at inference: no network, no external APIs/services, no downloads at runtime (all weights baked into the Docker image). |
| **R-RUN-2** | ≤ **30 s per sample** on 1× A30 (24 GB), 16-core Xeon Gold 6442Y, 64 GB RAM. Timeouts/crashes score 0 → enforce an internal per-sample time budget with a safe fallback that still returns `env.get_tool_calls()` + hash. |
| **R-RUN-3** | One sample failing must never stop the run: wrap each sample, always write exactly one line per sample. |

## 5. Reproducibility

| ID | Rule |
|----|------|
| **R-REP-1** [auto] | Fix all seeds (Python `random`, NumPy, PyTorch incl. CUDA, any other RNG). |
| **R-REP-2** [auto] | Deterministic decoding for ASR **and** LLM: greedy / `temperature=0`, `do_sample=False`, beam settings fixed. No sampling. |
| **R-REP-3** | Same input ⇒ same submission on re-run. Non-determinism ⇒ organisers average 5 runs. |

## 6. Submission format

| ID | Rule |
|----|------|
| **R-SUB-1** | UTF-8 JSON Lines, **exactly one line per sample** of the split, keys `id`, `tool_calls`, `state_hash`, written with `ensure_ascii=False`. |
| **R-SUB-2** | Out-of-scope ⇒ `tool_calls: []`. Missing info ⇒ `sys.ask_missing_info` with `{"tool_name": ..., "params": [...]}` (may follow other calls). |
| **R-SUB-3** | Validate before every upload: `python -m zaic_tools --split public_test --root .. --validate submission.jsonl` (run inside `public_test/`). |
| **R-SUB-4** | Public round: max **5 submissions/day**; ends **2026-11-16**. |

## 7. Docker / final solution

| ID | Rule |
|----|------|
| **R-DOCK-1** | Test data mounted read-only-style at **`/data`** (same layout as `public_test`); output at **`result/submission.jsonl`**. Inference script runs with no network. |
| **R-DOCK-2** | Package must include training code, inference code and data; deliver as a public Google Drive link + checksum via the "Solution Submission" form by **2026-11-16**. Max 2 submissions, the **last** one is evaluated. |
| **R-DOCK-3** | After freezing, private_test is run with the frozen image and the provided `zaic_tools`; organisers re-run the image to verify. No changes after freeze. |

## 8. Scoring (for reference — optimise for this)

- Per sample: gold `[]` ⇒ 1 iff submitted `[]`, else 0.
- Otherwise: score 0 if `state_hash ≠ gold` **or** M = 0; else F1 with P = M/(M+U), R = M/|G|.
  - Match = same tool name, all non-fuzzy args identical, every fuzzy (`x-match: fuzzy`) arg token-F1 > 0.6 (lowercase + NFC). One-to-one, order ignored.
  - U counts **every unmatched submitted call, including schema-rejected ones** ⇒ extra read calls, rejected calls and unrequested optional args all cost score.
- Final = 100/N · Σ scores. Tie-break: F1 ignoring hash, then lower average time per sample.

---

## 9. Project-specific interpretations (flagged — confirm with organisers if it matters)

- **F-1 API pipelines** (`gemini api/`, `gemi + nemo/`, `api_agent/`, `colab/`) use external APIs at
  inference. They can never be part of the Docker/final solution (R-RUN-1, R-RES-2). The "Every solution must
  meet all requirements" wording means public-LB results produced by them may also be considered non-compliant —
  treat them as dev/reference only.
- **F-2 Public-test artefacts** produced by external models (`*public_test*transcripts*.json`,
  `public_test_rationales.json`, `cache_public_*`, public `submission*.jsonl`) MUST NOT feed any training,
  fine-tuning, prompt/few-shot selection, threshold tuning or distillation (R-RES-3).
- **F-3 Tuning on public LB score** (choosing between pipelines by their LB number) is normal LB usage, not
  labelling — allowed, within the 5/day quota.
- **F-4 ASR training data**: train audio + `audio_text` (gold) is allowed. Clips from public_test may not be
  pseudo-labelled for ASR training.
