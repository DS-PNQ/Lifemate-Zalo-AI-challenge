"""Builds kaggle/kaggle_k2.ipynb: the API pipeline (api_agent/solve.py) on IFM K2 Horizon over the public test set.
Run: py kaggle/make_k2.py  (also rebuilds colab/zola_deepseek_bundle.zip, which holds the updated solve.py).
Cells are the DeepSeek Kaggle notebook's (colab/make_deepseek.py) with the backend, secret and file names swapped.
Dev/reference only (F-1): external API at inference. Train gold is the only source of examples and habits (R-RES-3)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colab"))
import make_deepseek as D  # noqa: E402


def sub(src, *pairs):
    for old, new in pairs:
        assert old in src, f"make_deepseek.py changed, not found: {old!r}"
        src = src.replace(old, new)
    return src


MD = """# ZAC 2026 LifeMate — IFM K2 Horizon capability test (Kaggle, CPU)

Runs the same agent as the Nemotron / DeepSeek pipelines (`api_agent/solve.py`) with **K2 Horizon**
(`IFM/K2-Horizon-375B-A23B`, api.ifm.ai, OpenAI format, reasoning always on at "high") on the **634 public-test samples**,
and writes the submission. Every sample gets its own `ToolEnv`; the final request is shown as two transcripts
(Gemini + Gipformer). Train gold is used *inside the prompt* (5 most similar train rows with their gold calls, argument
habits); no public-test text is used as an example (R-RES-3). Reference only: an API model can never be the final
solution (F-1).

**Limit.** IFM gives **10M tokens per key per day** (prompt + completion, reset 00:00 UTC; replayed reasoning counts as
prompt). Estimate ~25–35k tokens per sample → the 634 samples need **~20M tokens, i.e. two keys or two days**. With
**several keys** in the secret, the run moves to the next key as soon as one hits its cap (or is rejected) and stops only
when the last one does; everything is saved, resume later (below). No prices are published.

**One-time setup**
1. Run `py kaggle/make_k2.py` on your PC and upload `colab/zola_deepseek_bundle.zip` as a **private** Kaggle dataset
   (competition data, R-RES-4; re-upload it if you had an older one: cell 1 refuses a solve.py without key rotation).
   *Add Input* → attach it.
2. *Add-ons → Secrets* → add `IFM_API_KEY` and attach it to this notebook. Several keys go into this one secret,
   separated by commas or new lines; they are used in that order.
3. *Settings → Internet on*. *Accelerator: None.*

**How to run**
- Quick test in the editor: `LIMIT = 3` in cell 1, run cells 1–3 (separate files).
- Full run: `LIMIT = None`, then *Save Version → Save & Run All (Commit)*.
- **Resume** (daily cap, API errors, killed session): *Add Input → Notebook Outputs* → this notebook's last version, then
  *Save & Run All* again. Cell 1 copies the checkpoints; finished samples are never sent again.

**Outputs** (`/kaggle/working`): `submission.json` (= `submission_k2.jsonl`, JSON Lines as the competition requires),
`traces_public_k2.jsonl` (per sample: every step's thought, text, tool calls with status and result),
`cache_public_k2.jsonl`, `run_k2.log`. Samples K2 has not solved yet are taken from the 68.42 Nemotron submission
(`FILL_FROM_BASE`), so a partial run is still a valid file — cell 3 says how many; the LB score measures K2 only once
that number is 0. Public traces are reference only: never use them for training or prompts (F-2)."""

SETTINGS = r'''# 1. Settings, files, API key
import getpass, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
BACKEND = "ifm"       # IFM K2 Horizon (IFM/K2-Horizon-375B-A23B)
WORKERS = 4           # samples at the same time (IFM's per-minute limits are unpublished; lower it on many 429s)
LIMIT = None          # e.g. 3 for a quick test (separate files); None = all 634 public samples
FILL_FROM_BASE = True # samples not solved yet (daily cap, errors) are taken from the 68.42 Nemotron submission
TRAIN_ROWS = 0        # optional train check (cell 4) on this many rows with gold text; uses the same daily token cap

'''

KAGGLE_FILES = sub(D.KAGGLE_FILES,
                   ('"*deepseek*.jsonl"', '"*k2*.jsonl"'),
                   ("DEEPSEEK_API_KEY", "IFM_API_KEY"),
                   ('if not os.environ.get("IFM_API_KEY"):',
                    'if not os.environ.get("IFM_API_KEY"):  # one secret; several keys separated by commas or new lines'),
                   ('{"gwai": "https://1gw.gwai.cloud", "deepseek": "https://api.deepseek.com"}[BACKEND]',
                    '"https://api.ifm.ai"'))

TAIL = sub(D.TAIL,
           ("""if '"gwai"' not in""", """if 'KEY_AT' not in"""),
           ("(no gwai backend): rebuild the bundle with colab/make_deepseek.py",
            "(no ifm key rotation): rebuild the bundle with kaggle/make_k2.py"),
           ('os.environ.setdefault("GWAI_API_KEY", os.environ["DEEPSEEK_API_KEY"])  # one secret name for both backends\n', ""),
           ('COMMON = ["--backend", BACKEND, "--reasoning", REASONING, "--spend-file", str(SPEND), "--budget", str(BUDGET)]',
            'COMMON = ["--backend", BACKEND]'),
           ('print(f"run folder: {RUN}\\nspent so far: ${spent:.2f} of ${BUDGET:.2f}")',
            'print("run folder:", RUN, "| IFM keys:", len(os.environ["IFM_API_KEY"].replace(",", " ").split()))'),
           ("# budget, PAUSE file or empty balance", "# daily token cap, PAUSE file or rejected key"))

CELL_RUN = sub(D.CELL_RUN, ("(finished ones are not paid again)", "(finished ones are not sent again)"),
               ("deepseek", "k2"))
CELL_TRAIN = sub(D.CELL_TRAIN, ("deepseek", "k2"))

CELL_RESULT = r'''# 3. Validate the submission, summarise the run, export submission.json
import collections
r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", str(SUB)],
                   cwd=ROOT / "train+public_test/public_test", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
shutil.copy(SUB, RUN / "submission.json")  # same JSON Lines content: one line per sample (R-SUB-1)
rows = [json.loads(l) for l in open(TRACES, encoding="utf-8")] if TRACES.is_file() else []
n_lines = sum(1 for _ in open(SUB, encoding="utf-8"))
print(f"K2 Horizon solved {len(rows)} samples; {n_lines - len(rows)} of {n_lines} lines are from the 68.42 base submission")
if rows:
    print("stops:", dict(collections.Counter(o["stop"] for o in rows)))
    print(f"steps/sample {sum(len(o['steps']) for o in rows) / len(rows):.2f}, "
          f"seconds/sample {sum(o['seconds'] for o in rows) / len(rows):.1f}")
print("files:", RUN / "submission.json", SUB, TRACES, sep="\n  ")'''

CELL_STATUS = r'''# Status (any time, needs only cell 1)
for p in sorted(RUN.glob("cache_*.jsonl")):
    print(f"{p.name}: {sum(1 for _ in open(p, encoding='utf-8'))} samples")
print("paused:", PAUSE.exists())'''

if __name__ == "__main__":
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)}]
    cells += [code(c) for c in (SETTINGS + KAGGLE_FILES + TAIL, CELL_RUN, CELL_RESULT, CELL_TRAIN, CELL_STATUS)]
    out = HERE / "kaggle_k2.ipynb"
    json.dump({"cells": cells, "nbformat": 4, "nbformat_minor": 5,
               "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
    D.bundle()
