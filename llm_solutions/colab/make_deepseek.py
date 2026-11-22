"""Builds the DeepSeek V4.1 Flash run of the API pipeline (run: py make_deepseek.py):
  zola_deepseek_bundle.zip  upload to MyDrive/ZOLA_colab/ (competition data: keep it private, R-RES-4). No audio, no keys.
  zola_deepseek.ipynb       the notebook (Colab CPU runtime, or locally in this repo). Cell sources live here.
  ../kaggle/kaggle_deepseek.ipynb  the same notebook for Kaggle (CPU, Secrets, commit runs); only cell 1 differs.
Dev/reference only (F-1): external API at inference. Train gold is the only source of examples and habits (R-RES-3)."""
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZOLA = HERE.parent

MD_INTRO = """# ZAC 2026 LifeMate — DeepSeek V4.1 Flash API pipeline

Same agent as the Nemotron pipeline (`api_agent/solve.py`), with **DeepSeek V4.1 Flash** (`deepseek-flash`,
thinking mode) as the model. Every public sample gets its own `ToolEnv`; the final request is shown as two transcripts
(Gemini + Gipformer).

**How the train set is used.** DeepSeek's API offers no fine-tuning, so the train gold answers are used as
supervision *inside the prompt*: for each sample the 5 most similar train rows with their gold tool calls, and per
installed tool how often gold fills each optional argument (also enforced: arguments gold almost never passes are dropped
or bounced once). No public-test text is ever used as an example (R-RES-3).

**Which key.** `BACKEND = "gwai"` (default) is for a reseller key (genzshop.vn): requests go to `1gw.gwai.cloud` in
Anthropic Messages format, model `deepseek-v4.1-flash-1017`. `BACKEND = "deepseek"` is for a key from
platform.deepseek.com (`api.deepseek.com`). Both use the secret name `DEEPSEEK_API_KEY`. A rejected key stops the run
at once (HTTP 401/403).

**Budget ($10 on the key).** Every request's cost is computed from the API's own token counts at DeepSeek's *peak*
prices (input $0.30 / cached input $0.006 / output $1.20 per 1M tokens; off-peak is half) and added to
`spend_usd.txt` in the run folder. No new sample starts once it reaches `BUDGET`; this total is kept across sessions.
The reseller bills at its own prices, so with `gwai` this total is only an estimate: check the seller's dashboard
after the 3-sample test.
Measured prompt size: ~7.3k tokens on the first step. Estimate: **$2–4 off-peak, $4–7 at peak for all 634 public
samples**, ~$0.5 for the optional 60-row train check. Peak = 01–04 and 06–10 UTC on weekdays (08–11 and 13–17
Vietnam time); run outside those hours to pay half.

**Setup**
1. Run `colab/make_deepseek.py` on your PC and upload `zola_deepseek_bundle.zip` (~10 MB) to `MyDrive/ZOLA_colab/`.
2. Key: Colab sidebar → 🔑 *Secrets* → `DEEPSEEK_API_KEY` (notebook access on). Without it, cell 1 asks for the key
   (hidden input, never written to the notebook or Drive).
3. A **CPU runtime** is enough (no ASR, no GPU).
Locally (no Colab) the notebook uses this repo directly and writes to `colab/deepseek_run/`.

**Run:** cell 1, then cell 2 with `LIMIT = 3` once (costs a few cents, separate files), then `LIMIT = None`.
Pause: press Stop once (running samples finish and are saved) or create `PAUSE` in the run folder. Rerun cell 2 to
resume; finished samples are never paid twice.

**Outputs** (run folder `MyDrive/ZOLA_colab/deepseek_run/`): `submission_deepseek.jsonl` (to upload),
`traces_public_deepseek.jsonl` (per sample: every step's thought, text, tool calls with status and result, cost),
`cache_public_deepseek.jsonl`, `run_deepseek.log`, `spend_usd.txt`. Public traces are reference only: never use them for
training or prompts (F-2)."""

SETTINGS = r'''# 1. Settings, files, API key
import getpass, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
BACKEND = "gwai"      # "gwai" = reseller key (genzshop.vn, 1gw.gwai.cloud); "deepseek" = key from platform.deepseek.com
BUDGET = 9.50         # USD for ALL runs of this notebook together (key has $10; peak prices are assumed)
REASONING = "high"    # DeepSeek thinking effort: low / high / max, or "off" (no thinking, cheaper, usually worse)
WORKERS = 8           # samples at the same time (DeepSeek allows far more; the budget, not the rate, is the limit)
LIMIT = None          # e.g. 3 for a quick test (separate files, ~$0.03); None = all 634 public samples
FILL_FROM_BASE = True # samples not solved (budget, errors) are taken from the 68.42 Nemotron submission
PUBLIC = True         # False: skip the public run (cells 2-3), e.g. for a train-only version
TRAIN_ROWS = 60       # train run (cell 4) with gold text: 0 = skip, 60 = quick check, 1545 = every row (SFT data)
FOLDS = 10            # the train run goes in parts; each part's examples and habits come from the other parts
WHY = False           # train run: one tool per turn with a short written reason before each call (SFT rationales)

'''

COLAB_FILES = r'''try:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    COLAB, ROOT, RUN = True, Path("/content/ZOLA"), Path("/content/drive/MyDrive/ZOLA_colab/deepseek_run")
    if not (ROOT / "api_agent/solve.py").is_file():
        zipfile.ZipFile("/content/drive/MyDrive/ZOLA_colab/zola_deepseek_bundle.zip").extractall(ROOT)
except ImportError:  # local: this repo itself
    COLAB, userdata = False, None
    ROOT = Path.cwd().parent if (Path.cwd().parent / "api_agent").is_dir() else Path.cwd()
    RUN = ROOT / "colab/deepseek_run"
if not os.environ.get("DEEPSEEK_API_KEY"):
    key = None
    if COLAB:
        try:
            key = userdata.get("DEEPSEEK_API_KEY")
        except Exception:
            pass
    os.environ["DEEPSEEK_API_KEY"] = (key or getpass.getpass("DeepSeek API key: ")).strip()
'''

KAGGLE_FILES = r'''COLAB = False
IN = Path(os.environ.get("ZOLA_IN", "/kaggle/input"))
RUN = Path(os.environ.get("ZOLA_OUT", "/kaggle/working"))  # the version's Output tab
ROOT = Path(os.environ.get("ZOLA_ROOT", "/tmp/ZOLA"))     # writable copy in the layout solve.py expects
if not (ROOT / "api_agent/solve.py").is_file():
    src = IN
    z = next(IN.rglob("zola_deepseek_bundle.zip"), None)  # Kaggle normally unpacks an uploaded zip; handle both
    if z:
        src = ROOT.with_name(ROOT.name + "_src")
        zipfile.ZipFile(z).extractall(src)
    # found by content, not folder name, in case Kaggle renames folders with spaces or '+'
    solve = next((p for p in src.rglob("solve.py") if (p.parent / "transcripts_gemini_public_test.json").is_file()), None)
    tj, pj = next(src.rglob("train.jsonl"), None), next(src.rglob("public_test.jsonl"), None)
    if not (solve and tj and pj):
        raise SystemExit("Bundle not found: attach the zola_deepseek_bundle dataset to this notebook (Add Input).")
    shutil.rmtree(ROOT, ignore_errors=True)
    skip = shutil.ignore_patterns("__pycache__")
    shutil.copytree(solve.parent, ROOT / "api_agent", ignore=skip)
    shutil.copytree(tj.parent, ROOT / "train+public_test/train", ignore=skip)
    shutil.copytree(pj.parent, ROOT / "train+public_test/public_test", ignore=skip)

RUN.mkdir(parents=True, exist_ok=True)
for p in IN.rglob("*deepseek*.jsonl"):  # resume: an earlier version's output attached as input
    q = RUN / p.name
    if not q.is_file() or sum(1 for _ in open(p, encoding="utf-8")) > sum(1 for _ in open(q, encoding="utf-8")):
        shutil.copy(p, q)
        print("checkpoint copied from", p)
old = [float(p.read_text()) for p in IN.rglob("spend_usd.txt")]  # keep the budget across versions
if old and max(old) > (float((RUN / "spend_usd.txt").read_text()) if (RUN / "spend_usd.txt").is_file() else 0.0):
    (RUN / "spend_usd.txt").write_text(f"{max(old):.6f}")

if not os.environ.get("DEEPSEEK_API_KEY"):
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["DEEPSEEK_API_KEY"] = UserSecretsClient().get_secret("DEEPSEEK_API_KEY").strip()
    except Exception as e:
        raise SystemExit(f"No key ({type(e).__name__}): Add-ons -> Secrets -> DEEPSEEK_API_KEY, attach it to this notebook.")
import requests
try:
    requests.head({"gwai": "https://1gw.gwai.cloud", "deepseek": "https://api.deepseek.com"}[BACKEND], timeout=15)
except requests.RequestException as e:
    raise SystemExit(f"No internet ({type(e).__name__}): turn on Settings -> Internet, then run again.")
'''

TAIL = r'''RUN.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "api_agent"
if '"--why"' not in (CODE / "solve.py").read_text(encoding="utf-8"):
    raise SystemExit("solve.py is too old (no --why option): rebuild the bundle with colab/make_deepseek.py and upload it again.")
SPEND, PAUSE = RUN / "spend_usd.txt", RUN / "PAUSE"
os.environ.setdefault("GWAI_API_KEY", os.environ["DEEPSEEK_API_KEY"])  # one secret name for both backends
COMMON = ["--backend", BACKEND, "--reasoning", REASONING, "--spend-file", str(SPEND), "--budget", str(BUDGET)]
r = subprocess.run([sys.executable, "solve.py", "--split", "train", *COMMON, "--self-check"], cwd=CODE,
                   capture_output=True, text=True)
print("solve.py self-check:", (r.stdout + r.stderr).strip()[-300:])
assert r.returncode == 0, "solve.py self-check failed"
spent = float(SPEND.read_text()) if SPEND.is_file() else 0.0
print(f"run folder: {RUN}\nspent so far: ${spent:.2f} of ${BUDGET:.2f}")
import signal, time


def run_pass(cmd, log_name):
    """Runs solve.py, mirroring its output to the log. Returns the line that says why it stopped early, or None."""
    halt = None
    with open(RUN / log_name, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        def drain():
            nonlocal halt
            for line in p.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
                if line.startswith("STOPPED:"):  # budget, PAUSE file or empty balance
                    halt = line.strip()
            return p.wait()
        try:
            rc = drain()
        except KeyboardInterrupt:
            PAUSE.touch()
            print("\npausing: running samples finish and are saved (Stop again to abort them)...", flush=True)
            try:
                drain()
            except KeyboardInterrupt:
                p.send_signal(signal.SIGINT)
                time.sleep(3)
                p.kill()
            return "paused by hand"
    return halt or (f"solve.py exited with {rc}" if rc else None)'''
CELL_SETUP = SETTINGS + COLAB_FILES + TAIL
CELL_SETUP_KAGGLE = SETTINGS + KAGGLE_FILES + TAIL

MD_KAGGLE = """# ZAC 2026 LifeMate — DeepSeek V4.1 Flash API pipeline (Kaggle, CPU)

Same agent as the Nemotron pipeline (`api_agent/solve.py`), with **DeepSeek V4.1 Flash** (`deepseek-flash`,
thinking mode). Every public sample gets its own `ToolEnv`; the final request is shown as two transcripts
(Gemini + Gipformer). The train gold answers are used *inside the prompt* (DeepSeek's API has no fine-tuning): per sample
the 5 most similar train rows with their gold tool calls, and how often gold fills each optional argument. No
public-test text is used as an example (R-RES-3).

**Budget ($10 on the key).** Each request's cost (API token counts × DeepSeek's *peak* prices: input $0.30 / cached
$0.006 / output $1.20 per 1M; off-peak is half) is added to `spend_usd.txt`; no new sample starts once it reaches
`BUDGET`. Estimate: **$2–4 off-peak, $4–7 at peak for all 634 samples**, ~$0.5 for the 60-row train check (cell 4).
Peak = 01–04 and 06–10 UTC on weekdays (08–11 and 13–17 Vietnam time). The total carries over to a new version only
if the previous version's output is attached (see Resume); the key's own balance is the final limit either way.

**Which key.** `BACKEND = "gwai"` (default) is for a reseller key (genzshop.vn): requests go to `1gw.gwai.cloud` in
Anthropic Messages format, model `deepseek-v4.1-flash-1017`; the reseller bills at its own prices, so `spend_usd.txt`
is only an estimate there (check the seller's dashboard after the 3-sample test). `BACKEND = "deepseek"` is for a key
from platform.deepseek.com. Both use the secret `DEEPSEEK_API_KEY`. A rejected key stops the run at once (HTTP 401/403).

**One-time setup**
1. Run `colab/make_deepseek.py` on your PC and upload `colab/zola_deepseek_bundle.zip` (~6 MB) as a **private** Kaggle
   dataset (competition data, R-RES-4). *Add Input* → attach it.
2. *Add-ons → Secrets* → add `DEEPSEEK_API_KEY` and attach it to this notebook.
3. *Settings → Internet on* (phone-verified account). *Accelerator: None.*

**How to run**
- Quick test in the editor: `LIMIT = 3` in cell 1, run cells 1–3 (a few cents, separate files).
- Full run: `LIMIT = None`, then *Save Version → Save & Run All (Commit)*. It runs in the background with the browser
  closed: public run, then validation, then the train check (`TRAIN_ROWS = 0` skips it). Expected 1–3 h
  (estimate); Kaggle stops a session at 12 h.
- **Resume** (budget stop, API errors, a killed session): *Add Input → Notebook Outputs* → this notebook's last version,
  then *Save & Run All* again. Cell 1 copies the checkpoints and the spend total; finished samples are not paid twice.

**Train data for the student SFT** (written reasons for `rationales.json`): in cell 1 set `PUBLIC = False`,
`TRAIN_ROWS = 1545`, `WHY = True` and a `BUDGET` that fits the key, then *Save & Run All*. The rows run in `FOLDS`
parts (each part's examples and habits come from the other parts) with gold text; DeepSeek calls one tool per turn and
writes one short reason before each call. About $0.01 per row by the estimate (the 21-row check cost $0.19), so
~$15–20 for all rows; resume like the public run. Then on your PC:
`python student/why_from_traces.py --traces traces_train_deepseek1545_why.jsonl --base <teammate's rationales.json>
--out rationales.json` and attach that file to the SFT notebook (`WHY = True`). Only train traces may be used (R-RES-3).

**Outputs** (`/kaggle/working`, the version's *Output* tab): `submission_deepseek.jsonl` (to upload),
`traces_public_deepseek.jsonl` (per sample: every step's thought, text, tool calls with status and result, cost),
`cache_public_deepseek.jsonl`, `run_deepseek.log`, `spend_usd.txt`, train check files. Public traces are reference
only: never use them for training or prompts (F-2)."""

CELL_RUN = r'''# 2. Public run. Every finished sample is saved at once; rerun this cell to resume (finished ones are not paid again).
#    Pause: press Stop once (running samples finish), or create PAUSE in the run folder. Stop twice = abort.
PAUSE.unlink(missing_ok=True)
tag = f"_test{LIMIT}" if LIMIT else ""
SUB, TRACES = RUN / f"submission_deepseek{tag}.jsonl", RUN / f"traces_public_deepseek{tag}.jsonl"
cmd = [sys.executable, "-u", "solve.py", "--split", "public_test", *COMMON,
       "--transcripts", str(CODE / "transcripts_gemini_public_test.json"),
       "--extra-transcripts", str(CODE / "transcripts_gipformer_public_test.json"),
       "--workers", str(WORKERS), "--cache", str(RUN / f"cache_public_deepseek{tag}.jsonl"), "--traces", str(TRACES),
       "--out", str(SUB), "--stop-file", str(PAUSE)]
cmd += ["--base-submission", str(CODE / "submission_68.42.jsonl")] if FILL_FROM_BASE else []
cmd += ["--limit", str(LIMIT)] if LIMIT else []


print("" if PUBLIC else "PUBLIC = False: public run skipped")
for n in (1, 2) if PUBLIC else ():  # pass 2 only retries samples that failed in pass 1 (API errors)
    print(f"=== pass {n} ===", flush=True)
    halt = run_pass(cmd, f"run_deepseek{tag}.log")
    if halt:
        print("\nHALTED:", halt, "\nEverything finished so far is saved; rerun this cell to resume.")
        break'''

CELL_RESULT = r'''# 3. Validate the submission, summarise the run, download submission + thought traces
if PUBLIC and SUB.is_file():
    import collections
    r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", str(SUB)],
                       cwd=ROOT / "train+public_test/public_test", capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    rows = [json.loads(l) for l in open(TRACES, encoding="utf-8")] if TRACES.is_file() else []
    if rows:
        usd = [o.get("usd", 0) for o in rows]
        print(f"DeepSeek solved {len(rows)} samples, ${sum(usd):.2f} (${sum(usd) / len(rows):.4f} per sample); "
              f"total spent ${float(SPEND.read_text()):.2f} of ${BUDGET:.2f}")
        print("stops:", dict(collections.Counter(o["stop"] for o in rows)))
        n_lines = sum(1 for _ in open(SUB, encoding="utf-8"))
        if len(rows) < n_lines:
            print(f"NOTE: {n_lines - len(rows)} lines are from the 68.42 base submission, not DeepSeek.")
    if COLAB:
        from google.colab import files
        files.download(str(SUB))
        if TRACES.is_file():
            files.download(str(TRACES))
    else:
        print("files:", SUB, TRACES, sep="\n  ")
else:
    print("no public submission to check")'''

CELL_TRAIN = r'''# 4. Train run on TRAIN_ROWS rows with gold text, scored with the official formula. Rerun to resume (finished rows
#    are not paid again). The rows go in FOLDS parts: each part's examples and argument habits come only from the
#    other parts. WHY = True: one tool per turn with a short written reason before each call, the input of
#    student/why_from_traces.py (-> rationales.json for the student SFT notebook).
ids = [json.loads(l)["id"] for l in open(ROOT / "train+public_test/train/train.jsonl", encoding="utf-8") if l.strip()]
pick = ids[::max(1, len(ids) // max(1, TRAIN_ROWS))][:TRAIN_ROWS]
ttag = f"{len(pick)}{'_why' if WHY else ''}"
TCACHE, TTRACES = RUN / f"cache_train_deepseek{ttag}.jsonl", RUN / f"traces_train_deepseek{ttag}.jsonl"
PAUSE.unlink(missing_ok=True)
halt = None
for n in (1, 2) if pick else ():  # pass 2 retries rows that failed in pass 1 (API errors)
    for k in range(FOLDS):
        fold = pick[k::FOLDS]
        if fold and not halt:
            print(f"=== pass {n}, part {k + 1}/{FOLDS} ===", flush=True)
            halt = run_pass([sys.executable, "-u", "solve.py", "--split", "train", *COMMON, "--ids", ",".join(fold),
                             "--workers", str(WORKERS), "--cache", str(TCACHE), "--traces", str(TTRACES),
                             "--stop-file", str(PAUSE)] + (["--why"] if WHY else []), f"run_train_deepseek{ttag}.log")
if halt:
    print("HALTED:", halt, "\nEverything finished so far is saved; rerun this cell to resume.")
if TTRACES.is_file():
    tr = {}
    for l in open(TTRACES, encoding="utf-8"):
        o = json.loads(l)
        tr[o["id"]] = o
    print(f"train: {len(tr)} of {len(pick)} rows, score {100 * sum(o.get('score', 0) for o in tr.values()) / max(1, len(tr)):.2f}, "
          f"${sum(o.get('usd', 0) for o in tr.values()):.2f}\nfiles:", TCACHE, TTRACES, sep="\n  ")'''

CELL_STATUS = r'''# Status (any time, needs only cell 1)
for p in sorted(RUN.glob("cache_*.jsonl")):
    print(f"{p.name}: {sum(1 for _ in open(p, encoding='utf-8'))} samples")
print(f"spent ${float(SPEND.read_text()) if SPEND.is_file() else 0:.2f} of ${BUDGET:.2f} | paused: {PAUSE.exists()}")'''


def nb(kaggle=False):
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}
    md, setup = (MD_KAGGLE, CELL_SETUP_KAGGLE) if kaggle else (MD_INTRO, CELL_SETUP)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)}]
    cells += [code(c) for c in (setup, CELL_RUN, CELL_RESULT, CELL_TRAIN, CELL_STATUS)]
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    if not kaggle:
        meta["colab"] = {"provenance": []}
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": meta}


def bundle():
    code = "api_agent"
    files = {f"{code}/{f}": ZOLA / code / f for f in (
        "solve.py", "transcripts_gemini_public_test.json", "transcripts_gipformer_public_test.json",
        "submission_68.42.jsonl")}
    for split in ("train", "public_test"):  # everything ToolEnv and the validator need; no audio
        d = ZOLA / "train+public_test" / split
        for p in d.rglob("*"):
            rel = p.relative_to(d)
            if p.is_file() and rel.parts[0] != "audio" and "__pycache__" not in rel.parts:
                files[f"train+public_test/{split}/{rel.as_posix()}"] = p
    out = HERE / "zola_deepseek_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in files.items():
            z.write(src, arc)
    print(f"{len(files)} files -> {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    for out, kaggle in ((HERE / "zola_deepseek.ipynb", False), (ZOLA / "kaggle/kaggle_deepseek.ipynb", True)):
        json.dump(nb(kaggle), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", out)
    bundle()
