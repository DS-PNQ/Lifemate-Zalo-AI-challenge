"""Builds the Kaggle train run of the Nemotron API pipeline (run: py make_kaggle.py):
  zola_train_bundle.zip       upload as a PRIVATE Kaggle dataset (competition data, R-RES-4). No audio, no API keys.
  kaggle_nemotron_train.ipynb the notebook. Cell sources live here so they can be tested locally.
Dev/reference only (F-1): external API at inference. Train split only (R-RES-3)."""
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZOLA = HERE.parent

MD_INTRO = """# ZAC 2026 LifeMate — Nemotron API pipeline on the **train** split (Kaggle, CPU)

Runs `solve.py` (Nemotron 3 Ultra via the NVIDIA API) on all 1545 train rows with the gold request text, scores every
row with the official formula and saves the thought traces. No GPU and no audio are needed: use a **CPU** session, it
costs no GPU quota.

**Folds.** The rows are run in `FOLDS` parts. Each part gets its argument habits only from the other parts, and a row
is never its own example, so no row sees its own gold answer.

**One-time setup**
1. Run `kaggle/make_kaggle.py` on your PC and upload `zola_train_bundle.zip` (about 5 MB) as a **private** Kaggle
   dataset. *Add Input* → attach it to this notebook.
2. *Add-ons → Secrets* → add `NVIDIA_API_KEY` (your `nvapi-…` key) and attach it to this notebook.
3. *Settings → Internet on* (needs a phone-verified account). *Accelerator: None.*

**How to run**
- Quick test: set `LIMIT = 6` in cell 1 and run the cells in the editor.
- Full run: *Save Version → Save & Run All (Commit)*. It runs in the background, with the browser closed, for up to
  12 hours. The notebook stops itself after `HOURS` (11) so the session ends normally and the results are kept.
  A session that Kaggle kills at 12 hours loses everything.
- **Resume** (a full run is likely more than one session): open the notebook, *Add Input → Notebook Outputs* → this
  notebook's last version, then *Save & Run All* again. Cell 1 copies the largest checkpoint it finds in the inputs
  and only the missing rows are run. Keep `FOLDS` and `LIMIT` unchanged between sessions.

**Results** (in `/kaggle/working`, the version's *Output* tab)
- `cache_train_nemotron.jsonl`: per row the tool calls, state hash, final text, stop reason.
- `traces_train_nemotron.jsonl`: per row the score and every step's thought, text, calls with status and result.
- `run_train_nemotron.log`: the full log."""

CELL_SETUP = r'''# 1. Settings, bundle, checkpoint from an earlier version, API key, internet check
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
T0 = time.time()
LIMIT = None      # e.g. 6 for a quick test (separate files); None = all 1545 rows
WORKERS = 4       # rows solved at the same time; raise only if the API does not answer with rate-limit errors
FOLDS = 10        # keep unchanged when resuming
HOURS = 11.0      # stop starting new rows after this long; Kaggle kills a session at 12 h and then nothing is saved
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

IN = Path(os.environ.get("ZOLA_IN", "/kaggle/input"))
OUT = Path(os.environ.get("ZOLA_OUT", "/kaggle/working"))
ROOT = Path(os.environ.get("ZOLA_ROOT", "/tmp/ZOLA"))  # fixed layout solve.py expects; not part of the output
CODE, TRAIN = ROOT / "code", ROOT / "train+public_test/train"
if not (CODE / "solve.py").is_file():
    src = IN
    z = next(IN.rglob("zola_train_bundle.zip"), None)  # Kaggle normally unpacks an uploaded zip; handle both
    if z:
        src = ROOT.with_name(ROOT.name + "_src")
        zipfile.ZipFile(z).extractall(src)
    solve, tj = next(src.rglob("solve.py"), None), next(src.rglob("train.jsonl"), None)
    if not (solve and tj):
        raise SystemExit("Bundle not found: attach the zola_train_bundle dataset to this notebook (Add Input).")
    shutil.rmtree(ROOT, ignore_errors=True)
    CODE.mkdir(parents=True)
    shutil.copy(solve, CODE)
    shutil.copytree(tj.parent, TRAIN, ignore=shutil.ignore_patterns("audio", "__pycache__"))
if "def habits" not in (CODE / "solve.py").read_text(encoding="utf-8"):
    raise SystemExit("solve.py in the bundle is an old version. Rebuild the bundle with kaggle/make_kaggle.py.")

tag = f"_test{LIMIT}" if LIMIT else ""
CACHE, TRACES = OUT / f"cache_train_nemotron{tag}.jsonl", OUT / f"traces_train_nemotron{tag}.jsonl"
LOG, PAUSE = OUT / f"run_train_nemotron{tag}.log", OUT / "PAUSE"


def done_ids(p):  # rows with a real result, the same test as solve.py's cache loader
    ids = set()
    if Path(p).is_file():
        for l in open(p, encoding="utf-8"):
            try:
                o = json.loads(l)
                if o.get("tool_calls") or (o.get("final") or "").strip():
                    ids.add(o["id"])
            except Exception:
                pass
    return ids


for p in IN.rglob(CACHE.name):  # resume: the output of an earlier version attached as input
    if len(done_ids(p)) > len(done_ids(CACHE)):
        shutil.copy(p, CACHE)
        if p.with_name(TRACES.name).is_file():
            shutil.copy(p.with_name(TRACES.name), TRACES)
        print("checkpoint copied from", p)
print("rows already done:", len(done_ids(CACHE)))

if not os.environ.get("NVIDIA_API_KEY"):
    from kaggle_secrets import UserSecretsClient
    os.environ["NVIDIA_API_KEY"] = UserSecretsClient().get_secret("NVIDIA_API_KEY")
import requests
try:
    requests.head("https://integrate.api.nvidia.com", timeout=15)
except requests.RequestException as e:
    raise SystemExit(f"No internet ({type(e).__name__}): turn on Settings -> Internet, then run again.")
BASE_CMD = [sys.executable, "-u", "solve.py", "--split", "train", "--backend", "nim", "--model", MODEL]
r = subprocess.run(BASE_CMD + ["--self-check"], cwd=CODE, capture_output=True, text=True)
print("solve.py self-check:", (r.stdout + r.stderr).strip()[-300:])
assert r.returncode == 0, "solve.py self-check failed"'''

CELL_RUN = r'''# 2. Inference, fold by fold. Every finished row is saved at once; a rerun only does the missing rows.
#    Interactive pause: press Stop once (running rows finish and are saved), twice to abort.
import signal, threading
ids = [json.loads(l)["id"] for l in open(TRAIN / "train.jsonl", encoding="utf-8") if l.strip()][:LIMIT]
folds = [ids[k::FOLDS] for k in range(FOLDS)]
PAUSE.unlink(missing_ok=True)
timer = threading.Timer(max(0, T0 + HOURS * 3600 - time.time()), PAUSE.touch)  # time budget -> graceful stop
timer.daemon = True
timer.start()


def run_fold(fold):
    """Runs solve.py on one fold (its habits come from the other folds). Returns a reason to halt, or None."""
    cmd = BASE_CMD + ["--ids", ",".join(fold), "--workers", str(WORKERS), "--cache", str(CACHE),
                      "--traces", str(TRACES), "--stop-file", str(PAUSE)]
    halt = None
    with open(LOG, "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

        def drain():
            nonlocal halt
            for line in p.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
                if line.startswith("STOPPED:"):  # time budget, PAUSE file or daily quota
                    halt = line.strip()
            return p.wait()
        try:
            rc = drain()
        except KeyboardInterrupt:
            PAUSE.touch()
            print("\npausing: running rows finish and are saved (Stop again to abort them)...", flush=True)
            try:
                rc = drain()
            except KeyboardInterrupt:
                p.send_signal(signal.SIGINT)
                time.sleep(3)
                p.kill()
            return "stopped by hand"
    return halt or (f"solve.py exited with {rc}" if rc else None)


halt = None
for n in (1, 2):  # pass 2 retries rows that failed in pass 1 (API errors, empty replies)
    for k, fold in enumerate(folds):
        left = len(set(fold) - done_ids(CACHE))
        if not left or halt:
            continue
        print(f"\n=== pass {n}, fold {k + 1}/{FOLDS}: {left} of {len(fold)} rows to run ===", flush=True)
        halt = run_fold(fold)
timer.cancel()
if halt:
    print("\nHALTED:", halt, "\nEverything finished so far is saved. See the intro for how to resume.")'''

CELL_SUMMARY = r'''# 3. Score so far (official formula, computed by solve.py per row and stored in the traces)
import collections
rows = {}
if TRACES.is_file():
    for l in open(TRACES, encoding="utf-8"):
        try:
            o = json.loads(l)
            rows[o["id"]] = o
        except Exception:
            pass
rows = [rows[i] for i in done_ids(CACHE) if i in rows]
n_all = len(ids) if "ids" in globals() else 1545
if rows:
    sc = [o["score"] for o in rows]
    print(f"rows done: {len(rows)} / {n_all}" + ("" if len(rows) >= n_all else "  -> NOT finished: resume in a new version"))
    print(f"TRAIN SCORE: {100 * sum(sc) / len(sc):.2f}  | full score {sum(s == 1 for s in sc)}, zero {sum(s == 0 for s in sc)}")
    for stop, v in sorted(collections.Counter(o["stop"] for o in rows).items()):
        print(f"  stop={stop}: {v} rows, score {100 * sum(o['score'] for o in rows if o['stop'] == stop) / v:.2f}")
else:
    print("no rows done yet")'''


def nb():
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD_INTRO.splitlines(keepends=True)}]
    cells += [code(c) for c in (CELL_SETUP, CELL_RUN, CELL_SUMMARY)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


def bundle():
    train = ZOLA / "train+public_test/train"
    files = {"solve.py": ZOLA / "api_agent/solve.py"}
    for p in train.rglob("*"):  # everything ToolEnv needs; the gold-text run needs no audio
        rel = p.relative_to(train)
        if p.is_file() and rel.parts[0] != "audio" and "__pycache__" not in rel.parts:
            files[f"train/{rel.as_posix()}"] = p
    out = HERE / "zola_train_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in files.items():
            z.write(src, arc)
    print(f"{len(files)} files -> {out} ({out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    out = HERE / "kaggle_nemotron_train.ipynb"
    json.dump(nb(), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
    bundle()
