"""Generates zola_api_pipeline.ipynb (run: py make_notebook.py). Cell sources live here so they can be tested locally."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_INTRO = """# ZAC 2026 LifeMate — API pipeline on Colab (T4)

**What it does**
1. **ASR on the T4 GPU:** the team's fine-tuned Gipformer (Zipformer, int8 ONNX via sherpa-onnx CUDA) transcribes every public audio clip (final requests + earlier user turns). Numbers are converted to digits.
2. **Inference:** Nemotron 3 Ultra (NVIDIA API, free key) solves each sample with tool calls against its own `ToolEnv`. The final request is given as **two transcripts: Gemini + Gipformer** (8.0% / 10.3% word error on unseen train clips; they make different mistakes).
3. Writes and validates `submission_v2.jsonl`. Samples that still fail after a retry pass are filled from the previous run's `submission.jsonl` in the run folder (the 76.08 one), else from the 68.42 submission.
4. **Thought traces:** `traces_public_v2.jsonl` in the run folder, one line per sample: every step's thought and text, each tool call with its status (`ok`, `repeat`, `schema`, `rare_arg`, `missing_arg`, `read_cap`) and result, optional read filters that were dropped, and why the sample stopped. Reference only: never use public traces for training or prompts.

**v2 (this version)** uses new file names (`cache_public_v2.jsonl`, `submission_v2.jsonl`), so the 76.08 run's files are kept and nothing is reused from them.

The GPU is only used for ASR (a few minutes). Inference is API-bound: about **5 hours** for 634 samples.

**One-time setup**
1. Upload `zola_bundle.zip` to Google Drive at `MyDrive/ZOLA_colab/zola_bundle.zip`. It contains competition data, so keep the folder private. Also upload the current `api_agent/solve.py` to `MyDrive/ZOLA_colab/solve.py`: cell 1 copies it over the bundled one (code fixes without re-uploading the bundle) and stops if the old version is in use.
2. Colab sidebar → 🔑 *Secrets* → add `NVIDIA_API_KEY` (your `nvapi-…` key) and enable notebook access.
3. *Runtime → Change runtime type → T4 GPU.* The GPU is needed only for ASR (cell 3). If `transcripts_gipformer_public_test.json` is already in the run folder from an earlier run, a **CPU runtime works**: cell 3 finds every clip done and inference is API calls only.

**Pause / resume at will (checkpoints on Drive, `MyDrive/ZOLA_colab/run/`)**
- *ASR (cell 3):* saved after every 32 clips. Press Stop any time; run the cell again to continue.
- *Inference (cell 4):* every finished sample is saved immediately. To pause, press Stop once: no new samples start, the ones in progress finish and are saved (a few minutes), then the cell ends. Press Stop twice to abort right away (only the in-progress samples are redone). You can also pause from your phone or another PC by creating a file named `PAUSE` in that Drive folder.
- *Resume:* after a pause, a disconnect or a new day, run cells 1–3 (fast: they skip finished work), then cell 4. Resuming deletes `PAUSE` automatically.
- *Status cell:* shows progress any time (needs only cell 1).
- Free Colab stops after ~12 h or when idle with the tab closed; nothing finished is lost."""

CELL_SETUP = r'''# 1. GPU check, Google Drive, extract the bundle to local disk
import os, shutil, subprocess, sys, zipfile
GPU = bool(shutil.which("nvidia-smi"))  # a CPU runtime has no nvidia-smi at all
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout if GPU else
      "NO GPU: fine if the ASR transcripts are already in the run folder (cell 3 then has nothing to do); "
      "otherwise ASR runs on the CPU, much slower.")
from google.colab import drive
drive.mount("/content/drive")

BUNDLE = "/content/drive/MyDrive/ZOLA_colab/zola_bundle.zip"
RUN = "/content/drive/MyDrive/ZOLA_colab/run"   # caches, transcripts, submission (survive disconnects)
ROOT = "/content/ZOLA"                          # extracted bundle (fast local disk)
os.makedirs(RUN, exist_ok=True)
if not os.path.isdir(f"{ROOT}/train+public_test/public_test"):
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(ROOT)
print("bundle ready:", sorted(os.listdir(ROOT)))
# Code updates without re-uploading the bundle: MyDrive/ZOLA_colab/solve.py (if present) replaces the bundled one
import shutil
SOLVE_DRIVE, SOLVE_LOCAL = "/content/drive/MyDrive/ZOLA_colab/solve.py", f"{ROOT}/api_agent/solve.py"
if os.path.exists(SOLVE_DRIVE):
    shutil.copy(SOLVE_DRIVE, SOLVE_LOCAL)
if "def habits" in open(SOLVE_LOCAL, encoding="utf-8").read():
    print("solve.py: OK (v2: argument habits, read cap, thought traces)")
else:
    raise SystemExit("solve.py is an OLD version (no argument habits / thought traces). "
                     "Upload the current api_agent/solve.py to MyDrive/ZOLA_colab/solve.py and run this cell again.")'''

CELL_INSTALL = r'''# 2. sherpa-onnx with CUDA (same version as the teammate's CPU run: 2.0.1), CPU build as fallback
import shutil, subprocess, sys
PROVIDER = "cuda" if shutil.which("nvidia-smi") else "cpu"
if PROVIDER == "cuda":
    cuda_major = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.split("CUDA Version:")[-1].split(".")[0].strip()
    tag = "cuda13" if cuda_major == "13" else "cuda12"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"sherpa-onnx==2.0.1+{tag}.cudnn9.onnxruntime1.28.2",
                        "-f", "https://k2-fsa.github.io/sherpa/onnx/cuda.html"], capture_output=True, text=True)
    if r.returncode != 0:
        print("CUDA build not available for this runtime, using CPU:", r.stderr[-300:])
        PROVIDER = "cpu"
if PROVIDER == "cpu":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sherpa-onnx==2.0.1"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile", "requests"], check=True)
import sherpa_onnx
print("sherpa-onnx", sherpa_onnx.__version__, "| provider:", PROVIDER)'''

CELL_ASR = r'''# 3. Gipformer ASR on the T4 -> RUN/transcripts_gipformer_public_test.json  (keys: <id>, <id>_u<n>)
import json, os, re, sys, time, unicodedata
import soundfile as sf
import sherpa_onnx
sys.path.insert(0, f"{ROOT}/asr_zipformer")
from vi_itn import itn  # spoken numbers -> digits, the teammate's rules

TONES = {"̀", "́", "̃", "̉", "̣"}
def old_style_tone(word):  # hoà -> hòa: same post-processing as the teammate's asr_common.norm
    d = unicodedata.normalize("NFD", word)
    tones = [c for c in d if c in TONES]
    if len(tones) != 1:
        return word
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))
    if base.endswith(("oa", "oe")) or (base.endswith("uy") and not base.endswith("quy")):
        return unicodedata.normalize("NFC", base[:-2] + base[-2] + tones[0] + base[-1])
    return word
def norm(s):
    s = re.sub(r"[^\w\s]|_", " ", unicodedata.normalize("NFC", s).lower())
    return " ".join(old_style_tone(w) for w in s.split())

PUB = f"{ROOT}/train+public_test/public_test"
items = [json.loads(l) for l in open(f"{PUB}/public_test.jsonl", encoding="utf-8") if l.strip()]
jobs = []
for it in items:
    n = 0
    for h in it["history"]:
        if h["role"] == "user":
            n += 1
            jobs.append((f"{it['id']}_u{n}", h["audio"]))
    jobs.append((it["id"], it["audio"]))
GIP_OUT = f"{RUN}/transcripts_gipformer_public_test.json"
res = json.load(open(GIP_OUT, encoding="utf-8")) if os.path.exists(GIP_OUT) else {}
todo = [j for j in jobs if j[0] not in res]
print(f"{len(jobs)} clips, {len(res)} done, {len(todo)} to transcribe on {PROVIDER}")
if todo:
    md = f"{ROOT}/asr_zipformer/asr_model"
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{md}/encoder.int8.onnx", decoder=f"{md}/decoder.int8.onnx", joiner=f"{md}/joiner.int8.onnx",
        tokens=f"{md}/tokens.txt", num_threads=os.cpu_count(), provider=PROVIDER)  # greedy search: deterministic
    t0, dur, B = time.time(), 0.0, 32
    def save():  # atomic: a disconnect mid-write never corrupts the checkpoint
        json.dump(res, open(GIP_OUT + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(GIP_OUT + ".tmp", GIP_OUT)
    try:
        for i in range(0, len(todo), B):
            streams = []
            for _, a in todo[i:i + B]:
                w, sr = sf.read(f"{PUB}/{a}", dtype="float32")
                dur += len(w) / sr
                streams.append(rec.create_stream())
                streams[-1].accept_waveform(sr, w)
            try:
                rec.decode_streams(streams)
            except RuntimeError as e:  # out of memory on a batch of long clips -> one at a time
                print(f"  batch at {i} failed ({str(e)[:80]}...), decoding one by one")
                for s in streams:
                    rec.decode_stream(s)
            for (k, _), s in zip(todo[i:i + B], streams):
                res[k] = itn(norm(s.result.text))
            save()  # checkpoint after every batch of 32 clips
            if (i // B) % 10 == 0 or i + B >= len(todo):
                el = time.time() - t0
                print(f"  {min(i + B, len(todo))}/{len(todo)}  {dur / 60:.0f} min audio in {el:.0f}s ({dur / el:.0f}x real time)")
    except KeyboardInterrupt:
        save()
        raise SystemExit(f"paused: {len(res)}/{len(jobs)} clips saved. Run this cell again to resume.")
ref_path = f"{ROOT}/api_agent/transcripts_gipformer_reference.json"  # teammate's own CPU output
if os.path.exists(ref_path):
    ref = json.load(open(ref_path, encoding="utf-8"))
    common = [k for k in ref if k in res]
    same = sum(res[k] == ref[k] for k in common)
    print(f"identical to the teammate's transcripts: {same}/{len(common)} (a few differ: CPU vs GPU kernels, library build)")
print("example:", items[0]["id"], "->", res[items[0]["id"]])'''

CELL_RUN = r'''# 4. Inference: Nemotron 3 Ultra via NVIDIA API, Gemini + Gipformer transcripts.
#    Checkpoint: every finished sample is appended to the cache on Drive -> rerun this cell to resume.
#    PAUSE: press Colab's Stop button (or create a file named PAUSE in MyDrive/ZOLA_colab/run from any device).
#           Running samples finish and are saved (can take a few minutes); press Stop twice to abort at once.
import os, signal, subprocess, sys, time
from google.colab import userdata
os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")
LIMIT = None          # e.g. 3 for a quick test (uses a separate cache); None = all 634 samples
WORKERS = 4
tag = f"_test{LIMIT}" if LIMIT else ""
CODE = f"{ROOT}/api_agent"
BASE = f"{RUN}/submission.jsonl" if os.path.exists(f"{RUN}/submission.jsonl") else f"{CODE}/submission_68.42.jsonl"
cmd = [sys.executable, "-u", "solve.py", "--split", "public_test", "--backend", "nim",
       "--model", "nvidia/nemotron-3-ultra-550b-a55b",
       "--transcripts", f"{CODE}/transcripts_gemini_public_test.json",
       "--extra-transcripts", f"{RUN}/transcripts_gipformer_public_test.json",
       "--workers", str(WORKERS), "--cache", f"{RUN}/cache_public_v2{tag}.jsonl",  # traces: traces_public_v2{tag}.jsonl
       "--base-submission", BASE, "--out", f"{RUN}/submission_v2{tag}.jsonl"]
PAUSE = f"{RUN}/PAUSE"
if os.path.exists(PAUSE):
    os.remove(PAUSE)  # running this cell = resume
cmd += ["--stop-file", PAUSE] + (["--limit", str(LIMIT)] if LIMIT else [])


def run_pass(n):
    print(f"=== pass {n} ===", flush=True)
    with open(f"{RUN}/run_v2{tag}.log", "a", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            for line in p.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return p.wait()
        except KeyboardInterrupt:  # Stop button: graceful pause
            open(PAUSE, "w").close()
            print("\n⏸ pausing: no new samples; waiting for running ones to finish and save "
                  "(press Stop again to abort them)...", flush=True)
            try:
                for line in p.stdout:
                    print(line, end="")
                    log.write(line)
                p.wait()
            except KeyboardInterrupt:
                p.send_signal(signal.SIGINT)
                time.sleep(3)
                p.kill()
                print("aborted: samples that were running will rerun on resume")
            return "paused"


for n in (1, 2):  # pass 2 only retries samples that failed (e.g. "retries exhausted")
    rc = run_pass(n)
    if rc == "paused" or os.path.exists(PAUSE):
        print("\n⏸ PAUSED. Everything finished so far is saved. Run this cell again to resume.")
        break
    if rc != 0:
        print("solve.py exited with", rc)
        break'''

CELL_STATUS = r'''# Status (safe to run any time, e.g. after a reconnect: needs only cell 1)
import json, os
def count_lines(p):  # samples with a real result (cached empty API replies are redone, so not counted)
    if not os.path.exists(p):
        return 0
    ids = set()
    for l in open(p, encoding="utf-8"):
        try:
            o = json.loads(l)
            if o.get("tool_calls") or (o.get("final") or "").strip():
                ids.add(o["id"])
        except Exception:
            pass
    return len(ids)
gip = f"{RUN}/transcripts_gipformer_public_test.json"
print("ASR (Gipformer):", len(json.load(open(gip, encoding="utf-8"))) if os.path.exists(gip) else 0, "/ 1933 clips")
print("inference      :", count_lines(f"{RUN}/cache_public_v2.jsonl"), "/ 634 samples")
print("paused flag    :", os.path.exists(f"{RUN}/PAUSE"))
print("submission     :", os.path.exists(f"{RUN}/submission_v2.jsonl"))
log = f"{RUN}/run_v2.log"
if os.path.exists(log):
    print("\nlast log lines:\n" + "".join(open(log, encoding="utf-8").readlines()[-5:]))'''

CELL_VALIDATE = r'''# 5. Validate with the organisers' checker, then download the submission and the thought traces
import os, subprocess, sys
tag = globals().get("tag", "")  # set by cell 4; "" = the full run
SUB = f"{RUN}/submission_v2{tag}.jsonl"
r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", SUB],
                   cwd=f"{ROOT}/train+public_test/public_test", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
from google.colab import files
files.download(SUB)
TRACES = f"{RUN}/traces_public_v2{tag}.jsonl"  # thought traces, one line per sample (also kept on Drive)
if os.path.exists(TRACES):
    files.download(TRACES)'''


def nb():
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD_INTRO.splitlines(keepends=True)}]
    cells += [code(c) for c in (CELL_SETUP, CELL_INSTALL, CELL_ASR, CELL_RUN, CELL_STATUS, CELL_VALIDATE)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


if __name__ == "__main__":
    out = HERE / "zola_api_pipeline.ipynb"
    json.dump(nb(), open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
