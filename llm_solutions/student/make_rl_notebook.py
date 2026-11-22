"""Builds student/notebooks/qwen_k2_rl.ipynb: one Kaggle session (2x T4) that takes an old Qwen3.5-4B checkpoint,
rolls it out on train rows in the real ToolEnv (reward = official score), lets K2 Horizon write reasons for the same
train rows, fine-tunes the checkpoint on the reward-selected rows, and writes a new public submission. Every stage is
sized from the time left to DEADLINE_VN.
Run: py student/make_rl_notebook.py
infer.py / train_llm.py are the cells of neweval_sft / bowen_sft with exact-string patches (each must match once).
Rules: training data = train split only (gold, rewards, K2 reasons: R-RES-2/3/5); public_test is only inferred, offline
model and 01c transcripts (R-RUN-1); one ToolEnv per sample (R-ENV-2), greedy + seeds (R-REP-1/2)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "notebooks"


def cell_src(nb, start):
    cells = json.load(open(NB / nb, encoding="utf-8"))["cells"]
    (src,) = ["".join(c["source"]) for c in cells if "".join(c["source"]).startswith(start)]
    return src


def patch(src, *pairs):
    for old, new in pairs:
        assert src.count(old) == 1, f"must match once, found {src.count(old)}: {old[:70]!r}"
        src = src.replace(old, new)
    return src


INFER = patch(
    cell_src("neweval_sft.ipynb", "%%writefile infer.py"),
    ("import json\nimport re\n", "import json\nimport os\nimport re\n"),
    ('if split == "train":\n    rows = sum(lm.split_rows(rows)[1:], [])  # only held-out samples say anything\n',
     'if split == "train":  # rollout / gate rows in the order of the ids file. Scored with gold, so train rows only\n'
     '    _by = {x["id"]: x for x in rows}\n    rows = [_by[i] for i in json.load(open(os.environ["IDS_JSON"]))]\n'),
    ("            if time.time() - t0 > TIME_BUDGET:", "            if time.time() - t0 > budget:"),
    ('with open(f"pred_{split}_{mode}_{shard}.jsonl", "w", encoding="utf-8") as f:\n    for n, item in enumerate(rows):\n',
     'STOP_AT, budget = float(os.environ["STOP_AT"]), TIME_BUDGET\n'
     'with open(f"pred_{os.environ[\'TAG\']}_{split}_{shard}.jsonl", "w", encoding="utf-8") as f:\n'
     '    for n, item in enumerate(rows):\n'
     '        left = STOP_AT - time.time()\n'
     '        if split == "train" and left < 0:  # rollouts and the gate: rows not reached are simply not used\n'
     '            break\n'
     '        budget = min(TIME_BUDGET, max(5.0, left / (len(rows) - n)))  # public: the time left is shared out, so every sample gets its line before the deadline (R-RUN-3)\n'),
)

TRAIN = patch(
    cell_src("bowen_sft.ipynb", "%%writefile train_llm.py"),
    ("rows = rows if FINAL else lm.split_rows(rows)[0]\n",
     '_by = {x["id"]: x for x in rows}\n'
     'rows = [_by[i] for i in json.load(open("rl_rows.json"))]  # reward-selected train rows, failed rollouts twice\n'),
)

WHY = "%%writefile why_from_traces.py\n" + (HERE / "why_from_traces.py").read_text(encoding="utf-8")

MD = """# Qwen3.5-4B: K2 reasons + reward-filtered fine-tune of an old checkpoint → new public submission

Built by `student/make_rl_notebook.py`. One session, **GPU T4 x2, Internet on**, sized to have the new model ready by
`DEADLINE_VN` (default 20:30: the eval notebook's 2 h and the Docker build's 1 h still follow before 24:00).

**What it does**
1. **Rollouts** — the old checkpoint acts on train rows in the real `ToolEnv` (ASR text, greedy), 2 GPUs.
   Reward = the official score against train gold. Row order: 66 gate rows (never trained on here), the other 262
   held-out rows, then train rows until the stage's time is up.
2. **K2 Horizon** (background, CPU, optional) — `solve.py --backend ifm --why` on the same train rows: one tool per turn
   with one written reason. Used only where K2 itself scored 1.0 in gold order (`why_from_traces.py`).
3. **Policy improvement** — a fresh LoRA on the old weights, trained on the rows selected by reward: every failed rollout
   twice (gold trajectory as the target, K2's reasons where available), then successes on rows the checkpoint never
   trained on, then other successes, as many as fit in the stage's time. Merged → `llm_model/`.
4. **Gate** — the new model on the 66 gate rows against the old model's rollouts there (~12 min).
5. **Public run** only with `PUBLIC_RUN = True` (about 2 h more). By default the eval notebook makes the submission:
   attach this notebook's output (`llm_model/`, `lifemate.py`) to it.

**Eval notebook**: its held-out score is **not valid for this model** — 262 of the 328 held-out rows are trained on
here. Only the gate line compares old and new fairly. Run the eval notebook with `EVAL = False` (public only).

This is one round of reward-filtered imitation (rollout → reward → train on the corrected failures), not policy-gradient
RL: sampling several trajectories per row and several rounds do not fit in the hours left.

**Attach (Add Input)**
1. the competition data package (private),
2. the old checkpoint's notebook output: `llm_model/` + `lifemate.py` (exactly one `llm_model` among the inputs),
3. the 01c output: `asr_transcripts.json`, `asr_base_train.json` (or `asr_xfit_train.json`), `vi_itn.py`,
4. `rationales.json` — the file the checkpoint was trained with (all 1,545 train ids),
5. optional, for K2: the `zola_deepseek_bundle` dataset and the secret `IFM_API_KEY` (Add-ons → Secrets, attached).
   Without them the run goes on with the reasons of `rationales.json`.

**Run**: optional `SMOKE = True` first (~20 min: 8 rows per GPU, 3 updates, merge). Then
`SMOKE = False` → *Save Version → Save & Run All (Commit)*. The first code cell prints the time plan; it stops at once
if too little time is left.

**Output**: `llm_model/` + `lifemate.py` (new weights), `rl_summary.json` (gate result), `pred_*.jsonl`
(rollouts with rewards), `rl_rows.json`, `rationales_rl.json`, `traces_train_k2_why.jsonl`, `train.log`.

**Rules**: K2 sees train rows only and only to make training data (R-RES-2/3); nothing from public_test is trained on;
the public run uses the local model and 01c text only. Never attach `public_test_rationales.json` or public traces."""

CELL_ENV = r'''# 1. GPUs and packages (the versions the checkpoint was trained with)
import torch
assert torch.cuda.device_count() == 2, "Set Settings > Accelerator to GPU T4 x2"
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])
!pip install -q "transformers==5.22.1" "peft==0.21.2" "accelerate==1.15.0"
!pip uninstall -q -y torchao  # unused, and peft refuses the old version Kaggle ships
!python -c "import peft, transformers; import transformers.models.qwen3_5.modeling_qwen3_5 as q; assert hasattr(q, 'torch_chunk_gated_delta_rule'); print('transformers', transformers.__version__, 'peft', peft.__version__)"
assert _exit_code == 0, "the training script needs transformers 5.22.1 with the Qwen3.5 model code: is Internet on?"'''

CELL_SETTINGS = r'''# 2. Settings
import datetime as dt, glob, json, os, random, shutil, statistics as st, subprocess, sys, time
from pathlib import Path

SMOKE = False                      # True: 8 rows per GPU, 3 updates, 16 public samples (~25 min). Then False and Save & Run All
DEADLINE_VN = "2026-11-16 20:30"   # Vietnam time: llm_model/ is ready by then. After it: the commit saves ~9 GB of output, the eval notebook (2 h), the Docker build (1 h), all before 24:00
PUBLIC_RUN = False                 # False: the eval notebook makes the submission from llm_model/. True: also run public_test here (about 2 h more, before the deadline)
ROLL_CAP, TRAIN_CAP = 55, 80       # minutes at most for the rollouts and for training; both shrink when less time is left
PUBLIC_MIN = 120                   # first estimate for the public run, replaced by the speed the rollouts measure
SEC_PER_CONV = 7.7                 # training seconds per conversation on 2x T4 (bowen: 459 updates of 8 in 7.8 h)
USE_K2, K2_WORKERS, K2_ROWS = True, 4, 600  # K2 Horizon reasons for train rows; 10M tokens per key per day
os.environ["SMOKE"] = "1" if SMOKE else "0"
DEADLINE = dt.datetime.strptime(DEADLINE_VN, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone(dt.timedelta(hours=7))).timestamp()


def left():
    """Minutes to the deadline."""
    return (DEADLINE - time.time()) / 60


def run_logged(cmd, log, env=None):
    with open(log, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        return p.wait()


GATE_MIN = 15  # the new model on the 66 gate rows
flex = left() - 10 - 15 - GATE_MIN - PUBLIC_MIN * PUBLIC_RUN  # what rollouts + training can share: minus model loads, merge, gate, public run
print(f"{left():.0f} min to {DEADLINE_VN} (+07); rollouts and training share {flex:.0f} min")
assert SMOKE or flex >= 45, "too late for rollouts + training before the deadline: move DEADLINE_VN at your own risk"
ROLL_MIN = 6 if SMOKE else min(ROLL_CAP, 0.4 * flex)'''

SETTINGS_PY = r'''%%writefile settings.py
"""Shared by infer.py (neweval's settings), train_llm.py (bowen_sft's settings) and merge.py. The notebook passes the
per-stage values (model to load, training hours) as environment variables."""
import glob
import json
import os
from pathlib import Path

IN = os.environ.get("ZOLA_IN", "/kaggle/input")
SMOKE = os.environ.get("SMOKE") == "1"
SMOKE_STEPS = 3
SEED = 42

# inference, as neweval
DTYPE = "float16"
SHARDS = 2  # one process per T4
MAX_STEPS = 8
MAX_NEW_TOKENS = 320
TIME_BUDGET = 60.0  # per sample at most (the slowest sample of neweval's T4 run took 56.7 s); infer.py lowers it when the deadline is near
STRIP_MARKS = True
ASR_JSON = "asr_all.json"  # public_test: 01c text; train: the noisy text the checkpoint was trained on

# training, as bowen_sft except EPOCHS, LR, WARMUP_STEPS, MAX_HOURS, SAVE_STEPS and the base model
FINAL = False  # unused: the rows come from rl_rows.json
FP16 = True
MAX_LEN = 9000
DELTA_PART = 512
EPOCHS = 1  # rl_rows.json already repeats the failed rows
GRAD_ACCUM = 8
LR = 1e-4  # half of the first fine-tune: a short correction on already trained weights
WARMUP_STEPS = 5
LORA_R = 32
NOISE_RATE = 0.5
HISTORY_CALLS = True
WHY = json.load(open("rl_config.json"))["why"] if os.path.exists("rl_config.json") else True  # does the checkpoint write a reason first? Measured on the rollouts
WHY_HISTORY = False
MASK_RATE = 0.3
MAX_HOURS = float(os.environ.get("TRAIN_HOURS", 1.0))  # train_llm.py stops, saves and exits cleanly at this point
MIN_FREE_GB = 3
SAVE_STEPS = 100000  # no checkpoints on the way: the run is short and stops by time
RESUME, CHECKPOINT = False, None

ROOT = Path(sorted(glob.glob(f"{IN}/**/train/train.jsonl", recursive=True))[0]).parents[1]
_llm = sorted(glob.glob(f"{IN}/**/llm_model/config.json", recursive=True))
assert len(_llm) == 1, f"attach exactly one checkpoint output with llm_model/, found {_llm}"
LLM_OLD = str(Path(_llm[0]).parent)
LLM = os.environ.get("LLM_DIR") or LLM_OLD  # the model infer.py loads
MODEL_ID = LLM_OLD  # training starts from the old checkpoint's merged weights
LIFEMATE_DIR = "."  # the checkpoint's own lifemate.py, copied here by the notebook: the prompt must match the weights
_noisy = glob.glob(f"{IN}/**/asr_xfit_train.json", recursive=True) or glob.glob(f"{IN}/**/asr_base_train.json", recursive=True)
assert _noisy, "attach the 01c output: asr_base_train.json (or asr_xfit_train.json) is the train text"
NOISY_ASR = _noisy[0]
_pub = glob.glob(f"{IN}/**/asr_transcripts.json", recursive=True)
assert len(_pub) == 1, f"attach exactly one 01c output with asr_transcripts.json, found {_pub}"
ASR_PUBLIC = _pub[0]
_why = glob.glob(f"{IN}/**/rationales.json", recursive=True)
assert _why, "attach rationales.json, the file the checkpoint was trained with"
WHY_BASE, WHY_JSON = _why[0], "rationales_rl.json"'''

MERGE_PY = r'''%%writefile merge.py
"""Merge the new LoRA into the old checkpoint's weights, on CPU."""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import *

model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(LLM_OLD, dtype=DTYPE), "llm_lora").merge_and_unload()
params = sum(p.numel() for p in model.parameters())
assert params < 5e9, f"over the LLM limit: {params:,}"
print(f"LLM parameters: {params:,}")
model.save_pretrained("llm_model")
AutoTokenizer.from_pretrained(LLM_OLD).save_pretrained("llm_model")'''

CELL_PLAN = r'''# 3. Inputs, transcripts, row order
from settings import *

shutil.copy(Path(LLM_OLD).parent / "lifemate.py", "lifemate.py")  # saved next to the new weights too
import lifemate as lm

print("checkpoint:", LLM_OLD, "\ntrain text:", NOISY_ASR, "\npublic text:", ASR_PUBLIC, "\nreasons:", WHY_BASE)
rows, public = lm.load_split(ROOT, "train"), lm.load_split(ROOT, "public_test")
by_id = {x["id"]: x for x in rows}
asr = {k: v for k, v in json.load(open(ASR_PUBLIC, encoding="utf-8")).items() if k.startswith("public_test/")}
asr |= {f"train/{k}": v for k, v in json.load(open(NOISY_ASR, encoding="utf-8")).items()}
for split, part in (("public_test", public), ("train", rows)):
    miss = [c for x in part for c in lm.clips(x) if f"{split}/{c}" not in asr]
    assert not miss, f"{len(miss)} {split} clips have no transcript, e.g. {miss[:3]}: is this the full 01c output?"
json.dump(asr, open(ASR_JSON, "w", encoding="utf-8"), ensure_ascii=False)
whys = json.load(open(WHY_BASE, encoding="utf-8"))
assert all(x["id"] in whys for x in rows) and all(k.startswith("train_") for k in whys), "rationales.json must hold the train ids and nothing else (R-RES-3)"

fit, unseen, seen = lm.split_rows(rows)
held = unseen + seen
gate = [x["id"] for x in held[::5]]  # measured before and after, never trained on in this notebook
rest = [x["id"] for x in held if x["id"] not in set(gate)]
fit_ids = [x["id"] for x in fit]
random.Random(SEED).shuffle(fit_ids)
order = (rest + fit_ids) if SMOKE else gate + rest + fit_ids  # smoke: 16 rows that may be trained on, so training runs too
json.dump(order, open("roll_ids.json", "w"))
json.dump(gate, open("gate_ids.json", "w"))
print(f"rollout order: {len(gate)} gate rows, {len(rest)} other held-out rows, {len(fit_ids)} train-split rows; {ROLL_MIN:.0f} min")


def infer(split, tag, llm, minutes, ids=""):
    """Two infer.py processes, one per T4, until done or `minutes` are over. Returns their lines."""
    env = {**os.environ, "TAG": tag, "LLM_DIR": llm, "STOP_AT": str(time.time() + 60 * minutes), "IDS_JSON": ids, "TOKENIZERS_PARALLELISM": "false"}
    ps = [subprocess.Popen([sys.executable, "-u", "infer.py", split, "asr", str(s)], env={**env, "CUDA_VISIBLE_DEVICES": str(s)},
                           stdout=open(f"infer_{tag}_{s}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT) for s in range(SHARDS)]
    files, t0 = [f"pred_{tag}_{split}_{s}.jsonl" for s in range(SHARDS)], time.time()
    while any(p.poll() is None for p in ps):
        time.sleep(5 if os.environ.get("ZOLA_IN") else 120)
        print(f"  {tag} {split}: {sum(sum(1 for _ in open(f, encoding='utf-8')) for f in files if os.path.exists(f))} samples after {(time.time() - t0) / 60:.0f} min", flush=True)
    for s, p in enumerate(ps):
        if p.returncode:
            print(f"shard {s} exited with {p.returncode}:\n" + open(f"infer_{tag}_{s}.log", encoding="utf-8", errors="replace").read()[-2000:])
    return [json.loads(l) for f in files if os.path.exists(f) for l in open(f, encoding="utf-8")]'''

CELL_K2 = r'''# 4. K2 Horizon on the same train rows, in the background (CPU). Optional: any problem here only prints a note
K2, K2_TRACES, K2_STOP = None, Path("traces_train_k2_why.jsonl").resolve(), Path("K2_STOP").resolve()
if USE_K2:
    try:
        if not os.environ.get("IFM_API_KEY"):
            from kaggle_secrets import UserSecretsClient
            os.environ["IFM_API_KEY"] = UserSecretsClient().get_secret("IFM_API_KEY").strip()
        Z = Path(os.environ.get("ZOLA_ROOT", "/tmp/ZOLA"))  # writable copy in the layout solve.py expects
        solve = next((p for p in Path(IN).rglob("solve.py") if (p.parent / "transcripts_gemini_public_test.json").is_file()), None)
        assert solve and '"--why"' in solve.read_text(encoding="utf-8") and "KEY_AT" in solve.read_text(encoding="utf-8"), "zola_deepseek_bundle with the current solve.py is not attached (unpacked)"
        if not (Z / "api_agent/solve.py").is_file():
            shutil.copytree(solve.parent, Z / "api_agent", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(ROOT / "train", Z / "train+public_test/train", ignore=shutil.ignore_patterns("__pycache__", "audio"))
        K2_STOP.unlink(missing_ok=True)
        k2_ids = (rest + fit_ids)[: 16 if SMOKE else K2_ROWS]  # train rows only (R-RES-3); the gate rows are not trained on, so they need no reasons
        K2 = subprocess.Popen([sys.executable, "-u", "solve.py", "--split", "train", "--backend", "ifm", "--why", "--ids", ",".join(k2_ids),
                               "--workers", str(K2_WORKERS), "--cache", str(K2_TRACES.with_name("cache_train_k2_why.jsonl")), "--traces", str(K2_TRACES),
                               "--stop-file", str(K2_STOP)], cwd=Z / "api_agent", stdout=open("run_k2.log", "a", encoding="utf-8"),
                              stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        print(f"K2 started on {len(k2_ids)} train rows with {len(os.environ['IFM_API_KEY'].replace(',', ' ').split())} key(s); log: run_k2.log")
    except Exception as e:
        print(f"K2 skipped ({type(e).__name__}: {e}); the reasons of rationales.json are used")'''

CELL_ROLL = r'''# 5. Rollouts of the old checkpoint on train rows (reward = official score), then the reward-selected training rows
roll = {x["id"]: x for x in infer("train", "old", LLM_OLD, ROLL_MIN, "roll_ids.json")}
assert roll, "no rollout finished: see infer_old_0.log"
if K2:  # no new rows; the running ones finish and are saved
    K2_STOP.touch()
    try:
        K2.wait(timeout=240)
    except subprocess.TimeoutExpired:
        K2.kill()
    print("K2 log tail:\n" + open("run_k2.log", encoding="utf-8", errors="replace").read()[-800:])
k2 = {}
for l in open(K2_TRACES, encoding="utf-8") if K2_TRACES.is_file() else ():
    try:
        o = json.loads(l)
    except ValueError:  # a line cut by the kill
        continue
    if o["id"].startswith("train_"):
        k2[o["id"]] = o

held_set, gate_set = {x["id"] for x in held}, set(gate)
done = [i for i in order if i in roll and i not in gate_set]
fail = [i for i in done if roll[i]["score"] < 1]
ok_new = [i for i in done if roll[i]["score"] == 1 and i in held_set]  # right, but on rows the checkpoint may never have trained on
ok_old = [i for i in done if roll[i]["score"] == 1 and i not in held_set]
for name, ids in (("gate", gate), ("other held-out", rest), ("train split", fit_ids)):
    r = [roll[i]["score"] for i in ids if i in roll]
    if r:
        print(f"{name:15s} n={len(r):4d} reward {100 * st.mean(r):6.2f}  failed {sum(x < 1 for x in r)}")
sec = st.mean(x["sec"] for x in roll.values())
WHY_ON = sum('"why"' in (x["raw"] or [""])[0] for x in roll.values()) > len(roll) / 2
json.dump({"why": WHY_ON}, open("rl_config.json", "w"))
PUBLIC_MIN = len(public) / SHARDS * sec * 1.15 / 60 + 5  # measured speed of this model on this machine, 15% spare, model load
TRAIN_MIN = 8 if SMOKE else min(TRAIN_CAP, left() - 15 - GATE_MIN - PUBLIC_MIN * PUBLIC_RUN - 5)
rl = (fail * 2 + ok_new + ok_old)[: max(GRAD_ACCUM, int((TRAIN_MIN - 6) * 60 / SEC_PER_CONV))]
json.dump(rl, open("rl_rows.json", "w"))

from why_from_traces import convert
new_whys, why_stats = convert([by_id[i] for i in sorted(set(rl))], k2, whys)
json.dump(whys | new_whys, open(WHY_JSON, "w", encoding="utf-8"), ensure_ascii=False)
k2_ok = sum(o.get("score") == 1.0 for o in k2.values())
print(f"{sec:.1f} s per sample -> a public run takes about {PUBLIC_MIN:.0f} min ({'here' if PUBLIC_RUN else 'in the eval notebook'}), training gets {TRAIN_MIN:.0f} min; {left():.0f} min left")
print(f"checkpoint writes reasons: {WHY_ON}; K2 traces: {len(k2)} rows, {k2_ok} with reward 1; K2 reasons used in {why_stats['replaced']} of {len(set(rl))} training rows")
print(f"training rows: {len(rl)} = {len([i for i in rl if i in set(fail)])} from {len(fail)} failed rollouts (twice) + {len([i for i in rl if i in set(ok_new)])} held-out successes + {len([i for i in rl if i in set(ok_old)])} other successes")'''

CELL_TRAIN = r'''# 6. Fine-tune the old checkpoint on the selected rows, merge. A failure here does not stop the notebook
NEW = None
if TRAIN_MIN >= (0 if SMOKE else 15) and rl:
    env = {**os.environ, "TRAIN_HOURS": str((TRAIN_MIN - 1) / 60), "LLM_DIR": LLM_OLD, "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    rc = run_logged([sys.executable, "-u", "train_llm.py"], "train.log", env)  # exit code 137 means killed, usually out of host RAM
    if rc == 0 and os.path.isdir("llm_lora"):
        rc = run_logged([sys.executable, "-u", "merge.py"], "train.log", env)
    if rc == 0 and os.path.isfile("llm_model/config.json"):
        NEW = "llm_model"
    shutil.rmtree("ckpt", ignore_errors=True)  # only the stop checkpoint, the adapter is in llm_lora
print(f"new model: {NEW}; {left():.0f} min left" if NEW else "NO NEW MODEL (training skipped or failed, see train.log): there is nothing new to evaluate or to put in the Docker")'''

CELL_PUBLIC = r'''# 7. Optional (PUBLIC_RUN): public run with the new model (offline text: 01c transcripts), submission, validation
if PUBLIC_RUN and (NEW or SMOKE):
    pub = infer("public_test", "new", NEW or LLM_OLD, left() - 3)
    assert len({x["id"] for x in pub}) == len(pub)
    assert SMOKE or len(pub) == len(public), f"{len(pub)} lines, expected {len(public)}: see infer_new_*.log"
    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for x in pub:
            f.write(json.dumps({k: x[k] for k in ("id", "tool_calls", "state_hash")}, ensure_ascii=False) + "\n")
    import collections
    print(collections.Counter(x["stop"] for x in pub).most_common(), "|", sum(x["sec"] > 30 for x in pub), "samples over 30 s, slowest", f"{max(x['sec'] for x in pub):.1f} s")
    print(f"{len(pub)} lines, {sum(x['failed'] for x in pub)} failed, {sum(x['invalid'] for x in pub)} stopped on an invalid call, model: {NEW or 'OLD checkpoint'}")
    r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", os.path.abspath("submission.jsonl")],
                       cwd=ROOT / "public_test", capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"})
    print(r.stdout[-1500:], r.stderr[-1500:])
else:
    print('no public run here: attach this output (llm_model/, lifemate.py) to the eval notebook')'''

CELL_GATE = r'''# 8. Gate: new model against the old model's rollouts on the same held-out rows. The only clean comparison: the other held-out rows were trained on
summary = {"old_model": LLM_OLD, "new_model": NEW, "rollouts": len(roll), "failed_rollouts": len(fail), "training_rows": len(rl),
           "k2_rows": len(k2), "k2_reasons_used": why_stats["replaced"], "reasons": WHY_ON, "train_text": os.path.basename(NOISY_ASR)}
if NEW and (SMOKE or left() > 8):
    new = {x["id"]: x for x in infer("train", "gate", NEW, 20 if SMOKE else min(25, left() - 2), "gate_ids.json")}
    both = [i for i in gate if i in new and i in roll]
    if both:
        o, n = [roll[i]["score"] for i in both], [new[i]["score"] for i in both]
        summary |= {"gate_rows": len(both), "gate_old": 100 * st.mean(o), "gate_new": 100 * st.mean(n),
                    "gate_better": sum(a > b for a, b in zip(n, o)), "gate_worse": sum(a < b for a, b in zip(n, o))}
        print(f"gate, {len(both)} held-out rows: old {summary['gate_old']:.2f} -> new {summary['gate_new']:.2f}, better on {summary['gate_better']}, worse on {summary['gate_worse']}")
        print("With this few rows only a clear gap means something. If the old checkpoint was trained with FINAL = True it has seen these rows, which favours neither model.")
else:
    print("gate skipped:", "no new model" if not NEW else "no time left")
json.dump(summary, open("rl_summary.json", "w"), indent=1)
print(summary, f"\n{left():.0f} min before {DEADLINE_VN}; files: llm_model/, lifemate.py, rl_summary.json" + (", submission.jsonl" if PUBLIC_RUN else ""))'''

EVAL_MD = """# Eval of the `qwen_k2_rl` output (built by `student/make_rl_notebook.py` from `neweval_sft`)

Changes against `neweval_sft`:
1. **Model** = the attached `qwen_k2_rl` output (`llm_model/`, `lifemate.py`, `rl_summary.json`); it is taken first even if
   an older checkpoint is attached too.
2. **EVAL runs the 66 gate rows only** (every 5th held-out row). `qwen_k2_rl` trained on the other 262 held-out rows, so a
   score on them would be inflated. About 10 min instead of 45.
3. The report compares the new model with the **old checkpoint's rollouts on the same gate rows** (from the attached
   output) and prints `rl_summary.json`.

The public run, `submission.jsonl` and the validator are unchanged.

4. The held-out train text comes from the output's own `asr_all.json` (the stock-ASR text the old checkpoint's rollouts
   used), so no `raw_asr_train.json` is needed and the 01c `FINAL = True` output no longer stops the EVAL guard.

**Attach**: the data package, the `qwen_k2_rl` notebook output, the 01c output (`asr_transcripts.json`).
`EVAL = False` skips the gate rows and runs public_test only.
"""

EVAL_GATE = r'''            report(f"{mode} {stop}", [x for x in p if x["stop"] == stop])  # repaired: was invalid in 03c
    old = {x["id"]: x["score"] for f in glob.glob(str(RL_DIR / "pred_old_train_*.jsonl")) for x in map(json.loads, open(f, encoding="utf-8"))}
    new = {x["id"]: x["score"] for x in load("train", "asr")}
    both = sorted(set(old) & set(new))
    if both:  # same rows, same stock-ASR text, the only held-out rows the new model never trained on
        print(f"\ngate, {len(both)} rows: old checkpoint {100 * st.mean(old[i] for i in both):.2f} -> new model {100 * st.mean(new[i] for i in both):.2f}, better on {sum(new[i] > old[i] for i in both)}, worse on {sum(new[i] < old[i] for i in both)}")
    else:
        print("\nno gate row in common with the old checkpoint's rollouts (pred_old_train_*.jsonl in the attached output)")
    print("qwen_k2_rl's own numbers:", json.load(open(RL_DIR / "rl_summary.json", encoding="utf-8")))'''

if __name__ == "__main__":
    def code(src):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)}]
    cells += [code(c) for c in (CELL_ENV, CELL_SETTINGS, SETTINGS_PY, INFER, TRAIN, MERGE_PY, WHY, CELL_PLAN, CELL_K2, CELL_ROLL,
                                CELL_TRAIN, CELL_PUBLIC, CELL_GATE)]
    out = NB / "qwen_k2_rl.ipynb"
    json.dump({"cells": cells, "nbformat": 4, "nbformat_minor": 5,
               "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)

    # neweval_rl.ipynb: the eval notebook (neweval_sft) for this notebook's output. neweval_sft itself stays as it is: eval_vast.py reads it
    ev = json.load(open(NB / "neweval_sft.ipynb", encoding="utf-8"))

    def repl(start, *pairs):
        (c,) = [c for c in ev["cells"] if "".join(c["source"]).startswith(start)]
        c["source"] = (patch("".join(c["source"]), *pairs) if pairs else EVAL_MD).splitlines(keepends=True)

    repl("# SFT version of 03")
    repl("%%writefile settings.py",
         ('LLM = str(Path(glob.glob("/kaggle/input/**/llm_model/config.json", recursive=True)[0]).parent)\n',
          '_llm = sorted(glob.glob("/kaggle/input/**/llm_model/config.json", recursive=True), key=lambda p: not (Path(p).parents[1] / "rl_summary.json").is_file())  # the qwen_k2_rl output first, if an older checkpoint is attached too\n'
          'LLM = str(Path(_llm[0]).parent)\n'
          'RL_DIR = Path(LLM).parent  # rl_summary.json and the old checkpoint\'s rollouts pred_old_train_*.jsonl\n'
          'assert (RL_DIR / "rl_summary.json").is_file(), "attach the output of the qwen_k2_rl notebook (llm_model/, lifemate.py, rl_summary.json)"\n'))
    repl("import glob\nimport json\nimport re\nimport sys\nimport unicodedata\n\nimport jiwer",
         ('need = {s: [f"{s}/{c}" for x in rows for c in lm.clips(x)] for s, rows in\n',
          'if (RL_DIR / "asr_all.json").is_file():  # the train text qwen_k2_rl\'s rollouts used (stock ASR, never heard these clips): the gate rows compare like for like, and no raw_asr_train.json is needed\n'
          '    raw |= {k: v for k, v in json.load(open(RL_DIR / "asr_all.json", encoding="utf-8")).items() if k.startswith("train/")}\n'
          'need = {s: [f"{s}/{c}" for x in rows for c in lm.clips(x)] for s, rows in\n'))
    repl("%%writefile infer.py",
         ("    rows = sum(lm.split_rows(rows)[1:], [])  # only held-out samples say anything\n",
          "    rows = sum(lm.split_rows(rows)[1:], [])[::5]  # qwen_k2_rl's 66 gate rows: it trained on the other held-out rows\n"))
    repl("import collections\nimport json\nimport statistics as st",
         ('            report(f"{mode} {stop}", [x for x in p if x["stop"] == stop])  # repaired: was invalid in 03c', EVAL_GATE))
    json.dump(ev, open(NB / "neweval_rl.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", NB / "neweval_rl.ipynb")
