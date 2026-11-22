"""Build the SFT versions of the teammate's three Kaggle notebooks (student/SFT_PLAN.md) from the executed originals.

Each patch is an exact string that must occur once in its cell, so a changed original stops the build instead of being
patched in the wrong place. Outputs are cleared. The originals are only read.
  01 thbased01: digit clips train the ASR (vi_verbalise.py), cross-fitted train transcripts asr_xfit_train.json (A1, A2)
  02 bowen:     asr_xfit_train.json as the noisy text, reasons only on final-turn targets, RESUME off (B1, B2, B4)
  03 neweval:   offline run without raw files, asr_xfit_train.json for EVAL, guard against inflated EVAL (C2, C3)

Usage: python student/make_sft_notebooks.py [folder with the originals]
"""
import copy
import json
import sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Study\nlp\jigsaw dataset")
OUT = Path(__file__).resolve().parent / "notebooks"


def src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    return c | ({"execution_count": None, "outputs": []} if kind == "code" else {})


class Notebook:
    def __init__(self, name):
        self.nb = json.load(open(SRC / name, encoding="utf-8"))
        self.cells = self.nb["cells"]

    def at(self, cid):
        i = next((i for i, c in enumerate(self.cells) if c.get("id") == cid), None)
        assert i is not None, f"no cell {cid}"
        return i

    def patch(self, cid, old, new):
        c = self.cells[self.at(cid)]
        s = src(c)
        assert s.count(old) == 1, f"cell {cid}: {s.count(old)} matches for {old[:60]!r}"
        c["source"] = s.replace(old, new).splitlines(keepends=True)

    def insert(self, cid, *new, before=False):
        i = self.at(cid) + (0 if before else 1)
        self.cells[i:i] = [cell(k, t) for k, t in new]

    def save(self, name):
        nb = copy.deepcopy(self.nb)
        for c in nb["cells"]:
            c.pop("id", None)  # new cells have none; nbformat 4.5 wants all or none, Jupyter adds fresh ones
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        nb["nbformat_minor"] = min(nb.get("nbformat_minor", 4), 4)
        OUT.mkdir(exist_ok=True)
        json.dump(nb, open(OUT / name, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
        print("wrote", OUT / name, len(nb["cells"]), "cells")


# ---------------------------------------------------------------- 01 thbased01 (ASR)
VERBALISE_PY = r'''%%writefile vi_verbalise.py
"""Spoken Vietnamese for the digits of a normalised transcript, so clips whose gold text has digits can train the ASR:
"đặt báo thức 7 giờ 30" -> "đặt báo thức bảy giờ ba mươi". The inverse of vi_itn.py, checked against it below.
A token mixing letters and digits (vtv6, rv245, 24h) gives None: how it was spoken is not known, so the clip is skipped.
Self-check: python vi_verbalise.py"""
import random

from vi_itn import digits

WORD = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALES = [(10**9, "tỷ"), (10**6, "triệu"), (1000, "nghìn")]  # train gold says nghìn 125 times, ngàn 20


def below_100(n):
    t, u = divmod(n, 10)
    if not t:
        return WORD[u]
    head = "mười" if t == 1 else f"{WORD[t]} mươi"
    if not u:
        return head
    return f"{head} " + ("lăm" if u == 5 else "mốt" if u == 1 and t > 1 else "tư" if u == 4 and t > 1 else WORD[u])


def below_1000(n, full):
    """full: a group after a higher scale keeps its hundreds, as in 'hai nghìn không trăm linh năm'."""
    h, r = divmod(n, 100)
    if not h and not full:
        return below_100(r)
    return f"{WORD[h]} trăm" + (f" linh {WORD[r]}" if 0 < r < 10 else f" {below_100(r)}" if r else "")


def cardinal(n, full=False):
    for s, name in SCALES:
        if n >= s:
            q, r = divmod(n, s)
            return f"{cardinal(q, full)} {name}" + (f" {cardinal(r, True)}" if r else "")
    return below_1000(n, full)


def by_digit(s):
    """Phone numbers, tickets, codes: a leading zero, or 6+ digits that are not a round amount."""
    return len(s) > 1 and s[0] == "0" or len(s) >= 6 and int(s) % 1000 != 0


def spoken(s):
    return " ".join(WORD[int(c)] for c in s) if by_digit(s) else cardinal(int(s))


def verbalise(text):
    out = []
    for w in text.split():
        if w.isascii() and w.isdigit():
            out.append(spoken(w))
        elif any(c.isdigit() for c in w):
            return None
        else:
            out.append(w)
    return " ".join(out)


if __name__ == "__main__":
    for a, b in {
        "đặt báo thức 7 giờ sáng mai": "đặt báo thức bảy giờ sáng mai",
        "dưới 119 triệu thì nhắc chị 9 giờ": "dưới một trăm mười chín triệu thì nhắc chị chín giờ",
        "uống thuốc lúc 9 giờ kém 15": "uống thuốc lúc chín giờ kém mười lăm",
        "bật điều hòa 24 độ": "bật điều hòa hai mươi tư độ",
        "21 25 14 10 105 2026": "hai mươi mốt hai mươi lăm mười bốn mười một trăm linh năm hai nghìn không trăm hai mươi sáu",
        "chuyển 2500000 và 2005000": "chuyển hai triệu năm trăm nghìn và hai triệu không trăm linh năm nghìn",
        "vé số 412750": "vé số bốn một hai bảy năm không",
        "gọi 0912345678": "gọi không chín một hai ba bốn năm sáu bảy tám",
        "mở kênh vtv6": None,
        "chuyến rv245": None,
        "không có số": "không có số",
        "": "",
    }.items():
        assert verbalise(a) == b, (a, verbalise(a), b)
    rng = random.Random(0)
    for n in list(range(1, 100000)) + [rng.randrange(10**10) for _ in range(20000)]:
        s, w = str(n), spoken(str(n))
        got, want = (digits(w), s) if by_digit(s) else (digits(w + " đồng"), s + " đồng")  # a unit makes vi_itn read a lone digit
        assert got == want, (s, w, got)
    print("vi_verbalise self-check ok: vi_itn reads every spelled number back")
'''

XFIT_PY = r'''%%writefile asr_xfit.py
"""Cross-fitted transcripts of the train clips for 02: every clip is transcribed by the fold model that never heard it, so the
LLM learns from errors like the ones the fine-tuned model makes at test time (the stock model's are a different kind).
Same keys and ITN as asr_base_train.json. Folds are by train row, so a row's history and final clips share one."""
from asr_common import *

rows = lm.load_split(ROOT, "train")
truth = clip_texts(rows)
out = {}
for f in range(1 if SMOKE else XFOLDS):
    keys = list(clip_texts([x for i, x in enumerate(rows) if i % XFOLDS == f]))[: SMOKE_CLIPS if SMOKE else None]
    hyp = transcribe(keys, f"{XFIT_DIR}/fold{f}")
    print(f"fold {f}: {len(keys)} clips, WER {wer(keys, truth, hyp):.4f}, after ITN on both sides {wer(keys, truth, hyp, itn):.4f}", flush=True)
    out |= hyp
assert SMOKE or out.keys() == truth.keys(), "some train clips have no cross-fitted transcript"
val = [k for k in clip_texts(sum(lm.split_rows(rows)[1:], [])) if k in out]
print(f"{len(out)} clips, WER after ITN {wer(list(out), truth, out, itn):.4f}; on lifemate's {len(val)} held-out clips {wer(val, truth, out, itn):.4f} (stock 0.1609)")
fix = itn if ITN else str
json.dump({k.split("/", 1)[1]: fix(v) for k, v in out.items()}, open("asr_xfit_train.json", "w", encoding="utf-8"), ensure_ascii=False)
'''

XFIT_RUN = r'''import shutil

from settings import *

if XFIT:  # SMOKE: fold 0 only, 1 epoch, 200 clips, to prove the path
    for f in range(1 if SMOKE else XFOLDS):
        d = f"{XFIT_DIR}/fold{f}"
        !python -u train_asr.py {f}
        assert _exit_code == 0, f"fold {f} training failed, see the log above"
        !python ice.py export-onnx.py --tokens {BASE_DIR}/tokens.txt --epoch 99 --avg 1 --use-averaged-model 0 --exp-dir {d}
        assert _exit_code == 0, f"fold {f} export failed, see the log above"
        for part in ("encoder", "decoder", "joiner"):
            shutil.move(f"{d}/{part}-epoch-99-avg-1.int8.onnx", f"{d}/{part}.int8.onnx")
        shutil.copy(f"{BASE_DIR}/tokens.txt", d)
        for p in [*Path(d).glob("*-avg-1.onnx"), Path(d, "epoch-99.pt")]:  # /tmp, but a fold is ~600 MB
            p.unlink()
    !python asr_xfit.py
    assert _exit_code == 0, "this step failed, see the log above"
'''


def build_01():
    nb = Notebook("thbased01.ipynb")
    nb.insert("be2ada04", ("markdown", """# SFT version of 01c (built by `student/make_sft_notebooks.py`, plan in `student/SFT_PLAN.md`)

Two changes, both on by default, otherwise the notebook below is 01c unchanged:
1. **Digit clips train too** (`VERBALISE`): `vi_verbalise.py` spells the digits of the gold text the way `vi_itn.py` reads
   them back (checked over every number up to 100,000 and 20,000 random ones). This replaces the bullet below on the
   481 left-out clips: only clips with a letter+digit token (`vtv6`, `rv245`, `24h`) are still left out.
2. **Cross-fitted train transcripts** (`XFIT`): 5 more models, each trained without one fifth of the train rows for the
   best epoch of this run, transcribe the rows they never heard. `asr_xfit_train.json` is the noisy text 02 now trains on,
   and the held-out train text 03 scores on. ~15 min per fold on one T4 (estimate).

Run with `FINAL = False` (the default here) first: it prints the held-out WER per epoch and writes all files 02 and 03
need. `FINAL = True` afterwards gives the Docker's `asr_model/`; set `XFIT = False` for it, the folds would be the same.
"""), before=True)
    nb.patch("302210f2", "FINAL = True  # True: fine-tune on every train clip, nothing held out\nEPOCHS = 8  # FINAL: set this to the best epoch that the FINAL = False run printed\n",
             "FINAL = False  # True: fine-tune on every train clip, nothing held out\nEPOCHS = 8  # FINAL: set this to the best epoch that the FINAL = False run printed. Folds use that best epoch when it exists\n")
    nb.patch("302210f2", "SEED = 42\n", """SEED = 42
VERBALISE = True  # True: clips whose gold text has digits train too, spelled out by vi_verbalise.py. False: skipped, as in 01c
XFIT = True  # True: also train XFOLDS models, each without one fold of train rows, and transcribe that fold: asr_xfit_train.json
XFOLDS = 5
""")
    nb.patch("302210f2", 'ASR_DIR = "asr_model"  # the fine-tuned int8 ONNX, kept in the output for 03 and the Docker\n',
             'ASR_DIR = "asr_model"  # the fine-tuned int8 ONNX, kept in the output for 03 and the Docker\nXFIT_DIR = "/tmp/xfit"  # fold models, not kept in the output\n')
    nb.insert("97c15f55",
              ("markdown", "Digits of the gold text spelled out for training, the inverse of `vi_itn.py`. Its self-check reads every spelled number back through `vi_itn.py`."),
              ("code", VERBALISE_PY),
              ("code", '!python vi_verbalise.py\nassert _exit_code == 0, "this step failed, see the log above"'))
    t = "ee4e575d"
    nb.patch(t, "import argparse\nimport math\nimport random\nimport time\n", "import argparse\nimport math\nimport random\nimport sys\nimport time\n")
    nb.patch(t, "from asr_common import *\n", "from asr_common import *\nfrom vi_verbalise import verbalise\n\nFOLD = int(sys.argv[1]) if len(sys.argv) > 1 else None  # cross-fit: train without the rows i % XFOLDS == FOLD\n")
    nb.patch(t, "fit, *held = (rows, [], []) if FINAL else lm.split_rows(rows)\n",
             "if FOLD is None:\n    fit, *held = (rows, [], []) if FINAL else lm.split_rows(rows)\nelse:\n    fit, held = [x for i, x in enumerate(rows) if i % XFOLDS != FOLD], []\n")
    nb.patch(t, """    y = sp.encode(norm(text).upper())  # the model's own style: uppercase, old tone spelling, numbers as words
    if re.search(r"\\d", text) or unk in y:  # ponytail: no number verbaliser, so clips with digits are left out; ITN handles numbers after ASR
        skipped += 1""", """    spoken = verbalise(norm(text)) if VERBALISE else None if re.search(r"\\d", text) else norm(text)  # None: digits of unknown spoken form
    y = sp.encode(spoken.upper()) if spoken else [unk]  # the model's own style: uppercase, old tone spelling, numbers as words
    if unk in y:
        skipped += 1""")
    nb.patch(t, 'print(f"{len(data)} training clips, {sum(len(feats[k]) for k, _ in data) / 360000:.1f} h, {skipped} skipped for digits or unknown tokens, {len(held_keys)} held-out clips")',
             'spelled = sum(bool(re.search(r"\\d", truth[k])) for k, _ in data)\nprint(f"{len(data)} training clips ({spelled} with digits spelled out), {sum(len(feats[k]) for k, _ in data) / 360000:.1f} h, {skipped} skipped for unknown digits or tokens, {len(held_keys)} held-out clips, fold {FOLD}")')
    nb.patch(t, "for epoch in range(1, (1 if SMOKE else EPOCHS) + 1):",
             'epochs = int(Path("exp/best_epoch").read_text()) if FOLD is not None and Path("exp/best_epoch").exists() else EPOCHS  # the main run\'s best\nfor epoch in range(1, (1 if SMOKE else epochs) + 1):')
    nb.patch(t, 'Path("exp").mkdir(exist_ok=True)\ntorch.save({"model": best[2] if best else model.state_dict()}, "exp/epoch-99.pt")',
             'out = Path("exp" if FOLD is None else f"{XFIT_DIR}/fold{FOLD}")\nout.mkdir(parents=True, exist_ok=True)\ntorch.save({"model": best[2] if best else model.state_dict()}, out / "epoch-99.pt")\nif best:\n    Path("exp/best_epoch").write_text(str(best[1]))')
    nb.insert("bf61bf1c",
              ("markdown", "Cross-fitted train transcripts for 02 and 03 (`XFIT`): one model per fold of train rows, each transcribing only the fold it never heard."),
              ("code", XFIT_PY),
              ("code", XFIT_RUN))
    nb.save("thbased01_sft.ipynb")


# ---------------------------------------------------------------- 02 bowen (LLM)
def build_02():
    nb = Notebook("bowen.ipynb")
    nb.insert("2455317a", ("markdown", """# SFT version of 02 bowen (built by `student/make_sft_notebooks.py`, plan in `student/SFT_PLAN.md`)

Changes against bowen as saved, otherwise unchanged:
1. **Noisy text** = `asr_xfit_train.json` from `thbased01_sft` when attached (errors like the fine-tuned ASR's at test time),
   else `asr_base_train.json` (stock model, as before). The training log names the file used.
2. **Reasons only on final-turn targets** (`WHY_HISTORY = False`): bowen gave history calls their reason too, but inference
   renders history without reasons, so every final turn was learnt after a history that test never shows.
   `WHY_HISTORY = True` reproduces bowen.
3. `RESUME = False`: a checkpoint from another data or prompt version must not be continued.

Attach the output of `thbased01_sft` (`FINAL = False`) instead of 01c. Compare against bowen with `neweval_sft`, EVAL on.
""")
              , before=True)
    nb.patch("13dfbadb", "RESUME = True  # True: continue from the newest checkpoint in an attached output of this notebook",
             "RESUME = False  # True: continue from the newest checkpoint in an attached output of this notebook, only one with the same data and prompt")
    nb.patch("13dfbadb", "WHY = True  # True: every learnt step starts with a short written reason, from rationales.json (made by rationales/gen_rationales.py) attached as an input\n",
             "WHY = True  # True: every learnt step starts with a short written reason, from rationales.json (made by rationales/gen_rationales.py) attached as an input\n"
             "WHY_HISTORY = False  # True: history calls carry their reason too, as bowen. Inference renders history without reasons, so False matches it\n")
    nb.patch("13dfbadb", '_noisy = glob.glob("/kaggle/input/**/asr_base_train.json", recursive=True)',
             '_noisy = glob.glob("/kaggle/input/**/asr_xfit_train.json", recursive=True) or glob.glob("/kaggle/input/**/asr_base_train.json", recursive=True)  # cross-fitted 01c text first')
    t = "566cbccf"
    nb.patch(t, """        for i, why in zip(sorted(labelled), w):
            msgs[i] = lm.asst({"why": why} | json.loads(msgs[i]["content"]))
""", """        for i, why in zip(sorted(labelled), w):
            if WHY_HISTORY or i >= first:  # history calls stay as inference renders them, without a reason
                msgs[i] = lm.asst({"why": why} | json.loads(msgs[i]["content"]))
        assert WHY_HISTORY or not any('"why"' in m["content"] for m in msgs[:first]), x["id"]
""")
    nb.patch(t, "ASR noise: {bool(noisy)}", "ASR noise: {os.path.basename(NOISY_ASR)}, history reasons: {WHY_HISTORY}")
    nb.save("bowen_sft.ipynb")


# ---------------------------------------------------------------- 03 neweval (eval)
def build_03():
    nb = Notebook("neweval.ipynb")
    nb.insert("f884ade4", ("markdown", """# SFT version of 03 neweval (built by `student/make_sft_notebooks.py`, plan in `student/SFT_PLAN.md`)

`infer.py` is unchanged, so scores compare with neweval's. Changes in the transcript cell:
1. **Offline run**: with no `raw_asr*.json` attached, every clip comes from the attached 01c output. That is what the Docker
   can score (no API ASR at inference, R-RUN-1). Attach raw files only to measure another ASR.
2. **EVAL on by default**, with the held-out train clips from `asr_xfit_train.json` (attach the `thbased01_sft` output), whose
   models never heard them. Raw files still win where they have a clip.
3. **Guard**: EVAL stops when the 01c text gives a held-out WER under 0.05, which means that ASR trained on those clips
   (01c FINAL: 0.0194, stock 0.1609), so the score would be inflated.

Attach the LLM output to test and say which one in the results: neweval tested `qwenupdatefabe`, not `bowen`.
""")
              , before=True)
    nb.patch("4295620a", "EVAL = False  # needs the held-out train clips in the raw files. False: public_test only",
             "EVAL = True  # held-out train rows, clips from the raw files, else asr_xfit_train.json, else 01c (only if FINAL = False). False: public_test only")
    nb.patch("4295620a", "TIME_BUDGET = 120.0  # a file upload is not timed. The Docker must stay under the real limit of 30 s",
             "TIME_BUDGET = 120.0  # a file upload is not timed. The Docker must stay under the real limit of 30 s: 25 shows what that limit costs")
    nb.patch("4295620a", "  # the other ASR model's output, merged", "  # the other ASR model's output, merged. None: offline run on the 01c output only")
    t = "0288cac0"
    nb.patch(t, 'assert RAW_JSON, "attach a dataset with raw_asr*.json"\n', "")
    nb.patch(t, 'print("01c asr_transcripts.json attached:", base)\n',
             'print("01c asr_transcripts.json attached:", base)\nassert RAW_JSON or base, "attach raw_asr*.json, or a 01c output for an offline run"\nprint("offline run: every clip from 01c" if not RAW_JSON else f"{len(new)} clips from the raw files")\n')
    nb.patch(t, '    raw = json.load(open(base[0], encoding="utf-8")) | raw\n',
             '    raw = json.load(open(base[0], encoding="utf-8")) | raw\n'
             'xfit = glob.glob("/kaggle/input/**/asr_xfit_train.json", recursive=True)\n'
             'print("cross-fitted train transcripts:", xfit)\n'
             'if xfit:  # from fold models that never heard these clips, so the held-out score is not inflated. The raw files still win\n'
             '    raw |= {f"train/{k}": v for k, v in json.load(open(xfit[0], encoding="utf-8")).items() if f"train/{k}" not in new}\n')
    nb.patch(t, """    print(f"held-out WER {jiwer.wer([itn(norm(truth[k])) for k in keys], [asr[k] for k in keys]):.4f} on {len(keys)} clips (01b stock 0.1609, 01c FINAL heard them)")""",
             """    w = jiwer.wer([itn(norm(truth[k])) for k in keys], [asr[k] for k in keys])
    print(f"held-out WER {w:.4f} on {len(keys)} clips (01b stock 0.1609, 01c FINAL heard them: 0.0194)")
    assert w > 0.05 or any(k in new for k in keys), "this ASR trained on the held-out clips, the score would be inflated: attach asr_xfit_train.json or a FINAL = False 01c output\"""")
    nb.save("neweval_sft.ipynb")


if __name__ == "__main__":
    build_01()
    build_02()
    build_03()
