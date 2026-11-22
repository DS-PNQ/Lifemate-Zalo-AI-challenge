"""The eval notebook (notebooks/neweval_sft.ipynb) as one command for a rented Linux box with 1 GPU (vast.ai, RTX 5090).

On the box (image: PyTorch with a CUDA 12.8 build of torch >= 2.7, which a 5090 needs; ~40 GB disk):
  export KAGGLE_API_TOKEN=...            # or put kaggle.json in ~/.kaggle/
  python eval_vast_onefile.py            # the one file to upload: written here by  python student/eval_vast.py --pack
It installs the packages, downloads the inputs from Kaggle (SOURCES below, kept in --input, so a rerun downloads
nothing), runs held-out train + public_test and writes eval_out/submission.jsonl (validated), pred_*.jsonl, log_*.txt.

  --llm owner/notebook[/versions/N]   another model (its output must hold llm_model/ with lifemate.py beside it)
  --no-eval                           public_test only
  --no-pull                           no Kaggle: everything is already somewhere under --input (found by name)
  --shards N                          processes on the one GPU (4B fp16 is ~9-11 GB each; 2 fit 32 GB, 3 is untested)

The notebook's cells run as they are: infer.py is byte-identical, so scores compare with the Kaggle runs. Changed
are only the input paths and SMOKE / EVAL / GOLD / SHARDS. Added: the score on the gate rows of qwen_k2_rl.

Rules: eval only, not the Docker entry point. The network is used for pip and the Kaggle download, never by the
agent (R-RUN-1). Seeds and greedy decoding are infer.py's (R-REP-1/2). Times here are not the A30 latency (R-RUN-2).
The data package on a rented machine must stay private: destroy the instance afterwards (R-RES-4). The Kaggle token
is read by kagglehub from the environment and never printed or written (CLAUDE.md ground rule 5).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

PACKED = ["# SFT version of 03 neweval (built by `student/make_sft_notebooks.py`, plan in `student/SFT_PLAN.md`)\n\n`infer.py` is unchanged, so scores compare with neweval's. Changes in the transcript cell:\n1. **Offline run**: with no `raw_asr*.json` attached, every clip comes from the attached 01c output. That is what the Docker\n   can score (no API ASR at inference, R-RUN-1). Attach raw files only to measure another ASR.\n2. **EVAL on by default**, with the held-out train clips from `asr_xfit_train.json` (attach the `thbased01_sft` output), whose\n   models never heard them. Raw files still win where they have a clip.\n3. **Guard**: EVAL stops when the 01c text gives a held-out WER under 0.05, which means that ASR trained on those clips\n   (01c FINAL: 0.0194, stock 0.1609), so the score would be inflated.\n\nAttach the LLM output to test and say which one in the results: neweval tested `qwenupdatefabe`, not `bowen`.\n", '# 03 ASR compare · Qwen on another model\'s transcripts\n\n`adahome/03c_eval_repair.ipynb` with one change: the ASR text comes from a raw file written by any other ASR model, and this notebook converts it the way 01c does (`norm()`, then `vi_itn.py`) before the agent sees it. `infer.py` is byte-identical to 03c repair, so with the same LLM the score change is the ASR. See `instructions.md` beside this notebook.\n\n**Attach** (Kaggle, GPU T4 x2, internet on):\n1. the data package (private dataset),\n2. the LLM output to test, normally `png0806/bowen` (`llm_model/`, `lifemate.py`),\n3. one private dataset holding the transcripts as `raw_asr*.json`, for example `raw_asr_public.json` and `raw_asr_train.json`. All matches are merged.\n\n`vi_itn.py` is written by a cell below, so no dataset needs it. Attach a 01c output only with `FILL_FROM_01C = True`: clips missing from the raw files (for example history turns) then come from its `asr_transcripts.json`.\n\n**Raw file format**: one JSON object `{key: text}`, key = `<split>/<audio path from <split>.jsonl>`, for example `"public_test/audio/public_0002/public_0002.wav"`. One key per final request and per user turn in `history` (1,933 for public_test). Text can have capitals, punctuation and digits. A failed clip is `""`. The Gemini cache `gemini_asr_public_test.json` already has this format: rename it to `raw_asr_public.json`.\n\n**Settings**: `EVAL = True` needs the 1,089 clips of the 328 held-out train samples in the raw files, and the cell stops if any is missing. Set it `False` if you transcribed public_test only. `GOLD = False` skips the gold-text run (known: 71.77 for `bowen`).\n\n**Rules**: these transcripts are for measurement only. Never train or tune on public_test transcripts (labelling test data). No API ASR except Gemini under rule 6\'s second exception, and never in the Docker.\n\n**Output**: `asr_transcripts.json` (converted), `pred_*.jsonl`, `submission.jsonl`, held-out WER and score next to the `bowen` + 01c baseline (public 60.29).\n', 'import glob\nimport json\n\nimport torch\nassert torch.cuda.device_count() == 2, "Set Settings > Accelerator to GPU T4 x2"\nprint(torch.__version__, [torch.cuda.get_device_name(i) for i in range(2)])\n_config = json.load(open(glob.glob("/kaggle/input/**/llm_model/config.json", recursive=True)[0], encoding="utf-8"))\nTRANSFORMERS = {"spark2_5": "4.57.1", "lfm3": "5.20.2"}.get(_config["model_type"], "5.22.1")  # the version each model was trained on\nprint(_config["model_type"], "transformers", TRANSFORMERS)\n!pip install -q "transformers=={TRANSFORMERS}" "accelerate==1.15.0" "jiwer==4.0.0"\n', '%%writefile settings.py\n\nimport glob\nfrom pathlib import Path\n\nSMOKE = False  # True: 8 samples per GPU\nEVAL = True  # held-out train rows, clips from the raw files, else asr_xfit_train.json, else 01c (only if FINAL = False). False: public_test only\nFILL_FROM_01C = True  # also on by itself when an asr_transcripts.json is attached. True: clips missing from raw_asr*.json come from an attached 01c asr_transcripts.json (e.g. history turns), so only the rest is the new ASR\nGOLD = False  # also run gold text on the held-out samples (bowen: 71.77, already known)\nDTYPE = "float16"\nSHARDS = 2  # one process per T4\nMAX_STEPS = 8  # gold never uses more than 6 calls\nMAX_NEW_TOKENS = 320  # the longest gold call is 304 characters, plus a written reason when the model was trained with WHY\nTIME_BUDGET = 120.0  # a file upload is not timed. The Docker must stay under the real limit of 30 s: 25 shows what that limit costs\nSTRIP_MARKS = True  # remove the tagger\'s [ ] from ASR text. False: the old behaviour\nSEED = 42\n\nROOT = Path(glob.glob("/kaggle/input/**/train/train.jsonl", recursive=True)[0]).parents[1]\nLLM = str(Path(glob.glob("/kaggle/input/**/llm_model/config.json", recursive=True)[0]).parent)\nLIFEMATE_DIR = str(Path(LLM).parent)  # lifemate.py as saved with these weights\nRAW_JSON = sorted(glob.glob("/kaggle/input/**/raw_asr*.json", recursive=True))  # the other ASR model\'s output, merged. None: offline run on the 01c output only\nASR_JSON = "asr_transcripts.json"  # written by the next cell, in 01c\'s style', '%%writefile vi_itn.py\n"""Inverse text normalisation for lowercase Vietnamese ASR output: spoken numbers become digits, everything else stays.\n"chuyển hai trăm nghìn lúc chín rưỡi" -> "chuyển 200000 lúc 9 rưỡi". Text inside [entity marks] is left alone.\nRules come from train transcripts only. Self-check: python vi_itn.py"""\nimport re\n\nDIGIT = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}\nONES = DIGIT | {"mốt": 1, "tư": 4, "lăm": 5}  # the last three only mean a number after another number\nSCALE = {"tỷ": 10**9, "tỉ": 10**9, "triệu": 10**6, "nghìn": 1000, "ngàn": 1000}\nBEFORE = {"tháng", "mùng", "mồng", "tầng", "lầu", "quận", "kênh", "tập", "mức", "kém", "số", "ngày"}  # a lone digit after these is a number\nUNITS = {"giờ", "tiếng", "phút", "giây", "độ", "phần", "chỗ", "vé", "mét", "ký", "kg", "km", "lít", "đô", "đồng", "rưỡi", "ngày", "tuần", "tháng"}  # and before these\nYEAR = {"nay", "sau", "ngoái", "trước"}  # "năm" as year\n\n\ndef at(w, i):\n    return w[i] if i < len(w) else ""\n\n\ndef value(a):\n    return DIGIT[a] if a in DIGIT else int(a) if a.isdigit() else None\n\n\ndef tens(w, i):\n    """mười lăm, hai mươi mốt, năm chục, and the spoken short forms hai mốt, bốn lăm, hai sáu độ, hôm hai tám."""\n    a, b = at(w, i), at(w, i + 1)\n    if a == "mười":\n        v, j = 10, i + 1\n    elif DIGIT.get(a) and b in ("mươi", "chục"):\n        v, j = 10 * DIGIT[a], i + 2\n    elif DIGIT.get(a, 0) >= 2 and (b in ("mốt", "tư", "lăm") or (b in DIGIT and (at(w, i + 2) in ("độ", "phần") or (i and w[i - 1] in ("ngày", "hôm", "mùng", "mồng"))))):\n        return 10 * DIGIT[a] + ONES[b], i + 2\n    else:\n        return None\n    c = at(w, j)  # "năm" here is a year: mười năm, hai mươi năm. Five after a ten is spoken "lăm".\n    return (v + ONES[c], j + 1) if c in ONES and c not in ("không", "năm") else (v, j)\n\n\ndef tail(w, j, scale):\n    """hai triệu tám, hai triệu rưỡi: the unit below the scale is implied. "một" after a number means per: mười tám nghìn một ký."""\n    a = at(w, j)\n    if a == "rưỡi":\n        return scale // 2, j + 1\n    if a != "một" and ONES.get(a) and at(w, j + 1) not in UNITS | YEAR:\n        return ONES[a] * scale // 10, j + 1\n    return 0, j\n\n\ndef group(w, i):\n    """One number below 1000: (value, next index), or None."""\n    a, b = at(w, i), at(w, i + 1)\n    if value(a) is not None and b == "trăm":\n        v, j = 100 * value(a), i + 2\n    elif a == "trăm" and (tens(w, i + 1) or DIGIT.get(b, 0) >= 2):  # trăm hai mốt\n        v, j = 100, i + 1\n    else:\n        return tens(w, i) or ((value(a), i + 1) if value(a) is not None and b in SCALE else None)\n    if at(w, j) in ("lẻ", "linh") and at(w, j + 1) in DIGIT:\n        return v + DIGIT[w[j + 1]], j + 2\n    t = tens(w, j)\n    if t:\n        return v + t[0], t[1]\n    r, j = tail(w, j, 100)\n    return v + r, j\n\n\ndef cardinal(w, i):\n    """Longest number at w[i]: (value, next index), or None."""\n    total, j, scale = 0, i, 0\n    while g := group(w, j):\n        v, k = g\n        s = SCALE.get(at(w, k), 0)\n        if not s or (scale and s >= scale):\n            return (total + v, k) if not s else (total, j)\n        total, j, scale = total + v * s, k + 1, s\n    if not scale:\n        return None\n    r, j = tail(w, j, scale)\n    return total + r, j\n\n\ndef digits(s):\n    w, out, i = s.split(), [], 0\n    while i < len(w):\n        n = i\n        while at(w, n) in DIGIT:\n            n += 1\n        n -= n - i > 3 and at(w, n) in UNITS  # rv hai một chín tám giờ: the last digit belongs to the unit\n        if i and w[i - 1] == "thứ":  # thứ hai is Monday\n            out.append(w[i])\n            i += 1\n        elif c := cardinal(w, i):\n            out.append(str(c[0]))\n            i = c[1]\n        elif n - i >= 3:  # ponytail: any three digit words in a row become a digit string, "ba năm một lần" would too\n            out.append("".join(str(DIGIT[x]) for x in w[i:n]))\n            i = n\n        elif w[i] in DIGIT and w[i] != "không" and (at(w, i - 1) in BEFORE and i or at(w, i + 1) in UNITS):\n            out.append(str(DIGIT[w[i]]))\n            i += 1\n        else:\n            out.append(w[i])\n            i += 1\n    return " ".join(out)\n\n\ndef itn(s):\n    return " ".join(p if p[0] == "[" else digits(p) for p in re.split(r"(\\[[^\\]]*\\])", s) if p.strip())\n\n\nif __name__ == "__main__":\n    for a, b in {\n        "chuyển lan hai trăm nghìn": "chuyển lan 200000",\n        "hai triệu ba trăm tám chục nghìn": "2380000",\n        "một triệu rưỡi tiền cọc": "1500000 tiền cọc",\n        "hai triệu tám và nạp": "2800000 và nạp",\n        "hai triệu tám giờ sáng": "2000000 8 giờ sáng",\n        "mười tám nghìn một ký": "18000 1 ký",\n        "trăm hai mốt triệu": "121000000",\n        "bốn chục triệu": "40000000",\n        "năm trăm đô": "500 đô",\n        "bật điều hoà hai sáu độ": "bật điều hoà 26 độ",\n        "số không chín một hai bốn năm sáu bảy tám một ba": "số 09124567813",\n        "chín giờ kém mười lăm": "9 giờ kém 15",\n        "năm giờ bốn lăm": "5 giờ 45",\n        "đặt báo thức sáu rưỡi sáng mai": "đặt báo thức 6 rưỡi sáng mai",\n        "thứ hai tuần sau bay một mình": "thứ hai tuần sau bay một mình",\n        "ngày mốt mở một bài": "ngày mốt mở một bài",\n        "có không ba ơi năm nay": "có không ba ơi năm nay",\n        "mười năm rồi": "10 năm rồi",\n        "tháng chín mùng bốn": "tháng 9 mùng 4",\n        "gọi [anh ba] năm phút": "gọi [anh ba] 5 phút",\n        "[số một] hai mươi": "[số một] 20",\n        "vài trăm nghìn": "vài trăm nghìn",\n        "119 triệu": "119000000",\n        "rv hai một chín tám giờ": "rv 219 8 giờ",\n        "lịch hôm hai tám": "lịch hôm 28",\n        "": "",\n    }.items():\n        assert itn(a) == b, (a, itn(a), b)\n    print("vi_itn self-check ok")\n', 'import glob\nimport json\nimport re\nimport sys\nimport unicodedata\n\nimport jiwer\n\nfrom settings import *\n\nsys.path[:0] = [".", LIFEMATE_DIR]  # vi_itn.py is written by the cell above\nfrom vi_itn import itn\nimport lifemate as lm\n\nTONES = {"\\u0300", "\\u0301", "\\u0303", "\\u0309", "\\u0323"}\n\n\ndef old_style_tone(w):\n    """Same as 01c: hoà -> hòa, khoẻ -> khỏe, thuỷ -> thủy."""\n    d = unicodedata.normalize("NFD", w)\n    t = [c for c in d if c in TONES]\n    if len(t) != 1:\n        return w\n    b = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))\n    if b.endswith(("oa", "oe")) or (b.endswith("uy") and not b.endswith("quy")):\n        return unicodedata.normalize("NFC", b[:-2] + b[-2] + t[0] + b[-1])\n    return w\n\n\ndef norm(s):\n    """01c\'s norm(), plus thousands separators joined first (200.000đ -> 200000đ), which Gipformer never writes."""\n    s = re.sub(r"(?<=\\d)[.,](?=\\d{3}(?!\\d))", "", unicodedata.normalize("NFC", s))\n    s = re.sub(r"[^\\w\\s]|_", " ", s.lower())\n    return " ".join(old_style_tone(w) for w in s.split())\n\n\nraw = {}\nfor p in RAW_JSON:\n    part = json.load(open(p, encoding="utf-8"))\n    print(p, len(part), "clips")\n    raw |= part\nnew = set(raw)\nbase = glob.glob("/kaggle/input/**/asr_transcripts.json", recursive=True)\nprint("01c asr_transcripts.json attached:", base)\nassert RAW_JSON or base, "attach raw_asr*.json, or a 01c output for an offline run"\nprint("offline run: every clip from 01c" if not RAW_JSON else f"{len(new)} clips from the raw files")\nif FILL_FROM_01C or base:  # 01c text is already norm() + itn(), and both are idempotent on it\n    assert len(base) == 1, f"FILL_FROM_01C needs exactly one attached asr_transcripts.json, found {base}"\n    raw = json.load(open(base[0], encoding="utf-8")) | raw\nxfit = glob.glob("/kaggle/input/**/asr_xfit_train.json", recursive=True)\nprint("cross-fitted train transcripts:", xfit)\nif xfit:  # from fold models that never heard these clips, so the held-out score is not inflated. The raw files still win\n    raw |= {f"train/{k}": v for k, v in json.load(open(xfit[0], encoding="utf-8")).items() if f"train/{k}" not in new}\nneed = {s: [f"{s}/{c}" for x in rows for c in lm.clips(x)] for s, rows in\n        (("public_test", lm.load_split(ROOT, "public_test")), ("train", sum(lm.split_rows(lm.load_split(ROOT, "train"))[1:], [])))}\nfor s, keys in need.items():\n    miss = [k for k in keys if k not in raw]\n    print(f"{s}: {len(keys) - len(miss)} of {len(keys)} clips present, {sum(k in new for k in keys)} from the new ASR, {sum(not raw.get(k, \'x\').strip() for k in keys)} empty")\n    assert not miss or (s == "train" and not EVAL), f"{len(miss)} {s} clips missing, e.g. {miss[:3]}" + (". Set EVAL = False for public_test only" if s == "train" else ". Attach the 01c output (thbased01, the one with asr_transcripts.json) to fill them")\nasr = {k: itn(norm(v)) for k, v in raw.items()}\njson.dump(asr, open(ASR_JSON, "w", encoding="utf-8"), ensure_ascii=False)\n\nif EVAL:  # same WER as 01c reports after ITN: both sides through norm() and itn()\n    truth = {f"train/{c}": t for x in sum(lm.split_rows(lm.load_split(ROOT, "train"))[1:], []) for c, t in lm.gold_texts(x).items()}\n    keys = sorted(truth)\n    w = jiwer.wer([itn(norm(truth[k])) for k in keys], [asr[k] for k in keys])\n    print(f"held-out WER {w:.4f} on {len(keys)} clips (01b stock 0.1609, 01c FINAL heard them: 0.0194)")\n    assert w > 0.05 or any(k in new for k in keys), "this ASR trained on the held-out clips, the score would be inflated: attach asr_xfit_train.json or a FINAL = False 01c output"\nfor k in need["public_test"][:5]:\n    print(repr(raw[k][:120]), "->", repr(asr[k][:120]))\n', '%%writefile infer.py\n"""One shard of the agent: python infer.py <split> <gold|asr> <shard>. One process per split, each split has its own zaic_tools."""\nimport json\nimport re\nimport sys\nimport time\n\nimport torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer, set_seed\nfrom transformers.integrations import sdpa_attention\n\nfrom settings import *\n\nsdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False  # a T4 has no grouped-key kernel: Qwen3.5 fell back to float32 math and ran out of memory on long prompts (public_0110, public_0157)\n\nsys.path[:0] = [LIFEMATE_DIR, str(ROOT / sys.argv[1])]\nimport lifemate as lm\nfrom zaic_tools import ToolEnv\n\nsplit, mode, shard = sys.argv[1], sys.argv[2], int(sys.argv[3])\nset_seed(SEED)\ntok = AutoTokenizer.from_pretrained(LLM)\nmodel = AutoModelForCausalLM.from_pretrained(LLM, dtype=DTYPE, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True).eval()\nstop_id = tok.convert_tokens_to_ids(lm.STOP)  # not every repo has a generation_config, so the default eos may never fire\ncatalog = lm.load_catalog(ROOT, split)\nrows = lm.load_split(ROOT, split)\nasr = json.load(open(ASR_JSON, encoding="utf-8")) if mode == "asr" else {}\nif split == "train":\n    rows = sum(lm.split_rows(rows)[1:], [])  # only held-out samples say anything\nrows = rows[shard::SHARDS][: 8 if SMOKE else None]\n\n\n@torch.inference_mode()\ndef generate(msgs):\n    ids = tok(lm.prompt_text(msgs), return_tensors="pt", add_special_tokens=False).to(model.device)\n    assert ids.input_ids.shape[1] + MAX_NEW_TOKENS <= 16384, "over the competition context limit"\n    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, eos_token_id=stop_id, pad_token_id=stop_id)\n    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)\n\n\ndef clean(text):\n    return re.sub(r"[\\[\\]]", "", text) if STRIP_MARKS else text  # norm() removes all punctuation, so every bracket here is a mark\n\n\ndef missing(name, args, installed):\n    """The ask gold would make for an installed tool that misses a required or one-of parameter, else None. Reads too: gold asks for them 7 times (flights.search date). A one-of group names its first member, as gold does for ewallet.transfer (contact_id)."""\n    tool = catalog.get(name)\n    if tool is None or name.split(".")[0] not in installed:\n        return None\n    p = tool.get("parameters", {})\n    empty = lambda k: args.get(k) in (None, "", [], {})  # zaic_tools counts these as missing\n    need = [k for k in p.get("required", []) if empty(k)]\n    if p.get("x-one-of") and all(any(empty(k) for k in grp) for grp in p["x-one-of"]):\n        need.append(p["x-one-of"][0][0])\n    return {"tool_name": name, "params": need} if need else None\n\n\ndef run(item):\n    env = ToolEnv(split, item["id"], root=ROOT)  # exactly one per sample, every call on it goes into the submission\n    texts = lm.gold_texts(item) if mode == "gold" else {c: clean(asr[f"{split}/{c}"]) for c in lm.clips(item)}\n    msgs = lm.build_messages(item, texts, catalog)\n    t0, invalid, failed, stop, raw = time.time(), 0, 0, "max_steps", []\n    retried = False\n    try:\n        for _ in range(MAX_STEPS):\n            if time.time() - t0 > TIME_BUDGET:\n                stop = "budget"\n                break\n            raw.append(generate(msgs))\n            action = lm.parse_action(raw[-1])\n            if action is None:  # parse_action returns None for both, and it stays as saved with the weights\n                stop = "done" if \'"done"\' in raw[-1] else "unparsable"\n                break\n            call = {"name": action[0], "arguments": action[1]}\n            said = json.JSONDecoder().raw_decode(raw[-1][raw[-1].index("{"):])[0]  # the model\'s own JSON, so a trained "why" stays in context. Equals call for other models\n            check = env.validate(*action)\n            if not check["ok"]:  # validate is not logged: a call we never make costs no U penalty\n                ask = missing(*action, item["context"]["installed_apps"])\n                if ask and env.validate(lm.ASK, ask)["ok"]:  # gold ends on its ask too\n                    env.call(lm.ASK, ask)\n                    stop = "repaired"\n                    break\n                if not retried:  # one more try with the error as the result (4 of 5 invalid stops on public called an uninstalled contacts.search)\n                    retried = True\n                    msgs += [lm.asst(said), lm.result_msg(check)]\n                    continue\n                invalid, stop = 1, "invalid"\n                break\n            if env.get_tool_calls().count(call) >= 2:  # greedy loop, gold repeats a call at most once\n                stop = "repeat"\n                break\n            if catalog.get(call["name"], {}).get("kind") == "read" and [c["name"] for c in env.get_tool_calls()].count(call["name"]) >= 2:  # gold reads one tool 3+ times in 1 of 1,545 samples; LFM3 ver3 retried empty searches 127 times on public, then sometimes wrote on a guess\n                stop = "repeat"\n                break\n            result = env.call(*action)\n            msgs += [lm.asst(said), lm.result_msg(result)]\n            if "query" in action[1] and not catalog[call["name"]]["parameters"].get("required") and any(v == [] for v in result.values()):  # gold never reads an empty list (0 of 234 contacts.search): the query was misheard, so list everything once and let the model pick. 25 public samples hit this (contacts, event_tickets); a full list fits MAX_RESULT_CHARS\n                msgs += [lm.asst({"name": call["name"], "arguments": {}}), lm.result_msg(env.call(call["name"], {}))]\n            if call["name"] == lm.ASK:  # gold always ends here\n                stop = "ask"\n                break\n    except Exception as e:  # keep the calls already made, a crash must not lose the whole line\n        failed, stop = 1, "crash"\n        print(item["id"], repr(e), flush=True)\n    return {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(), "sec": time.time() - t0, "invalid": invalid, "failed": failed, "stop": stop, "raw": raw}\n\n\nwith open(f"pred_{split}_{mode}_{shard}.jsonl", "w", encoding="utf-8") as f:\n    for n, item in enumerate(rows):\n        line = run(item)\n        if split == "train":  # scoring only: the second env replays gold for its hash, nothing here reaches a submission\n            gold = item["label"]["tool_calls"]\n            ref = ToolEnv(split, item["id"], root=ROOT)\n            for c in gold:\n                ref.call(c["name"], c["arguments"])\n            line["score"] = lm.score(line["tool_calls"], gold, line["state_hash"], ref.get_state_hash(), catalog)\n            line["unseen"] = bool(lm.used_apps(item) & lm.HOLDOUT_APPS)\n        f.write(json.dumps(line, ensure_ascii=False) + "\\n")\n        f.flush()\n        if n % 25 == 0:\n            print(split, mode, shard, n, "/", len(rows), f"{line[\'sec\']:.1f}s", flush=True)', 'from settings import *\n\nprint("LLM:", LLM, "\\nASR:", RAW_JSON, "->", ASR_JSON)\n\nif EVAL:\n    if GOLD:\n        !CUDA_VISIBLE_DEVICES=0 python infer.py train gold 0 & CUDA_VISIBLE_DEVICES=1 python infer.py train gold 1 & wait\n    !CUDA_VISIBLE_DEVICES=0 python infer.py train asr 0 & CUDA_VISIBLE_DEVICES=1 python infer.py train asr 1 & wait\n!CUDA_VISIBLE_DEVICES=0 python infer.py public_test asr 0 & CUDA_VISIBLE_DEVICES=1 python infer.py public_test asr 1 & wait\n', 'import collections\nimport json\nimport statistics as st\n\n\ndef load(split, mode):\n    return [json.loads(l) for f in sorted(glob.glob(f"pred_{split}_{mode}_*.jsonl")) for l in open(f, encoding="utf-8")]\n\n\ndef report(name, part):\n    if part:\n        s = sorted(x["sec"] for x in part)\n        print(f"{name:20s} n={len(part):4d} score={100 * st.mean(x[\'score\'] for x in part):6.2f} invalid={sum(x[\'invalid\'] for x in part)} failed={sum(x[\'failed\'] for x in part)} p95={s[int(.95 * len(s))]:.1f}s max={s[-1]:.1f}s")\n\n\nif EVAL:\n    for mode in ("gold", "asr"):\n        p = load("train", mode)\n        report(f"{mode} unseen", [x for x in p if x["unseen"]])\n        report(f"{mode} seen", [x for x in p if not x["unseen"]])\n        report(f"{mode} under 25 s", [x for x in p if x["sec"] <= 25])\n        report(f"{mode} over 25 s", [x for x in p if x["sec"] > 25])\n        for stop in sorted({x["stop"] for x in p}):\n            report(f"{mode} {stop}", [x for x in p if x["stop"] == stop])  # repaired: was invalid in 03c', 'pred = load("public_test", "asr")\nassert len({x["id"] for x in pred}) == len(pred)\nexpected = sum(1 for _ in open(ROOT / "public_test/public_test.jsonl", encoding="utf-8"))\nassert SMOKE or len(pred) == expected, f"{len(pred)} lines, expected {expected}"\nwith open("submission.jsonl", "w", encoding="utf-8") as f:\n    for x in pred:\n        f.write(json.dumps({k: x[k] for k in ("id", "tool_calls", "state_hash")}, ensure_ascii=False) + "\\n")\nprint(collections.Counter(x["stop"] for x in pred).most_common(), sum(x["sec"] > 30 for x in pred), "samples over 30 s")\nfor x in [x for x in pred if x["stop"] == "unparsable"][:5]:\n    print(x["id"], repr(x["raw"][-1][:200]))\nprint(f"{len(pred)} lines, {sum(x[\'failed\'] for x in pred)} failed, {sum(x[\'invalid\'] for x in pred)} stopped on an invalid call, slowest {max(x[\'sec\'] for x in pred):.1f}s")\n!cd "{ROOT}/public_test" && PYTHONPATH=. python -m zaic_tools --split public_test --root .. --validate /kaggle/working/submission.jsonl\nprint("Baseline, bowen + 01c thbased01: public 60.29, held-out gold 71.77, held-out asr 66.06 (inflated, 01c heard those clips)")\n']  # --pack fills this in a copy of this file

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="eval_in")
ap.add_argument("--out", default="eval_out")
ap.add_argument("--llm", default="hoangducbao/k2-rl", help="Kaggle notebook whose output holds llm_model/ + lifemate.py")
ap.add_argument("--shards", type=int, default=2, help="processes on the one GPU")
ap.add_argument("--no-eval", action="store_true", help="public_test only")
ap.add_argument("--no-pull", action="store_true", help="inputs are already under --input")
ap.add_argument("--gold", action="store_true", help="also run gold text on the held-out rows")
ap.add_argument("--smoke", action="store_true", help="8 samples per shard")
ap.add_argument("--notebook", default=str(Path(__file__).resolve().parent / "notebooks/neweval_sft.ipynb"))
ap.add_argument("--pack", action="store_true", help="write eval_vast_onefile.py (this file + the notebook's cells) and stop")
args = ap.parse_args()

# folder name: (kagglehub function, handle, single file). The data package and the raw transcripts are private datasets
SOURCES = {"data": ("dataset_download", "hoangducbao/zola-deepseek-bundle", None),
           "llm": ("notebook_output_download", args.llm, None),
           "asr": ("notebook_output_download", "png0806/asrtest1", "asr_transcripts.json"),  # 01c output, only this file
           "raw": ("dataset_download", "hoangducbao/eval-raw-asr", None)}  # held-out train clips, stock Gipformer

CELLS = PACKED or ["".join(c["source"]) for c in json.load(open(args.notebook, encoding="utf-8"))["cells"]]
if args.pack:
    marker = "PACKED = " + "None  #"
    src = Path(__file__).read_text(encoding="utf-8")
    assert src.count(marker) == 1
    one = Path(__file__).with_name("eval_vast_onefile.py")
    one.write_text(src.replace(marker, f"PACKED = {CELLS!r}  #"), encoding="utf-8")
    sys.exit(f"wrote {one}")

inp, out = Path(args.input).resolve(), Path(args.out).resolve()
assert inp not in (out, *out.parents), "--out inside --input: the next run would find two asr_transcripts.json"
inp.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)


def cell(start, *pairs):
    """The one cell that starts with `start`, with exact-string patches that must each match once."""
    (src,) = [c for c in CELLS if c.startswith(start)]
    for old, new in pairs:
        assert src.count(old) == 1, f"must match once, found {src.count(old)}: {old[:60]!r}"
        src = src.replace(old, new)
    return src.split("\n", 1)[1] if src.startswith("%%") else src


def py(*argv, wait=True, log=None, cwd=out):
    p = subprocess.Popen([sys.executable, *argv], cwd=cwd, stdout=log, stderr=subprocess.STDOUT if log else None)
    if wait:
        assert p.wait() == 0, f"{argv[0]} failed"
    return p


def write(name, src):
    assert "/kaggle/" not in src, f"{name} still has a Kaggle path"
    (out / name).write_text(src, encoding="utf-8")


import torch  # noqa: E402  the image's own build

# before any download: infer.py falls back to the CPU without a word, and an old torch build cannot run on a 5090 (sm_120)
assert torch.cuda.is_available(), "no CUDA: this box needs a torch build for its GPU"
torch.zeros(8, device="cuda").sum().item()
print(torch.__version__, torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB", flush=True)

root = dict.fromkeys(SOURCES, inp)  # where each input is looked for
if not args.no_pull:
    py("-m", "pip", "install", "-q", "kagglehub")
    os.environ["KAGGLEHUB_CACHE"] = str(inp / "kagglehub")  # not ~/.cache: on the big disk, and a rerun finds it. No leading dot: glob ** skips hidden folders
    import kagglehub

    def get(fn, handle, file=None):
        """One download, retried: a single file resumes where a broken connection left it."""
        for attempt in range(5):
            try:
                return Path(getattr(kagglehub, fn)(handle, **({"path": file} if file else {})))
            except Exception as e:
                print(f"  attempt {attempt + 1} failed: {e!r}", flush=True)
                err = e
        sys.exit(f"could not download {handle} {file or ''}: {err!r}\nIs KAGGLE_API_TOKEN set (or ~/.kaggle/kaggle.json), and can that "
                 "account open it? A teammate's private notebook must be shared with you.")

    def model_files(handle):
        """llm_model/ + lifemate.py of the notebook's latest output. The whole output comes as one zip built on the fly,
        which cannot resume: a broken connection lost the ~8 GB model every time."""
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        names, token = [], ""
        with build_kaggle_client() as client:
            while True:
                r = ApiListKernelSessionOutputRequest()
                r.user_name, r.kernel_slug, r.page_size = *handle.split("/")[:2], 200
                if token:
                    r.page_token = token
                resp = client.kernels.kernels_api_client.list_kernel_session_output(r)
                names += [f.file_name for f in resp.files or []]
                token = resp.next_page_token
                if not token:
                    break
        want = [n for n in names if n.startswith("llm_model/") or n == "lifemate.py"]
        assert {"llm_model/config.json", "lifemate.py"} <= set(want), f"no llm_model/ + lifemate.py in the output of {handle}: {names[:20]}"
        return want

    for name, (fn, handle, file) in SOURCES.items():
        if name == "raw" and args.no_eval:
            continue
        print("Kaggle:", handle, file or "", flush=True)
        if name == "llm":  # shortcut: lists the latest version's files, also for a .../versions/N handle (same names expected)
            files = model_files(handle)
            for f in files:
                print("  ", f, flush=True)
                p = get(fn, handle, f)
            root[name] = p.parents[len(Path(files[-1]).parts) - 1]  # the output's top folder
        else:
            root[name] = get(fn, handle, file)
    root["asr"], root["data"] = root["asr"].parent, inp
    if not list(inp.rglob("train/train.jsonl")):  # the bundle was uploaded as a zip and Kaggle kept it zipped
        for z in inp.rglob("*.zip"):
            zipfile.ZipFile(z).extractall(inp / "unzipped")
g = {k: v.as_posix() for k, v in root.items()}
for name, pattern in (("data", "train/train.jsonl"), ("llm", "llm_model/config.json"), ("asr", "asr_transcripts.json")) + (() if args.no_eval else (("raw", "raw_asr*.json"),)):
    found = list(root[name].rglob(pattern))
    assert len(found) == 1 or (name == "raw" and (found or args.no_pull)), (
        f"need exactly one {pattern} under {root[name]}, found {[str(p) for p in found]}; files there: {[p.name for p in root[name].rglob('*') if p.is_file()][:20]}")
(config,) = root["llm"].rglob("llm_model/config.json")
assert (config.parents[1] / "lifemate.py").is_file(), f"no lifemate.py beside {config.parent}"

# the notebook's install cell: its transformers version for this model type, its pip line
setup = {line.split(" ", 1)[0]: line for line in cell("import glob\nimport json\n\nimport torch").split("\n")}
ns = {"_config": json.load(open(config, encoding="utf-8"))}
exec(setup["TRANSFORMERS"], ns)
py("-m", *shlex.split(setup["!pip"][1:].replace("{TRANSFORMERS}", ns["TRANSFORMERS"])))

write("settings.py", cell("%%writefile settings.py",
                          ("SMOKE = False", f"SMOKE = {args.smoke}"), ("EVAL = True", f"EVAL = {not args.no_eval}"),
                          ("GOLD = False", f"GOLD = {args.gold}"), ("SHARDS = 2", f"SHARDS = {args.shards}"),
                          ('"/kaggle/input/**/train/train.jsonl"', f'"{g["data"]}/**/train/train.jsonl"'),
                          ('"/kaggle/input/**/llm_model/config.json"', f'"{g["llm"]}/**/llm_model/config.json"'),
                          ('"/kaggle/input/**/raw_asr*.json"', f'"{g["raw"]}/**/raw_asr*.json"')))
write("vi_itn.py", cell("%%writefile vi_itn.py"))
write("infer.py", cell("%%writefile infer.py"))  # unchanged
write("transcripts.py", cell("import glob\nimport json\nimport re\nimport sys\nimport unicodedata",
                             ('"/kaggle/input/**/asr_transcripts.json"', f'"{g["asr"]}/**/asr_transcripts.json"'),
                             ('"/kaggle/input/**/asr_xfit_train.json"', f'"{inp.as_posix()}/**/asr_xfit_train.json"')))
write("report.py", "from settings import *\n" + cell("import collections") + "\n" + cell(
    'pred = load("public_test", "asr")',
    ('!cd "{ROOT}/public_test" && PYTHONPATH=. python -m zaic_tools --split public_test --root .. --validate /kaggle/working/submission.jsonl',
     'import subprocess, sys\nsubprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", '
     f'{(out / "submission.jsonl").as_posix()!r}], cwd=ROOT / "public_test", check=True)')) + '''
if EVAL:  # qwen_k2_rl trains on the other held-out rows, so for its models only this line is a clean score
    sys.path.insert(0, LIFEMATE_DIR)
    import lifemate as lm
    gate = {x["id"] for x in sum(lm.split_rows(lm.load_split(ROOT, "train"))[1:], [])[::5]}
    report("asr gate rows", [x for x in load("train", "asr") if x["id"] in gate])
print("submission:", ''' + repr((out / "submission.jsonl").as_posix()) + ")\n")

py("vi_itn.py")  # self-check
py("transcripts.py")
runs = ([("train", "gold")] if args.gold and not args.no_eval else []) + ([] if args.no_eval else [("train", "asr")]) + [("public_test", "asr")]
for split, mode in runs:
    print("running", split, mode, "- follow with: tail -f", out / f"log_{split}_{mode}_0.txt", flush=True)
    logs = [open(out / f"log_{split}_{mode}_{i}.txt", "w", encoding="utf-8") for i in range(args.shards)]
    procs = [py("infer.py", split, mode, str(i), wait=False, log=f) for i, f in enumerate(logs)]
    assert not any([p.wait() for p in procs]), f"a shard of {split} {mode} failed, see log_{split}_{mode}_*.txt"
py("report.py")

failed = sum(json.loads(line)["failed"] for f in out.glob("pred_*.jsonl") for line in open(f, encoding="utf-8"))
if failed:  # infer.py keeps a crashed sample's line, so an out-of-memory error only shows up here
    sys.exit(f"{failed} samples crashed (see log_*.txt). If it is CUDA out of memory, rerun with --shards {max(1, args.shards - 1)}")
