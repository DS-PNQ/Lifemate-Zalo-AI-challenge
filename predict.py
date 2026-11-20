"""LifeMate inference for the Docker image: /data -> /result/submission.jsonl, fully offline.

Per sample: ASR of the final clip and every history user clip (01c's fine-tuned Gipformer, sherpa-onnx greedy, CPU), then 01c's norm() and
vi_itn.py, then the agent loop of adahome/03c_eval_repair.ipynb (validate first, missing-parameter repair, empty-search retry, repeat guards)
with the fine-tuned Qwen3.5-4B served in-process by vLLM, greedy. The 120 s budget of 03c is replaced by a per-sample deadline: no model
step starts after STEP_DEADLINE seconds, and the calls already made are always written.

Weights in WEIGHTS: llm_model/ and lifemate.py from the same 02 output (the prompt must match the weights), asr_model/ and vi_itn.py from 01c."""
import json
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
import wave
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA", "/data"))
OUT = Path(os.environ.get("OUT", "/result/submission.jsonl"))
WEIGHTS = Path(os.environ.get("WEIGHTS", "/code/saved_models"))
MAX_STEPS = 8  # gold never uses more than 6 calls
MAX_NEW_TOKENS = 320  # the longest gold call is 304 characters, plus a written reason when the model was trained with WHY
STEP_DEADLINE = float(os.environ.get("STEP_DEADLINE", 20))  # seconds since the sample started (ASR included). One step after it ends well under 30 s
VOTES = int(os.environ.get("VOTES", 1))  # >1: each step asks the tool list in VOTES fixed orders in one batch and keeps the most common call, as 03c repair's VOTES
SEED = 42


def find_split(data):
    """(root, split) as zaic_tools wants them: <root>/<split>/<split>.jsonl. /data may be the split folder itself (the brief mounts the
    private test folder there) or a folder holding split folders. SPLIT picks one when there are several."""
    found = {c.name[: -len("_tools_catalog.json")]: c.parent for c in [*data.glob("*_tools_catalog.json"), *data.glob("*/*_tools_catalog.json")]}
    tests = sorted(s for s in found if s != "train")
    split = os.environ.get("SPLIT") or (tests[0] if len(tests) == 1 else sys.exit(f"set SPLIT, found {sorted(found)} in {data}"))
    folder = found[split]
    if folder.name == split:
        return folder.parent, split
    link = Path(tempfile.gettempdir()) / "zaic_root" / split  # ToolEnv reads <root>/<split>/..., so give the mounted folder its split name
    link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(folder.resolve(), target_is_directory=True)
    return link.parent, split


def setup(root, split, weights):
    """lifemate.py and vi_itn.py from the weights, zaic_tools from the split (each split ships its own)."""
    global lm, itn, ToolEnv, catalog
    sys.path[:0] = [str(weights), str(root / split)]
    import lifemate as lm
    from vi_itn import itn
    from zaic_tools import ToolEnv

    catalog = lm.load_catalog(root, split)


TONES = {"̀", "́", "̃", "̉", "̣"}


def old_style_tone(word):
    """hoà -> hòa, khoẻ -> khỏe, thuỷ -> thủy: one spelling on both sides of WER. Copied from 01c's asr_common.py."""
    d = unicodedata.normalize("NFD", word)
    tones = [c for c in d if c in TONES]
    if len(tones) != 1:
        return word
    base = unicodedata.normalize("NFC", "".join(c for c in d if c not in TONES))
    if base.endswith(("oa", "oe")) or (base.endswith("uy") and not base.endswith("quy")):
        return unicodedata.normalize("NFC", base[:-2] + base[-2] + tones[0] + base[-1])
    return word


def norm(s):
    """Lowercase, no punctuation, one tone spelling. Copied from 01c's asr_common.py: the LLM saw exactly this text."""
    s = re.sub(r"[^\w\s]|_", " ", unicodedata.normalize("NFC", s).lower())
    return " ".join(old_style_tone(w) for w in s.split())


def read_wav(path):
    """16-bit PCM as float32 in [-1, 1), what soundfile.read(dtype="float32") gave 01c. stdlib wave, so no libsndfile."""
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, f"{path}: expected 16-bit mono"
        return w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def load_asr(model_dir):
    import sherpa_onnx

    rec = sherpa_onnx.OfflineRecognizer.from_transducer(  # greedy search, as in 01c
        encoder=f"{model_dir}/encoder.int8.onnx",
        decoder=f"{model_dir}/decoder.int8.onnx",
        joiner=f"{model_dir}/joiner.int8.onnx",
        tokens=f"{model_dir}/tokens.txt",
        num_threads=int(os.environ.get("ASR_THREADS", 4)),  # fixed, not os.cpu_count(): a container sees the host's CPUs, and the text changes with the thread count (3 of 78 clips at 8)
    )

    def transcribe(paths):
        out = []
        for p in paths:  # one clip at a time: batching pads and changed 14 of 78 clips, and was slower (0.6 s against 1.0 s for 13 clips here)
            s = rec.create_stream()
            s.accept_waveform(*read_wav(p))
            rec.decode_stream(s)
            out.append(itn(norm(s.result.text)))  # asr_transcripts.json of 01c: ITN on the normalised text
        return out

    return transcribe


def load_llm(model_dir):
    from vllm import LLM, SamplingParams

    kwargs = dict(model=str(model_dir), dtype="float16", seed=SEED, max_model_len=16384, max_num_seqs=max(4, VOTES), gpu_memory_utilization=0.85,
                  generation_config="vllm")  # ignore the repo's sampling defaults: a repetition penalty would corrupt repeated IDs
    kwargs |= json.loads(os.environ.get("LLM_KWARGS", "{}"))  # calibration knob, e.g. {"enforce_eager": true} or {"enable_prefix_caching": false}
    llm = LLM(**kwargs)  # prefix caching is vLLM's default; the startup line below says whether it is on for this model
    print("vLLM cache config:", getattr(getattr(llm.llm_engine, "vllm_config", None), "cache_config", None), flush=True)
    tok = llm.get_tokenizer()
    stop = tok.encode(lm.STOP, add_special_tokens=False)
    assert len(stop) == 1, f"{lm.STOP} is not one token: {stop}"
    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS, stop_token_ids=stop)

    def generate(msgs):
        prompts = [{"prompt_token_ids": tok.encode(lm.prompt_text(variant(msgs, k)), add_special_tokens=False)} for k in range(VOTES)]  # tokenised as in 03c, so vLLM adds nothing
        assert len(prompts[0]["prompt_token_ids"]) + MAX_NEW_TOKENS <= 16384, "over the competition context limit"
        return pick([o.outputs[0].text for o in llm.generate(prompts, params, use_tqdm=False)])

    return generate


TOOLS = "Danh sách tool:\n"
AGREE = []  # per step of the current sample: [votes for the kept call, the order that gave it]


def variant(msgs, k):
    """msgs with the tool list in its k-th fixed order; 0 is the order build_messages gave. From 03c repair."""
    if not k:
        return msgs
    head, sep, tools = msgs[0]["content"].partition(TOOLS)
    if not sep:
        raise SystemExit(f"VOTES > 1 needs {TOOLS!r} in the system prompt of this lifemate.py")  # not an Exception: run() must not swallow it sample by sample
    lines = tools.split("\n")
    random.Random(f"{k}|{msgs[0]['content']}").shuffle(lines)  # the same order at every step of a sample and in every run, so each order's prefix stays cached
    return [{**msgs[0], "content": head + TOOLS + "\n".join(lines)}] + msgs[1:]


def key(raw):
    """What the vote compares: the call with free-text (fuzzy) values blanked. From 03c repair."""
    action = lm.parse_action(raw)
    if action is None:
        return "done" if '"done"' in raw else "unparsable"
    props = catalog.get(action[0], {}).get("parameters", {}).get("properties", {})
    return json.dumps([action[0], {k: "~" if props.get(k, {}).get("x-match") == "fuzzy" else v for k, v in action[1].items()}], sort_keys=True, ensure_ascii=False)


def pick(outs):
    """Plurality over the orders, ties to the lowest order: when every answer differs this is order 0, the VOTES = 1 answer. From 03c repair."""
    keys = [key(o) for o in outs]
    best = max(keys, key=lambda k: (keys.count(k), -keys.index(k)))
    AGREE.append([keys.count(best), keys.index(best)])
    return outs[keys.index(best)]


def missing(name, args, installed):
    """The ask gold would make for an installed tool that misses a required or one-of parameter, else None. From 03c repair."""
    tool = catalog.get(name)
    if tool is None or name.split(".")[0] not in installed:
        return None
    p = tool.get("parameters", {})
    empty = lambda k: args.get(k) in (None, "", [], {})  # zaic_tools counts these as missing
    need = [k for k in p.get("required", []) if empty(k)]
    if p.get("x-one-of") and all(any(empty(k) for k in grp) for grp in p["x-one-of"]):
        need.append(p["x-one-of"][0][0])
    return {"tool_name": name, "params": need} if need else None


def run(item, folder, split, root, generate, transcribe):
    """One sample: (submission line, details). The loop is 03c repair's, with STEP_DEADLINE in place of its 120 s budget."""
    t0 = time.time()
    env = ToolEnv(split, item["id"], root=root)  # exactly one per sample, every call on it goes into the submission
    stop, raw, texts, asr_sec = "max_steps", [], {}, -1.0
    retried = False
    AGREE.clear()
    try:
        paths = lm.clips(item)
        texts = dict(zip(paths, transcribe([folder / p for p in paths])))
        asr_sec = time.time() - t0
        msgs = lm.build_messages(item, texts, catalog)
        for _ in range(MAX_STEPS):
            if time.time() - t0 > STEP_DEADLINE:
                stop = "deadline"
                break
            raw.append(generate(msgs))
            action = lm.parse_action(raw[-1])
            if action is None:  # parse_action returns None for both
                stop = "done" if '"done"' in raw[-1] else "unparsable"
                break
            call = {"name": action[0], "arguments": action[1]}
            said = lm.reply(raw[-1]) if hasattr(lm, "reply") else lm.asst(json.JSONDecoder().raw_decode(raw[-1][raw[-1].index("{"):])[0])  # the model's own output as trained: its thought (ver5 think) or its JSON with a "why"
            check = env.validate(*action)
            if not check["ok"]:  # validate is not logged: a call we never make costs no U penalty
                ask = missing(*action, item["context"]["installed_apps"])
                if ask and env.validate(lm.ASK, ask)["ok"]:  # gold ends on its ask too
                    env.call(lm.ASK, ask)
                    stop = "repaired"
                    break
                if not retried:  # one more try with the error as the result (4 of 5 invalid stops on public called an uninstalled contacts.search)
                    retried = True
                    msgs += [said, lm.result_msg(check)]
                    continue
                stop = "invalid"
                break
            if env.get_tool_calls().count(call) >= 2:  # greedy loop, gold repeats a call at most once
                stop = "repeat"
                break
            if catalog.get(call["name"], {}).get("kind") == "read" and [c["name"] for c in env.get_tool_calls()].count(call["name"]) >= 2:  # gold reads one tool 3+ times in 1 of 1,545 samples
                stop = "repeat"
                break
            result = env.call(*action)
            msgs += [said, lm.result_msg(result)]
            if "query" in action[1] and not catalog[call["name"]]["parameters"].get("required") and any(v == [] for v in result.values()):  # misheard query: list everything once and let the model pick (+0.81 on public)
                msgs += [lm.asst({"name": call["name"], "arguments": {}}), lm.result_msg(env.call(call["name"], {}))]
            if call["name"] == lm.ASK:  # gold always ends here
                stop = "ask"
                break
    except Exception as e:  # keep the calls already made, a crash must not lose the line
        stop = "crash"
        print(item["id"], repr(e), flush=True)
    line = {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(), "time": int((time.time() - t0) * 1000)}  # ms from before ToolEnv and audio, as the organisers' guide
    return line, {"id": item["id"], "sec": round(time.time() - t0, 2), "asr_sec": round(asr_sec, 2), "stop": stop, "raw": raw, "agree": list(AGREE), "text": texts}


def write(root, split, generate, transcribe, out):
    folder = root / split
    rows = lm.load_split(root, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with open(out, "w", encoding="utf-8") as f, open(out.with_name(out.name.replace("submission", "details")), "w", encoding="utf-8") as d:
        for n, item in enumerate(rows):
            line, info = run(item, folder, split, root, generate, transcribe)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            d.write(json.dumps(info, ensure_ascii=False) + "\n")
            f.flush()
            d.flush()
            stats.append(info)
            if n % 25 == 0:
                print(f"{n + 1}/{len(rows)} {info['id']} {info['sec']}s {info['stop']}", flush=True)
    secs = sorted(s["sec"] for s in stats)
    print(f"{len(stats)} lines -> {out}. Stops {Counter(s['stop'] for s in stats).most_common()}. "
          f"Mean {sum(secs) / max(len(secs), 1):.2f}s, max {secs[-1] if secs else 0}s, {sum(s > 30 for s in secs)} over 30 s", flush=True)


def main():
    root, split = find_split(DATA)
    print("split", split, "root", root, flush=True)
    t = time.time()
    setup(root, split, WEIGHTS)
    transcribe = load_asr(WEIGHTS / "asr_model")
    generate = load_llm(WEIGHTS / "llm_model")
    generate([{"role": "system", "content": "warm-up"}, {"role": "user", "content": "xin chào"}])  # CUDA graphs and kernels before any sample is timed
    print(f"load model: {time.time() - t:.1f}s", flush=True)
    t = time.time()
    write(root, split, generate, transcribe, OUT)
    print(f"predict: {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":  # vLLM's engine core is a child process, so the guard is required
    main()
