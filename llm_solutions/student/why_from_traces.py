"""DeepSeek train traces (Kaggle notebook, WHY = True) -> rationales.json for the student SFT notebook (bowen.ipynb, WHY = True).

rationales.json = {row id: [reason per history call..., reason per final call..., reason for done]}; the SFT notebook takes the
last len(labelled) entries. This script copies --base and replaces the final-call reasons of a row with DeepSeek's written
reasons only when the trace is a clean copy of gold: score 1.0, every call executed (no bounced/schema/repeat calls), one call
per step, the same tool names in gold order, and a short non-empty reason before every call. History and done reasons stay.
Sources: train split only (R-RES-3, F-2): gold labels + DeepSeek run on gold text (R-RES-2, development only).

Usage: python student/why_from_traces.py --traces traces_train_deepseek1545_why.jsonl [more traces] --base rationales.json --out rationales_ds.json
       python student/why_from_traces.py --self-check
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CHARS = 300  # the student writes the reason before every action and has 30 s per sample (R-RUN-2); longer ones are not cut, the base reason stays


def n_history_calls(x):
    return sum(len(t.get("tool_calls") or []) for t in x["history"] if t["role"] == "assistant")


def reasons(x, trace):
    """DeepSeek's reasons for x's final calls in gold order, or (None, why not)."""
    if not x["label"]["tool_calls"]:
        return None, "gold has no calls"
    if not trace or trace.get("score") != 1.0:
        return None, "score below 1"
    steps = [s for s in trace["steps"] if s["calls"]]
    if any(len(s["calls"]) > 1 for s in steps):
        return None, "several calls in one step"
    if any(s["calls"][0]["status"] != "ok" for s in steps):
        return None, "a call was not executed"
    if [s["calls"][0]["name"] for s in steps] != [c["name"] for c in x["label"]["tool_calls"]]:
        return None, "other call order"
    out = [re.sub(r"\s+", " ", s.get("text") or "").strip() for s in steps]
    if not all(out):
        return None, "no written reason"
    if max(map(len, out)) > MAX_CHARS:
        return None, f"reason over {MAX_CHARS} chars"
    return out, "replaced"


def convert(rows, traces, base):
    out, why = {}, collections.Counter()
    for x in rows:
        b = base[x["id"]]
        n = len(x["label"]["tool_calls"])
        if len(b) != n_history_calls(x) + n + 1:
            raise SystemExit(f"{x['id']}: base has {len(b)} reasons, expected history calls + final calls + done")
        r, w = reasons(x, traces.get(x["id"])) if x["id"] in traces else (None, "no trace")
        why[w] += 1
        out[x["id"]] = b[: len(b) - n - 1] + r + b[-1:] if r else b
    return out, why


def self_check():
    call = lambda name: {"name": name, "arguments": {}}  # noqa: E731
    x = {"id": "t1", "history": [{"role": "user"}, {"role": "assistant", "tool_calls": [call("a.read")]}],
         "label": {"tool_calls": [call("a.read"), call("a.write")]}}
    base = {"t1": ["h", "f1", "f2", "done"]}
    step = lambda name, text, status="ok": {"text": text, "calls": [dict(call(name), status=status)]}  # noqa: E731
    good = {"id": "t1", "score": 1.0, "steps": [step("a.read", " cần  ID "), step("a.write", "ghi"), {"text": "xong", "calls": []}]}
    assert convert([x], {"t1": good}, base)[0]["t1"] == ["h", "cần ID", "ghi", "done"]
    bad = [dict(good, score=0.5), dict(good, steps=[step("a.write", "ghi"), step("a.read", "đọc")]),
           dict(good, steps=[step("a.read", "x", "rare_arg"), step("a.read", "x"), step("a.write", "y")]),
           dict(good, steps=[step("a.read", ""), step("a.write", "y")]),
           dict(good, steps=[{"text": "x", "calls": [dict(call("a.read"), status="ok"), dict(call("a.write"), status="ok")]}])]
    for t in bad:
        assert convert([x], {"t1": t}, base)[0]["t1"] == base["t1"], t
    assert convert([x], {}, base)[1] == {"no trace": 1}
    empty = dict(x, history=[], label={"tool_calls": []})
    assert convert([empty], {"t1": dict(good, steps=[{"text": "ngoài phạm vi", "calls": []}])}, {"t1": ["done"]})[0]["t1"] == ["done"]
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", help="train traces from the notebook's cell 4 with WHY = True")
    ap.add_argument("--base", help="existing rationales.json with every train row (the SFT notebook asserts that)")
    ap.add_argument("--out", help="new rationales.json")
    ap.add_argument("--train", default=str(ROOT / "train+public_test/train/train.jsonl"))
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if not (a.traces and a.base and a.out):
        ap.error("--traces, --base and --out are required")
    rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    traces = {}
    for f in a.traces:
        for l in open(f, encoding="utf-8"):
            o = json.loads(l)
            if not o["id"].startswith("train_"):
                raise SystemExit(f"{f}: {o['id']} is not a train row; only train traces may become training data (R-RES-3)")
            traces[o["id"]] = o
    out, why = convert(rows, traces, json.load(open(a.base, encoding="utf-8")))
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} rows -> {a.out}; final-call reasons from DeepSeek in {why['replaced']} rows")
    print("kept the base reasons because:", dict(why))


if __name__ == "__main__":
    main()
