"""Local check of predict.py without vLLM or a GPU: python test_agent.py <data root> <weights dir> <train|public_test> [asr model dir]
weights dir needs lifemate.py (from the 02 output) and vi_itn.py (from 01c). One split per process: each split ships its own zaic_tools.

train: an oracle model replays gold (half its steps with a "why" key) on gold text, and every sample is scored against a gold replay.
public_test: a scripted model drives every stop path (empty-search retry, repair, invalid, unparsable, deadline), the output goes through
the organisers' validator. With an ASR model dir, a few clips go through load_asr too."""
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import predict as P

root, weights, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert P.find_split(root)[1] == "public_test" and P.find_split(root / "public_test") == (root, "public_test")  # both /data layouts, no symlink needed here
if os.name != "nt" and split == "public_test":  # /data named "data" in the container: find_split links it under the split's name (needs a privilege on Windows)
    link = Path(tempfile.mkdtemp()) / "data"
    link.symlink_to(root / split, target_is_directory=True)
    r, s = P.find_split(link)
    assert s == split and (r / s / f"{s}.jsonl").exists(), (r, s)
P.setup(root, split, weights)
lm = P.lm
rows = lm.load_split(root, split)
out = Path(tempfile.mkdtemp()) / "result" / "submission.jsonl"

# the vote (VOTES > 1): an order keeps every tool line and nothing else changes, order 0 is the prompt itself, plurality with ties to order 0
msgs = lm.build_messages(rows[0], {c: "xin chào" for c in lm.clips(rows[0])}, P.catalog)
v1 = P.variant(msgs, 1)
assert P.variant(msgs, 0) is msgs and v1 == P.variant(msgs, 1) and v1[1:] == msgs[1:] and v1[0] != msgs[0]
assert sorted(v1[0]["content"].split("\n")) == sorted(msgs[0]["content"].split("\n"))
a = '{"why": "x", "name": "sms.send", "arguments": {"contact_id": "c_1", "content": "mai họp"}}'
b = '{"name": "sms.send", "arguments": {"contact_id": "c_1", "content": "họp mai nhé"}}'  # same call, other wording: one vote with a
c = '{"name": "sms.send", "arguments": {"contact_id": "c_2", "content": "mai họp"}}'
assert P.pick([c, a, b]) == a and P.pick([c, a]) == c and P.pick(['{"done": true}', c, '{"done": true}']) == '{"done": true}'
assert P.AGREE == [[2, 1], [1, 0], [2, 0]], P.AGREE
P.AGREE.clear()

if split == "train":
    script = {}

    def generate(msgs):
        return next(script["it"])

    for item in rows:
        steps = [{"why": "lý do.", **c} if i % 2 else c for i, c in enumerate(item["label"]["tool_calls"])] + [{"done": True}]
        script[item["id"]] = (json.dumps(s, ensure_ascii=False) for s in steps)

    def transcribe(paths):
        item = current["item"]
        texts = lm.gold_texts(item)
        return [texts[str(p.relative_to(root / split)).replace("\\", "/")] for p in paths]

    current, scores, stops = {}, [], Counter()
    for item in rows:
        current["item"], script["it"] = item, script[item["id"]]
        line, info = P.run(item, root / split, split, root, generate, transcribe)
        ref = P.ToolEnv(split, item["id"], root=root)  # scoring only, train
        for c in item["label"]["tool_calls"]:
            ref.call(c["name"], c["arguments"])
        s = lm.score(line["tool_calls"], item["label"]["tool_calls"], line["state_hash"], ref.get_state_hash(), P.catalog)
        scores.append(s)
        stops[info["stop"]] += 1
        if s < 1:
            print("below 1:", item["id"], info["stop"], round(s, 3), [c["name"] for c in line["tool_calls"]], "gold", [c["name"] for c in item["label"]["tool_calls"]])
    print(f"oracle on {len(rows)} train samples: {100 * sum(scores) / len(scores):.2f}, {sum(s == 1 for s in scores)} at 1.0, stops {stops.most_common()}")
    assert sum(scores) / len(scores) > 0.99
    sys.exit()

# public_test: one behaviour per sample, cycling
writes = lambda item: [t for a in item["context"]["installed_apps"].values() for t in a["tools"] if P.catalog[t]["kind"] == "write"]
reads_q = lambda item: [t for a in item["context"]["installed_apps"].values() for t in a["tools"] if "query" in P.catalog[t]["parameters"].get("properties", {}) and not P.catalog[t]["parameters"].get("required")]
plans = {
    "retry": lambda it: [{"name": reads_q(it)[0], "arguments": {"query": "zzzz qqqq"}}, {"done": True}] if reads_q(it) else [{"done": True}],
    "repair": lambda it: [{"why": "thiếu.", "name": writes(it)[0], "arguments": {}}, {"done": True}],
    "made_up": lambda it: [{"name": "nope.tool", "arguments": {}}] * 2,  # wrong again after the retry: invalid
    "fixed": lambda it: [{"name": "nope.tool", "arguments": {}}, {"done": True}],  # the retry shows the error, then the model stops
    "garbage": lambda it: ["không có json"],
    "slow": lambda it: [{"name": t, "arguments": {}} for t in reads0(it)[:2]] or [{"done": True}],
}
reads0 = lambda item: [t for a in item["context"]["installed_apps"].values() for t in a["tools"] if P.catalog[t]["kind"] == "read" and not P.catalog[t]["parameters"].get("required") and not P.catalog[t]["parameters"].get("x-one-of")]
kinds = itertools.cycle(plans)
sample = rows[:50]
script, current, seen = {}, {}, []
for item in sample:
    kind = next(kinds)
    script[item["id"]] = (kind, [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in plans[kind](item)])


def generate(msgs):
    kind, steps = script[current["id"]]
    if kind == "fixed" and len(steps) == 1:
        seen.append(str(msgs[-1]))
    if kind == "slow":
        time.sleep(0.3)
    return steps.pop(0) if steps else '{"done": true}'


def transcribe(paths):
    current["id"] = paths[0].parent.name  # audio/<id>/<id>.wav: run() transcribes before the first step
    return ["gọi cho minh anh"] * len(paths)


P.STEP_DEADLINE = 0.2  # "slow" passes it after one step, the others finish before it
lm.load_split = lambda r, s: sample
P.write(root, split, generate, transcribe, out)
details = [json.loads(l) for l in open(out.with_name("details.jsonl"), encoding="utf-8")]
by_kind = Counter((script[d["id"]][0], d["stop"]) for d in details)
print(sorted(by_kind.items()))
assert by_kind[("slow", "deadline")] > 0 and {s for (k, s) in by_kind if k == "slow"} <= {"deadline", "done"}, by_kind
assert {s for (k, s) in by_kind if k == "garbage"} == {"unparsable"} and {s for (k, s) in by_kind if k == "made_up"} == {"invalid"}
assert {s for (k, s) in by_kind if k == "fixed"} == {"done"} and seen and all("nope.tool" in m for m in seen), seen
assert by_kind[("repair", "repaired")] > 0 and by_kind[("retry", "done")] > 0
lines = [json.loads(l) for l in open(out, encoding="utf-8")]
assert [l["id"] for l in lines] == [x["id"] for x in sample] and all(set(l) == {"id", "tool_calls", "state_hash", "time"} and isinstance(l["time"], int) for l in lines)
retried = [l for l in lines if script[l["id"]][0] == "retry" and len(l["tool_calls"]) == 2]
assert retried and all(l["tool_calls"][1]["arguments"] == {} for l in retried), "empty-search retry did not list everything"
slow = [l for l, d in zip(lines, details) if d["stop"] == "deadline"]
assert all(len(l["tool_calls"]) == 1 for l in slow), "a deadline stop lost its calls"
r = subprocess.run([sys.executable, "-m", "zaic_tools", "--split", split, "--root", str(root), "--validate", str(out)], cwd=root / split, capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip().splitlines()[-1])
assert r.returncode == 0 and r.stdout.strip().endswith(", 0 schema errors"), r.stdout + r.stderr

if len(sys.argv) > 4:  # real sherpa-onnx on a few clips, compared with soundfile's reader that 01c used when it is installed
    clips = [root / split / c for x in rows[:3] for c in lm.clips(x)]
    try:
        import soundfile as sf

        assert all((P.read_wav(c)[1] == sf.read(str(c), dtype="float32")[0]).all() for c in clips)
    except ImportError:
        print("soundfile not installed, reader comparison skipped")
    t = time.time()
    texts = P.load_asr(sys.argv[4])(clips)
    print(f"ASR {len(clips)} clips in {time.time() - t:.1f}s:", texts[:3])
print("public_test check ok")
