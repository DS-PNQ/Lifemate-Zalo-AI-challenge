"""E2 (and later E3+): run a ≤4B student on lifemate's held-out train rows and score it like the notebooks' train check.

Agent = student/lifemate.py prompt (plain ChatML, one JSON action per turn) + the branches of the Qwen3.8 notebooks'
step(): validate before call, a failed validate that only misses parameters becomes sys.ask_missing_info, one retry
with the validation error, stop on a repeated call, stop after an ask, MAX_STEPS. No demos, ARG_HINTS or EXTRA_RULES:
this is the prompt the SFT student is trained on, so zero-shot (E2) and fine-tuned (E3) runs compare on equal terms.

Thinking is off: the assistant turn starts with an empty think block ("<think>\n\n</think>\n\n", Qwen's own
non-thinking prefill) unless --no-empty-think. The training data (build_sft.py) has no think block at all, so a
fine-tuned model is run with --no-empty-think.

Backends:
  vllm  Linux, fastest; all live samples are sent together each step (prefix caching on).
  hf    transformers (Qwen3.5 needs a recent version; install flash-linear-attention + causal-conv1d for speed),
        one prompt at a time; works on Windows.
  stub  no model: replays gold (every 5th sample answers junk first) to test the loop, scorer and outputs.
--sequential runs one sample at a time, so time per sample is the R-RUN-2 latency (≤30 s on an A30), not throughput.

Rules: train split only (R-RES-3); one ToolEnv per sample for the agent, a second one only to replay gold for the
gold hash (scorer, as lifemate.demo) (R-ENV-2); submitted calls = env.get_tool_calls() unedited (R-ENV-3); greedy
decoding and fixed seeds (R-REP-1/2); context ≤ 16,384 (R-SIZE-2). Reference/eval script, not the Docker entry point.

Usage:
  python student/eval_student.py --backend vllm --model Qwen/Qwen3.5-4B --name e2_qwen35_4b
  python student/eval_student.py --backend stub --name stub_check
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "train+public_test"
sys.path.insert(0, str(ROOT / "student"))
sys.path.insert(0, str(DATA / "train"))  # the split's own zaic_tools (R-ENV-6)
import lifemate as lm  # noqa: E402
from zaic_tools import ToolEnv  # noqa: E402

SEED = 0
MAX_LEN = 16384  # R-SIZE-2
MAX_NEW = 384  # gold actions are ≤ ~210 tokens (E1)
MAX_STEPS = 8  # gold never uses more than 6 calls
EMPTY_THINK = "<think>\n\n</think>\n\n"


def missing(name, args, installed, catalog):
    """The ask gold would make for an installed tool that misses a required or one-of parameter, else None (as the notebooks)."""
    tool = catalog.get(name)
    if tool is None or name.split(".")[0] not in installed:
        return None
    p = tool.get("parameters", {})
    empty = lambda k: args.get(k) in (None, "", [], {})
    need = [k for k in p.get("required", []) if empty(k)]
    if p.get("x-one-of") and all(any(empty(k) for k in grp) for grp in p["x-one-of"]):
        need.append(p["x-one-of"][0][0])
    return {"tool_name": name, "params": need} if need else None


def step(s, answer, catalog):
    """Apply one model answer to the sample's env; False when the sample stops."""
    env, msgs = s["env"], s["msgs"]
    s["raw"].append(answer)
    action = lm.parse_action(answer)
    if action is None:
        s["stop"] = "done" if '"done"' in answer else "unparsable"
        return False
    call = {"name": action[0], "arguments": action[1]}
    said = json.JSONDecoder().raw_decode(answer[answer.index("{"):])[0]
    check = env.validate(*action)
    if not check["ok"]:  # validate is not logged: a call we never make costs nothing
        ask = missing(*action, s["item"]["context"]["installed_apps"], catalog)
        if ask and env.validate(lm.ASK, ask)["ok"]:
            env.call(lm.ASK, ask)
            s["stop"] = "repaired"
            return False
        if not s["retried"]:
            s["retried"] = True
            msgs += [lm.asst(said), lm.result_msg(check)]
            return True
        s["stop"] = "invalid"
        return False
    calls = env.get_tool_calls()
    if calls.count(call) >= 2 or catalog.get(call["name"], {}).get("kind") == "read" and [c["name"] for c in calls].count(call["name"]) >= 2:
        s["stop"] = "repeat"
        return False
    result = env.call(*action)
    msgs += [lm.asst(said), lm.result_msg(result)]
    if call["name"] == lm.ASK:  # gold always ends here
        s["stop"] = "ask"
        return False
    return True


class Stub:
    """Replays gold. Every 5th sample first calls a tool that does not exist, which must be retried once with the error."""

    def generate(self, prompts, samples):
        out = []
        for s in samples:
            k = len(s["raw"])
            if s["idx"] % 5 == 4 and k == 0:
                s["junked"] = True
                out.append('{"name": "khong.co", "arguments": {}}')
                continue
            gold = s["item"]["label"]["tool_calls"]
            j = k - (1 if s.get("junked") else 0)
            out.append(json.dumps({"name": gold[j]["name"], "arguments": gold[j]["arguments"]}, ensure_ascii=False) if j < len(gold) else '{"done": true}')
        return out


class VLLM:
    def __init__(self, model, gpu_mem):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, max_model_len=MAX_LEN, enable_prefix_caching=True, seed=SEED, gpu_memory_utilization=gpu_mem,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, stop=[lm.STOP], seed=SEED)  # R-REP-2
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts, samples):
        outs = self.llm.generate([{"prompt_token_ids": self.tok(p, add_special_tokens=False)["input_ids"]} for p in prompts], self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HF:
    def __init__(self, model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)  # R-REP-1
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda").eval()
        self.stop = self.tok.convert_tokens_to_ids(lm.STOP)

    def generate(self, prompts, samples):
        out = []
        for p in prompts:
            ids = self.tok(p, add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
            with self.torch.no_grad():
                g = self.model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=self.stop, pad_token_id=self.stop)  # R-REP-2
            out.append(self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False).split(lm.STOP)[0])
        return out


def prompt_len(backend, text):
    tok = getattr(backend, "tok", None)
    return len(tok(text, add_special_tokens=False)["input_ids"]) if tok else len(text) // 3


def run(samples, backend, gen_prefix, catalog):
    live = [s for s in samples if s["stop"] is None]
    while live:
        prompts = []
        for s in live:
            prompts.append(lm.prompt_text(s["msgs"]) + gen_prefix)
            if prompt_len(backend, prompts[-1]) + MAX_NEW > MAX_LEN:
                s["stop"] = "context"
        keep = [(s, p) for s, p in zip(live, prompts) if s["stop"] is None]
        if not keep:
            break
        t = time.perf_counter()
        answers = backend.generate([p for _, p in keep], [s for s, _ in keep])
        dt = (time.perf_counter() - t) / len(keep)
        nxt = []
        for (s, _), a in zip(keep, answers):
            s["seconds"] += dt  # batched: the step's wall time shared equally
            if step(s, a, catalog):
                if len(s["raw"]) >= MAX_STEPS:
                    s["stop"] = "max_steps"
                else:
                    nxt.append(s)
        live = nxt


def gold_hash(item):
    env = ToolEnv("train", item["id"], root=DATA)  # zac-allow: R-ENV-3 scorer-only env: replays gold for the gold hash, never submitted
    for c in item["label"]["tool_calls"]:
        env.call(c["name"], c["arguments"])
    return env.get_state_hash()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("vllm", "hf", "stub"), required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--name", required=True, help="output folder under student/runs/")
    ap.add_argument("--texts", choices=("gold", "asr"), default="gold", help="asr = stock Gipformer transcripts of the train audio")
    ap.add_argument("--every", type=int, default=4, help="every Nth held-out row (4 → 82 rows, as the notebooks' train check; 1 → 328)")
    ap.add_argument("--sequential", action="store_true", help="one sample at a time: time per sample = latency")
    ap.add_argument("--no-empty-think", action="store_true", help="for fine-tuned models trained without a think block")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the N rows with the shortest prompts")
    args = ap.parse_args()
    random.seed(SEED)  # R-REP-1

    rows, catalog = lm.load_split(DATA, "train"), lm.load_catalog(DATA, "train")
    _, unseen, seen = lm.split_rows(rows)
    items = [(x, "unseen") for x in unseen[::args.every]] + [(x, "seen") for x in seen[::args.every]]
    if args.limit:
        items = sorted(items, key=lambda t: len(lm.prompt_text(lm.build_messages(t[0], lm.gold_texts(t[0]), catalog))))[:args.limit]
    asr = json.load(open(ROOT / "asr_zipformer/asr_base_train.json", encoding="utf-8")) if args.texts == "asr" else {}

    backend = Stub() if args.backend == "stub" else VLLM(args.model, args.gpu_mem) if args.backend == "vllm" else HF(args.model)
    gen_prefix = "" if args.no_empty_think else EMPTY_THINK
    samples = []
    for i, (x, kind) in enumerate(items):
        texts = {k: asr.get(k, v) for k, v in lm.gold_texts(x).items()}
        samples.append({"idx": i, "item": x, "kind": kind, "env": ToolEnv("train", x["id"], root=DATA),  # exactly one env per sample (R-ENV-2)
                        "msgs": lm.build_messages(x, texts, catalog), "raw": [], "stop": None, "retried": False, "seconds": 0.0})

    t0 = time.perf_counter()
    if args.sequential:
        for s in samples:
            run([s], backend, gen_prefix, catalog)
    else:
        run(samples, backend, gen_prefix, catalog)
    wall = time.perf_counter() - t0

    out = Path(ROOT / "student/runs" / args.name)
    out.mkdir(parents=True, exist_ok=True)
    scored = []
    for s in samples:
        x = s["item"]
        calls = s["env"].get_tool_calls()  # unedited (R-ENV-3)
        gold = x["label"]["tool_calls"]
        ok = s["env"].get_state_hash() == gold_hash(x)
        sc = lm.score(calls, gold, "h" if ok else "p", "h", catalog)
        scored.append({"id": x["id"], "kind": s["kind"], "score": sc, "hash_ok": ok, "stop": s["stop"], "calls": calls, "gold": gold,
                       "raw": s["raw"], "seconds": round(s["seconds"], 3)})
    with open(out / "scored.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mean = lambda v: round(100 * sum(v) / len(v), 2) if v else None
    by = lambda k: [r["score"] for r in scored if r["kind"] == k]
    nonempty = [r for r in scored if r["gold"]]
    summary = {"name": args.name, "model": args.model, "backend": args.backend, "texts": args.texts, "rows": len(scored),
               "empty_think": not args.no_empty_think, "sequential": args.sequential,
               "score": mean([r["score"] for r in scored]), "unseen": mean(by("unseen")), "seen": mean(by("seen")),
               "hash_ok_nonempty": mean([r["hash_ok"] for r in nonempty]),
               "ask_rate": mean([any(c["name"] == lm.ASK for c in r["calls"]) for r in scored]),
               "empty_rate": mean([not r["calls"] for r in scored]), "gold_empty_rate": mean([not r["gold"] for r in scored]),
               "calls_per_sample": round(sum(len(r["calls"]) for r in scored) / len(scored), 2),
               "gold_calls_per_sample": round(sum(len(r["gold"]) for r in scored) / len(scored), 2),
               "stops": dict(Counter(r["stop"] for r in scored)),
               "seconds_per_sample": {"mean": round(sum(r["seconds"] for r in scored) / len(scored), 2), "max": max(r["seconds"] for r in scored)},
               "wall_seconds": round(wall, 1)}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
