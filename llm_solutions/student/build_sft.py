"""E1: build the student SFT set from train gold and audit it (CPU only, no model weights).

Format = student/lifemate.py, the Qwen3.8 notebooks' version (plain ChatML, JSON action per assistant turn, no think block), so the data
matches what lifemate's agent loop sends at inference. Per row and transcript variant:
  messages = build_messages(row, texts) + gold_steps(row); loss only on the assistant turns after the final request.
Variants: "gold" = audio_text; "asr" = stock (not fine-tuned) Gipformer transcripts of every clip
(asr_zipformer/asr_base_train.json, train audio only: F-4), so the student also sees realistic ASR errors.
Sources: train split only (R-RES-3, F-2). No *_environment access (R-ENV-1): tool results come from the gold label.

Checks: every assistant target parses back with lifemate.parse_action to the gold call (or None for done);
the labelled tokens decode back to exactly those targets; token lengths per tokenizer vs the 16,384 limit (R-SIZE-2).

Usage: python student/build_sft.py [--out student/data]
"""
import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student"))  # lifemate.py copied from the Qwen3.8 notebooks (cell 3)
import lifemate as lm  # noqa: E402

DATA = ROOT / "train+public_test"
TOKENIZERS = {"Qwen3.5-4B": ROOT / "models/tokenizers/Qwen3.5-4B", "Qwen3-4B-Instruct-2507": ROOT / "models/tokenizers/Qwen3-4B-Instruct-2507"}
MAX_CTX = 16384  # R-SIZE-2
SEED = 0  # R-REP-1


def asr_texts(row, asr):
    """Stock-Gipformer transcript per clip; gold text for a clip it lacks (counted)."""
    gold = lm.gold_texts(row)
    return {k: asr.get(k, v) for k, v in gold.items()}, sum(k not in asr for k in gold)


def examples(rows, catalog, asr, split, rng):
    out, missing = [], 0
    for x in rows:
        variants = [("gold", lm.gold_texts(x))]
        texts, miss = asr_texts(x, asr)
        missing += miss
        if texts != variants[0][1]:
            variants.append(("asr", texts))
        for name, t in variants:
            prompt = lm.build_messages(x, t, catalog, rng if split == "train" else None)  # shuffled tools only for training
            out.append({"id": x["id"], "variant": name, "split": split, "first_target": len(prompt), "messages": prompt + lm.gold_steps(x)})
    return out, missing


def check_targets(ex, row):
    """Each assistant target after the final request parses to the gold call, in order; the last one is done."""
    targets = [m["content"] for m in ex["messages"][ex["first_target"]:] if m["role"] == "assistant"]
    gold = row["label"]["tool_calls"]
    assert len(targets) == len(gold) + 1, ex["id"]
    for t, g in zip(targets, gold):
        assert lm.parse_action(t) == (g["name"], g["arguments"]), (ex["id"], t)
    assert lm.parse_action(targets[-1]) is None and json.loads(targets[-1]) == {"done": True}, ex["id"]
    return targets


def check_labels(tok, enc, targets):
    """Labelled tokens decode to the targets, each followed by <|im_end|>, and nothing else is labelled."""
    ids, labels = enc["input_ids"], enc["labels"]
    runs, cur = [], []
    for i, l in zip(ids, labels):
        if l != -100:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    got = [tok.decode(r) for r in runs]
    want = [t + lm.STOP for t in targets]
    assert got == want, (got[:2], want[:2])


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "student/data"))
    args = ap.parse_args()
    random.seed(SEED)  # R-REP-1 (the tool shuffle uses its own seeded rng below)
    rng = random.Random(SEED)

    rows, catalog = lm.load_split(DATA, "train"), lm.load_catalog(DATA, "train")
    by_id = {x["id"]: x for x in rows}
    with open(ROOT / "asr_zipformer/asr_base_train.json", encoding="utf-8") as f:
        asr = json.load(f)
    train, unseen, seen = lm.split_rows(rows)

    data, missing = [], 0
    for split, part in (("train", train), ("val_unseen", unseen), ("val_seen", seen)):
        ex, m = examples(part, catalog, asr, split, rng)
        data += ex
        missing += m
    for ex in data:
        check_targets(ex, by_id[ex["id"]])

    from transformers import AutoTokenizer

    stats = {"rows": {"train": len(train), "val_unseen": len(unseen), "val_seen": len(seen)}, "asr_clips_missing": missing,
             "examples": dict(collections.Counter((e["split"], e["variant"]) for e in data).items()), "tokenizers": {}}
    stats["examples"] = {f"{s}/{v}": n for (s, v), n in stats["examples"].items()}
    stats["targets"] = {"assistant_steps_train": sum(len(by_id[e["id"]]["label"]["tool_calls"]) + 1 for e in data if e["split"] == "train"),
                        "calls_train": sum(len(by_id[e["id"]]["label"]["tool_calls"]) for e in data if e["split"] == "train")}
    for name, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(path)
        total, labelled, prompt, sys_len, worst = [], [], [], [], None
        for ex in data:
            enc = lm.encode(tok, ex["messages"], lm.targets(ex["messages"], ex["first_target"], False))
            check_labels(tok, enc, check_targets(ex, by_id[ex["id"]]))
            n, k = len(enc["input_ids"]), sum(l != -100 for l in enc["labels"])
            total.append(n)
            labelled.append(k)
            prompt.append(len(tok(lm.prompt_text(ex["messages"][:ex["first_target"]]), add_special_tokens=False)["input_ids"]))
            sys_len.append(len(tok(ex["messages"][0]["content"], add_special_tokens=False)["input_ids"]))
            if worst is None or n > worst[0]:
                worst = (n, ex["id"], ex["variant"])
        d = lambda v: {"mean": round(statistics.mean(v)), "p50": pct(v, .5), "p90": pct(v, .9), "p99": pct(v, .99), "max": max(v)}
        stats["tokenizers"][name] = {"vocab": len(tok), "total": d(total), "first_prompt": d(prompt), "system": d(sys_len),
                                     "labelled": d(labelled), "labelled_sum_train": sum(k for k, e in zip(labelled, data) if e["split"] == "train"),
                                     "over_8192": sum(n > 8192 for n in total), f"over_{MAX_CTX}": sum(n > MAX_CTX for n in total),
                                     "longest": worst}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val_unseen", "val_seen"):
        with open(out / f"sft_{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                if ex["split"] == split:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
