"""Write one short Vietnamese reason for every gold step of the train split, locally: no API, no model, train labels only.

Output rationales.json: {sample id: [reason, ...]} in the order the LLM notebook labels steps: every history tool call,
then every final tool call, then the final {"done": true}. Upload it to Kaggle beside the data and set WHY = True in 02c ver3.

Reasons are built from the gold call, the catalog and the results of earlier steps. Their value is grounding:
an ID argument is traced to the earlier result that holds it, with that item's name.
ponytail: templates cannot explain a skipped branch of a conditional; a local open model could, if these prove too thin.

python gen_rationales.py            writes rationales.json and prints a few samples"""
import json
from pathlib import Path

ROOT = Path(r"D:\ZOLA\train+public_test\train+public_test")
OUT = Path(__file__).with_name("rationales.json")
LABELS = ("name", "title", "content", "label", "subject", "route", "description", "type")


def find_item(obj, value):
    """The first dict anywhere in obj whose id is value."""
    if isinstance(obj, dict):
        if obj.get("id") == value:
            return obj
        obj = list(obj.values())
    if isinstance(obj, list):
        for o in obj:
            if (hit := find_item(o, value)) is not None:
                return hit
    return None


def describe(item):
    return next((str(item[k]) for k in LABELS if isinstance(item.get(k), (str, int, float)) and str(item[k]).strip()), "")


def reason(call, earlier, catalog):
    """earlier: [(tool name, result)] of every step before this one, history included."""
    name, args = call["name"], call["arguments"]
    if name == "sys.ask_missing_info":
        return f"Thiếu {', '.join(args.get('params', []))} cho {args.get('tool_name')}, không tra được nên hỏi lại người dùng."
    tool = catalog.get(name, {})
    parts = []
    for k, v in args.items():
        if not (k.endswith("_id") or k.endswith("_ids")) or not isinstance(v, (str, list)):
            continue
        for one in v if isinstance(v, list) else [v]:
            src = next(((n, it) for n, r in reversed(earlier) if (it := find_item(r, one)) is not None), None)
            if src:
                label = describe(src[1])
                parts.append(f"{one}{f' ({label})' if label else ''} lấy từ kết quả {src[0]}")
            else:
                parts.append(f"{one} đã có trong hội thoại")
    what = tool.get("description", name).rstrip(".")
    if parts:
        return f"{what}: " + "; ".join(parts) + "."
    if tool.get("kind") == "read":
        return f"Cần đọc trước: {what}."
    return f"Người dùng yêu cầu: {what}."


def sample(x, catalog):
    out, earlier = [], []
    for t in x["history"]:
        for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or []):
            out.append(reason(c, earlier, catalog))
            earlier.append((c["name"], r))
    for c, r in zip(x["label"]["tool_calls"], x["label"]["tool_results"]):
        out.append(reason(c, earlier, catalog))
        earlier.append((c["name"], r))
    out.append("Đã làm xong mọi việc người dùng yêu cầu." if x["label"]["tool_calls"] else "Không cần hoặc không thể gọi tool nào cho yêu cầu này.")
    return out


def expected(x):
    """Same count as 02c's targets() with HISTORY_CALLS: history calls + final calls + done."""
    return sum(len(t.get("tool_calls") or []) for t in x["history"]) + len(x["label"]["tool_calls"]) + 1


def main():
    rows = [json.loads(l) for l in open(ROOT / "train" / "train.jsonl", encoding="utf-8") if l.strip()]
    catalog = {t["name"]: t for a in json.load(open(ROOT / "train" / "train_tools_catalog.json", encoding="utf-8"))["apps"] for t in a["tools"]}
    whys = {x["id"]: sample(x, catalog) for x in rows}
    for x in rows:
        assert len(whys[x["id"]]) == expected(x), x["id"]  # every history call has its result in train
    json.dump(whys, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    n = sum(map(len, whys.values()))
    print(f"{len(whys)} samples, {n} reasons, mean {sum(len(w) for v in whys.values() for w in v) / n:.0f} characters -> {OUT}")
    for i in ("train_1590", "train_0074", "train_1694"):
        print(i, json.dumps(whys[i], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
