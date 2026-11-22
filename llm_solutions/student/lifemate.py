"""LifeMate shared code: data split, prompt format, action parser, scorer.
02 trains on this file and saves it next to the model; 03 imports that copy, so the prompt always matches the weights.
Self-check: python lifemate.py <data root>"""
import collections
import json
import random
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

ASK = "sys.ask_missing_info"
BOS, STOP = "", "<|im_end|>"
ROLES = {r: f"<|im_start|>{r}\n" for r in ("system", "user", "assistant")}  # plain ChatML: Qwen templates add a think block only to the last assistant turn, which would make training and inference differ
GEN = ROLES["assistant"]
MAX_RESULT_CHARS = 1500  # ponytail: hard cut on long tool results, summarise per tool if truncation ever hides an ID
HOLDOUT_APPS = {"door_lock", "learning", "library", "loyalty", "subscriptions", "household_chores", "home_security"}  # train-only apps: stands in for the unseen apps of the test splits
RELATIVE = {0: "hôm nay", 1: "ngày mai", 2: "ngày kia"}
WEEKS = ["tuần này", "tuần sau", "tuần sau nữa"]  # Monday-start weeks: train gold reads "thứ bảy tuần sau" as next week's Saturday
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
RULES = """Bạn là trợ lý giọng nói tiếng Việt. Hãy xử lý yêu cầu CUỐI CÙNG của người dùng bằng cách gọi tool, người dùng có thể đổi ý giữa chừng.
Mỗi lượt trả về đúng một dòng JSON, không viết gì khác:
- Gọi tool: {"name": "<app.tool>", "arguments": {...}}
- Kết thúc: {"done": true}
Quy tắc:
- Chỉ gọi tool trong danh sách, đúng tên và đúng kiểu tham số. Không tự đoán ID, hãy đọc bằng tool trước khi ghi. ID đã có trong hội thoại thì dùng lại, không đọc lại.
- Chỉ gọi tool thật sự cần. Đọc xong mà yêu cầu cần ghi thì phải ghi tiếp, đừng dừng ở bước đọc.
- Tham số không bắt buộc: chỉ điền khi người dùng nói rõ, không tự điền giá trị mặc định. Tool đọc không thêm bộ lọc nếu người dùng không nêu.
- Yêu cầu có điều kiện ("nếu mai mưa thì..."): đọc điều kiện trước, chỉ làm nhánh đúng. Điều kiện sai thì {"done": true}.
- Tìm thấy đúng một kết quả phù hợp thì dùng luôn. Không tìm thấy, nhiều kết quả không phân biệt được, hoặc thiếu tham số bắt buộc: làm xong các việc khác rồi gọi sys.ask_missing_info với tool_name là tool định gọi và params là tên các tham số còn thiếu, sau đó kết thúc.
- Không app nào làm được, người dùng tự huỷ ("à thôi khỏi"), phủ định ("đừng tắt"), hoặc chỉ trò chuyện: {"done": true} ngay, không gọi tool.
- Thời gian dùng ISO-8601 đủ giây và múi giờ, ví dụ 2026-11-11T05:30:00+07:00. Tra ngày trong bảng 14 ngày tới, không tự tính.
- Số tiền là số nguyên đồng (2 triệu = 2000000). Số đọc từng chữ số thì chép đúng thứ tự, số điện thoại giữ dạng chuỗi.
- Văn bản tự do (tin nhắn, tiêu đề, ghi chú): viết lại ngắn gọn ý chính, giữ dấu tiếng Việt, tên riêng và con số. Từ khoá tìm kiếm chỉ là tên hoặc cụm từ người dùng nhắc."""


def load_split(root, split):
    with open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_catalog(root, split):
    with open(Path(root) / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return {t["name"]: t for a in json.load(f)["apps"] for t in a["tools"]}


def used_apps(x):
    return {(c["arguments"]["tool_name"] if c["name"] == ASK else c["name"]).split(".")[0] for c in x["label"]["tool_calls"]}


def clips(x):
    """Audio paths of a sample: the final request plus every user turn of the history."""
    return [x["audio"]] + [t["audio"] for t in x["history"] if t["role"] == "user"]


def gold_texts(x):
    return {x["audio"]: x["audio_text"]} | {t["audio"]: t["audio_text"] for t in x["history"] if t["role"] == "user"}


def split_rows(rows):
    """(train, val_unseen_apps, val_seen_apps). Final run: train on all rows instead."""
    train, unseen, seen = [], [], []
    for i, x in enumerate(rows):
        (unseen if used_apps(x) & HOLDOUT_APPS else seen if i % 12 == 0 else train).append(x)
    return train, unseen, seen


def render_tool(t):
    """Keeps kind, returns and x-ref: they say what is a read and which read supplies an ID, which an unseen app needs."""
    p = t["parameters"]
    o = {"name": t["name"], "kind": t["kind"], "description": t["description"]}
    if p.get("properties"):
        o["parameters"] = {k: {a: b for a, b in v.items() if a != "x-match"} for k, v in p["properties"].items()}
    for k in ("required", "x-one-of"):
        if p.get(k):
            o[k] = p[k]
    if t.get("returns"):
        o["returns"] = t["returns"]
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def system_prompt(ctx, tools):
    """Context stays in the system prompt, not beside the request as in Spark ver3: history calls are training targets too, and they need the clock."""
    now = datetime.fromisoformat(ctx["current_time"])
    days = ", ".join(f"{WEEKDAYS[d.weekday()]} {d:%Y-%m-%d} ({', '.join(filter(None, [RELATIVE.get(i), WEEKS[(now.weekday() + i) // 7]]))})" for i, d in enumerate(now + timedelta(i) for i in range(14)))
    return (f"{RULES}\n\nBây giờ: {ctx['current_time']} ({WEEKDAYS[now.weekday()]})\n14 ngày tới: {days}\n"
            f"Vị trí: {json.dumps(ctx.get('location'), ensure_ascii=False)}\n"
            f"Người dùng: {json.dumps(ctx.get('user_profile'), ensure_ascii=False)}\n\nDanh sách tool:\n" + "\n".join(render_tool(t) for t in tools))


def asst(obj):
    return {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)}


def result_msg(result):
    return {"role": "user", "content": "KẾT QUẢ: " + json.dumps(result, ensure_ascii=False)[:MAX_RESULT_CHARS]}


def build_messages(item, texts, catalog, rng=None):
    """texts: {audio path: transcript}. rng shuffles the tools, so the model cannot learn a position. Roles strictly alternate."""
    ctx = item["context"]
    names = [t for a in ctx["installed_apps"].values() for t in a["tools"]] + [ASK]
    if rng:
        rng.shuffle(names)
    msgs = [{"role": "system", "content": system_prompt(ctx, [catalog[n] for n in names])}]
    for turn in item["history"]:
        if turn["role"] == "user":
            msgs.append({"role": "user", "content": texts[turn["audio"]]})
            continue
        results = turn.get("tool_results") or []
        for i, c in enumerate(turn.get("tool_calls") or []):
            msgs += [asst({"name": c["name"], "arguments": c["arguments"]}), result_msg(results[i] if i < len(results) else {"ok": True})]
        msgs.append(asst({"done": True, "text": turn.get("text", "")}))
    msgs.append({"role": "user", "content": texts[item["audio"]]})
    return msgs


def gold_steps(x):
    """Gold trajectory after the final user turn: call, result, ..., done."""
    msgs = []
    for c, r in zip(x["label"]["tool_calls"], x["label"]["tool_results"]):
        msgs += [asst({"name": c["name"], "arguments": c["arguments"]}), result_msg(r)]
    return msgs + [asst({"done": True})]


def parse_action(text):
    """(name, arguments), or None for done / unparsable output (both end the turn)."""
    try:
        o = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
    except ValueError:
        return None
    if not isinstance(o.get("name"), str):
        return None
    return o["name"], o["arguments"] if isinstance(o.get("arguments"), dict) else {}


def tokens(s):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(s).lower()))  # word tokens, so "nhé." equals "nhé". The official tokeniser is unconfirmed


def token_f1(a, b):
    a, b = tokens(a), tokens(b)
    c = sum((collections.Counter(a) & collections.Counter(b)).values())
    return 2 * c / (len(a) + len(b)) if c else 0.0


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # type-strict: 1, 1.0 and true all differ


def call_match(p, g, catalog):
    pa, ga = p.get("arguments") or {}, g.get("arguments") or {}
    if p["name"] != g["name"] or set(pa) != set(ga):
        return False
    props = catalog.get(g["name"], {}).get("parameters", {}).get("properties", {})
    return all(token_f1(pa[k], v) > 0.6 if props.get(k, {}).get("x-match") == "fuzzy" else same(pa[k], v) for k, v in ga.items())


def matches(pred, gold, catalog):
    """Size of the maximum one-to-one matching (Kuhn), which a greedy pass can undercount."""
    edges = [[j for j, g in enumerate(gold) if call_match(p, g, catalog)] for p in pred]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(len(pred)))


def score(pred, gold, pred_hash, gold_hash, catalog):
    """Official formula from the description."""
    if not gold:
        return float(not pred)
    if pred_hash != gold_hash:
        return 0.0
    m = matches(pred, gold, catalog)
    if not m:
        return 0.0
    precision, recall = m / len(pred), m / len(gold)
    return 2 * precision * recall / (precision + recall)


def render(msgs):
    return "".join(f"{ROLES[m['role']]}{m['content']}{STOP}\n" for m in msgs)


def prompt_text(msgs):
    """Inference prompt, rendered here so that training and inference always see the same text."""
    return BOS + render(msgs) + GEN


def targets(msgs, first_target, history_calls):
    """Indices of the assistant turns to learn: every turn from first_target, plus the tool calls of history turns if history_calls.
    History replies ({"done": true, "text": ...}) stay unlabelled, since the final turn never writes text."""
    return {i for i, m in enumerate(msgs) if m["role"] == "assistant" and (i >= first_target or history_calls and m["content"].startswith('{"name"'))}


def mask_apps(msgs, apps, rng):
    """Training only: every installed app gets a made-up name in every turn, so the model has to read the tool list instead of
    remembering app names (function masking, as in Hammer, arXiv 2410.04587). Train results and transcripts never hold `app.`, checked."""
    new = {a: "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(4, 9))) for a in apps}
    pat = re.compile(rf"(?<![\w@.])({'|'.join(map(re.escape, new))})\.(?=[a-z_])")
    return [m | {"content": pat.sub(lambda g: new[g.group(1)] + ".", m["content"])} for m in msgs]


def encode(tok, msgs, labelled):
    """Tokenise one conversation, labelling only the assistant turns whose index is in labelled."""
    text, spans = BOS, []
    for i, m in enumerate(msgs):
        start = len(text) + len(ROLES[m["role"]])
        text += render([m])
        if i in labelled:
            spans.append((start, len(text) - 1))  # content + end token, without the newline
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    labels = [t if any(a <= s and e <= b for a, b in spans) else -100 for t, (s, e) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def demo(root):
    sys.path.insert(0, str(Path(root) / "train"))
    from zaic_tools import ToolEnv

    rows, catalog = load_split(root, "train"), load_catalog(root, "train")
    for x in rows[:200]:  # gold replay must score full marks, otherwise the scorer or the data path is broken
        env = ToolEnv("train", x["id"], root=root)  # zac-allow: R-ENV-3 self-check gold replay, never submitted
        gold = x["label"]["tool_calls"]
        results = [env.call(c["name"], c["arguments"]) for c in gold]
        assert results == x["label"]["tool_results"], x["id"]
        assert score(env.get_tool_calls(), gold, "h", "h", catalog) == 1.0, x["id"]

    g = [{"name": "sms.send", "arguments": {"contact_id": "c_1", "content": "mai họp lúc chín giờ"}}]
    near = [{"name": "sms.send", "arguments": {"contact_id": "c_1", "content": "Mai họp lúc chín giờ nhé"}}]
    extra = g + [{"name": "contacts.search", "arguments": {"query": "x"}}]
    wrong_id = [{"name": "sms.send", "arguments": {"contact_id": "c_2", "content": "mai họp lúc chín giờ"}}]
    assert score(near, g, "a", "a", catalog) == 1.0
    assert score(g, g, "a", "b", catalog) == 0.0
    assert score(wrong_id, g, "a", "a", catalog) == 0.0
    assert score([], g, "a", "a", catalog) == 0.0
    assert score(g, [], "a", "a", catalog) == 0.0
    assert score([], [], "x", "y", catalog) == 1.0
    assert abs(score(extra, g, "a", "a", catalog) - 2 / 3) < 1e-9
    assert token_f1("mai họp nhé.", "Mai họp, nhé") == 1.0
    fz = {"t.x": {"parameters": {"properties": {"q": {"x-match": "fuzzy"}}}}}
    assert not call_match({"name": "t.x", "arguments": {"q": "a b", "n": 1}}, {"name": "t.x", "arguments": {"q": "a b"}}, fz)  # extra key
    assert not call_match({"name": "t.x", "arguments": {"n": 1}}, {"name": "t.x", "arguments": {"n": True}}, fz)  # type
    q = [{"name": "t.x", "arguments": {"q": s}} for s in ("một hai ba bốn", "hai ba năm sáu", "một hai ba năm")]
    assert matches(q[:2], [q[2], q[0]], fz) == 2  # the first pred fits both golds, the second only the first: greedy finds 1

    assert parse_action('{"name": "a.b", "arguments": {"x": 1}} rác') == ("a.b", {"x": 1})
    assert parse_action('rác {"name": "a.b"}') == ("a.b", {})
    assert parse_action('{"done": true}') is None
    assert parse_action("không có json") is None
    assert parse_action('{"name": ') is None

    sp = system_prompt({"current_time": "2026-09-24T09:00:00+07:00"}, [])  # a Thursday
    assert "Thứ Năm 2026-09-24 (hôm nay, tuần này)" in sp and "Chủ Nhật 2026-09-27 (tuần này)" in sp and "Thứ Bảy 2026-10-03 (tuần sau)" in sp, sp

    train, unseen, seen = split_rows(rows)
    assert len(train) + len(unseen) + len(seen) == len(rows)
    assert not any(used_apps(x) & HOLDOUT_APPS for x in train)
    for x in rows:
        msgs = build_messages(x, gold_texts(x), catalog) + gold_steps(x)
        roles = [m["role"] for m in msgs[1:]]
        assert all(a != b for a, b in zip(roles, roles[1:])), x["id"]
        first = len(msgs) - 2 * len(x["label"]["tool_calls"]) - 1
        assert msgs[first - 1]["content"] == x["audio_text"], x["id"]
        hist = [c for t in x["history"] for c in t.get("tool_calls") or []]
        assert len(targets(msgs, first, True)) == len(hist) + len(x["label"]["tool_calls"]) + 1 and len(targets(msgs, first, False)) == len(x["label"]["tool_calls"]) + 1, x["id"]
        m = mask_apps(msgs, x["context"]["installed_apps"], random.Random(0))
        called = [json.loads(t["content"])["name"] for t in m[first:] if t["role"] == "assistant" and t["content"].startswith('{"name"')]
        assert all(f'"name":"{n}"' in m[0]["content"] for n in called) and not any(f'"name":"{a}.' in m[0]["content"] for a in x["context"]["installed_apps"]), x["id"]
    print("lifemate self-check ok:", len(train), "train,", len(unseen), "unseen-app val,", len(seen), "seen-app val")


if __name__ == "__main__":
    demo(sys.argv[1])