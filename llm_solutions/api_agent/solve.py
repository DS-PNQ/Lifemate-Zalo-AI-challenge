"""Gemini (ASR, via transcripts.json) + free LLM (tool calling) agent for ZAC 2026 LifeMate — dev/teacher only.

Independent of the other folders: reads only the dataset, ../train+public_test, and files in this folder.

Design
  * ONE sample per conversation; native OpenAI-style tool calling in a real loop:
    model calls tools -> env.call -> results fed back -> ... until the model stops calling tools.
  * env.validate() first: schema-invalid calls are bounced back to the model and never logged (no precision loss).
  * Identical repeated calls are served from cache and not logged twice.
  * Full history: earlier user turns as text, earlier assistant turns with their tool calls/results.
  * Few-shot: nearest TRAIN examples (gold calls) by utterance similarity.
  * Switchable backend, ONE key per run (no rotation across accounts), except ifm:
        --backend openrouter  (default model nvidia/nemotron-3-ultra-550b-a55b:free)
        --backend nim         (NVIDIA build.nvidia.com; pass --model with the NIM model id)
        --backend deepseek    (DeepSeek V4.1 Flash, paid: --spend-file + --budget stop new samples at a USD cap)
        --backend gwai        (DeepSeek V4.1 Flash via the 1gw.gwai.cloud reseller gateway, Anthropic Messages format)
        --backend ifm         (IFM K2 Horizon, api.ifm.ai; key IFM_API_KEY; several keys rotate: the next one takes over when one hits its 10M tokens/day cap or is rejected;
                              stops when the last one does)
  * Stops cleanly when the daily free quota is used up; rerun the same command later to resume.
  * Empty API replies (HTTP 200, no tool calls, no text) are re-asked; if the first step stays empty the sample
    is NOT cached (no ToolEnv is created) and reruns on the next pass. Old cached empty replies are ignored.
  * Argument habits of train gold (habits/review): optional arguments gold almost never fills are dropped from
    reads and bounced once on writes; ones gold always fills are asked for once. Shown to the model as hints too.
  * A read tool runs at most READ_CAP times per sample (gold: 3 or fewer in all but 1 of 1545 rows).
  * Thought traces: one line per sample in --traces (every step: thought, text, each call with status and result).
  * On train: local score with the official formula.  On public: submission.jsonl; samples not solved yet
    are filled from --base-submission (e.g. your best leaderboard file) so a partial run is still submittable.

  py solve.py --split train --sample 8                      # quick quality check on train
  py solve.py --split public_test                           # resumable public run
  py solve.py --split public_test --base-submission "../gemi + nemo/submission.jsonl"
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "train+public_test"

BACKENDS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "prefix": "sk-or-",
                   "model": "nvidia/nemotron-3-ultra-550b-a55b:free"},
    "nim": {"url": "https://integrate.api.nvidia.com/v1/chat/completions", "prefix": "nvapi-", "model": None},
    # DeepSeek V4.1 Flash. USD per 1M tokens at PEAK rates (off-peak is half), so the budget stop errs on the safe side.
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "prefix": "sk-", "model": "deepseek-flash",
                 "price": {"hit": 0.006, "miss": 0.30, "out": 1.20}},
    # Reseller gateway for DeepSeek V4.1 Flash (genzshop.vn keys). It serves only the Anthropic Messages format
    # (/v1/chat/completions answers 403 "blocked"). Its own prices are unknown: DeepSeek's peak prices as an estimate.
    "gwai": {"url": "https://1gw.gwai.cloud/v1/messages", "prefix": "sk-", "model": "deepseek-v4.1-flash-1017",
             "format": "anthropic", "price": {"hit": 0.006, "miss": 0.30, "out": 1.20}},
    # IFM K2 Horizon (docs.ifm.ai): OpenAI format, reasoning always on ("high" only), 10M tokens/day per key, no prices.
    "ifm": {"url": "https://api.ifm.ai/v1/chat/completions", "prefix": None, "model": "IFM/K2-Horizon-375B-A23B"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train", choices=["train", "public_test"])
ap.add_argument("--backend", default="openrouter", choices=list(BACKENDS))
ap.add_argument("--model", default=None)
ap.add_argument("--key-index", type=int, default=0,
                help="which key of this backend to use (ifm: the first one; it moves on to the next when a key runs out)")
ap.add_argument("--sample", type=int, default=None, help="random N samples (seed 0)")
ap.add_argument("--ids", default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shots", type=int, default=5)
ap.add_argument("--max-steps", type=int, default=10)
ap.add_argument("--reasoning", default="medium",
                help="low/medium/high (deepseek, gwai also max), or 'off' (OpenRouter, DeepSeek, gwai only)")
ap.add_argument("--spend-file", default=None,
                help="paid backends: running USD total of every request, shared by all runs that pass the same file")
ap.add_argument("--budget", type=float, default=None,
                help="USD: start no new samples once the --spend-file total reaches this (running ones finish)")
ap.add_argument("--base-submission", default=None)
ap.add_argument("--cache", default=None)
ap.add_argument("--holdout-apps", default=None,
                help="train only, comma list: run the rows that use these apps, with examples and argument habits "
                     "taken only from rows that do not (stands in for apps the test split has and train lacks)")
ap.add_argument("--self-check", action="store_true", help="offline asserts on the argument-habit logic, then exit")
ap.add_argument("--why", action="store_true",
                help="train only (SFT data): one tool per turn, with one short written reason before each call")
ap.add_argument("--traces", default=None, help="thought traces, one JSON line per sample (default: cache name, cache -> traces)")
ap.add_argument("--workers", type=int, default=1, help="samples solved in parallel (each still has its own ToolEnv)")
ap.add_argument("--text-source", default="gold", choices=["gold", "asr"],
                help="train only: gold audio_text, or ASR transcripts from --transcripts")
ap.add_argument("--transcripts", default=None,
                help="ASR transcripts JSON (keys <id>, <id>_u<n>). public: primary transcripts (default ./transcripts.json). train: the run is restricted to "
                     "samples present in it (for both text sources, so gold vs asr compare the same samples)")
ap.add_argument("--stop-file", default=None,
                help="if this file appears, start no new samples, let running ones finish, write output and exit")
ap.add_argument("--out", default=None, help="public: where to write the submission (default ./submission.jsonl)")
ap.add_argument("--extra-transcripts", nargs="*", default=[],
                help="other ASR versions of the same audio (same key format); the model sees all distinct versions")
A = ap.parse_args()

sys.path.insert(0, str(DATA / A.split))
from zaic_tools import ToolEnv  # noqa: E402
from zaic_tools.utils import _tokens, _canon_scalar  # noqa: E402

BK = BACKENDS[A.backend]
MODEL = A.model or BK["model"]
if not MODEL:
    sys.exit("--model is required for this backend")


def load_keys():
    """Keys of this backend from --key-index on. ifm rotates through all of them (chat() moves to the next key when one
    is refused or hits its daily cap); every other backend uses only the first."""
    raw = os.environ.get({"nim": "NVIDIA_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                          "deepseek": "DEEPSEEK_API_KEY", "gwai": "GWAI_API_KEY", "ifm": "IFM_API_KEY"}[A.backend], "").strip()
    if raw:  # e.g. Colab/Kaggle secrets; keeps keys out of files. Several keys: separated by commas, spaces or newlines
        keys = raw.replace(",", " ").split()[A.key_index:]
        if not keys:
            sys.exit(f"no {A.backend} key #{A.key_index} in the environment variable")
        return keys if A.backend == "ifm" else keys[:1]
    keys = []
    for line in open(HERE / "api_keys.txt", encoding="utf-8"):
        s, label = line.strip(), ""
        if s.upper().startswith(("OPENROUTER:", "NVIDIA:", "NIM:", "DEEPSEEK:", "GWAI:", "IFM:")):
            label, s = s.split(":", 1)
            label, s = label.upper(), s.strip()
        if A.backend in ("gwai", "ifm"):  # no reliable key prefix: only lines labelled GWAI: / IFM:
            ok = label == A.backend.upper()
        else:
            ok = (s.startswith(BK["prefix"]) and label not in ("GWAI", "IFM")
                  and not (A.backend == "deepseek" and s.startswith("sk-or-")))
        if ok:
            keys.append(s)
    if len(keys) <= A.key_index:
        sys.exit(f"no {A.backend} key #{A.key_index} in api_keys.txt "
                 + (f"(a line '{A.backend.upper()}: ...')" if A.backend in ("gwai", "ifm")
                    else f"(keys start with {BK['prefix']})"))
    keys = keys[A.key_index:]
    return keys if A.backend == "ifm" else keys[:1]


KEYS = load_keys()
KEY_AT = [0]  # index of the key in use, shared by all worker threads
KEY_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------------ catalog -> tools
CAT = json.load(open(DATA / A.split / f"{A.split}_tools_catalog.json", encoding="utf-8"))
TOOLS = {t["name"]: t for a in CAT["apps"] for t in a["tools"]}
APP_DESC = {a["id"]: f"{a.get('name_vi', '')} — {a.get('description', '')}" for a in CAT["apps"]}
TOOLS.setdefault("sys.ask_missing_info", {
    "name": "sys.ask_missing_info", "kind": "meta",
    "description": "Hỏi lại người dùng khi thiếu tham số bắt buộc hoặc mơ hồ (vd. nhiều liên hệ trùng tên). "
                   "tool_name = tool định gọi, params = các tham số còn thiếu/mơ hồ.",
    "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"},
                                                    "params": {"type": "array", "items": {"type": "string"}}},
                   "required": ["tool_name", "params"]}})
FUZZY = {(n, k) for n, t in TOOLS.items()
         for k, p in ((t.get("parameters") or {}).get("properties") or {}).items() if p.get("x-match") == "fuzzy"}


def clean(s):
    if isinstance(s, list):
        return [clean(x) for x in s]
    if not isinstance(s, dict):
        return s
    out, hints = {}, []
    for k, v in s.items():
        if k == "x-ref":
            hints.append(f"ID lấy từ {v}")
        elif k == "x-match":
            hints.append("văn bản tự do")
        elif k.startswith("x-"):
            continue
        elif k == "properties":
            out[k] = {pk: clean(pv) for pk, pv in v.items()}
        else:
            out[k] = clean(v)
    if hints:
        out["description"] = (out.get("description", "") + " (" + "; ".join(hints) + ")").strip()
    return out


def fname(n):
    return n.replace(".", "__")


def tool_spec(name):
    t = TOOLS[name]
    params = clean(t.get("parameters") or {})
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    desc = t.get("description", "")
    if (t.get("parameters") or {}).get("x-one-of"):
        desc += f" | Bắt buộc đủ MỘT trong các nhóm: {t['parameters']['x-one-of']}"
    if t.get("returns"):
        desc += f" | Trả về: {t['returns']}"
    desc += f" | kind={t.get('kind')}"
    return {"type": "function", "function": {"name": fname(name), "description": desc[:1000], "parameters": params}}

# ------------------------------------------------------------------------------------------ few-shot
TRAIN = [json.loads(l) for l in open(DATA / "train/train.jsonl", encoding="utf-8") if l.strip()]

# ------------------------------------------------------------------------------------------ argument habits
# How often train gold fills each optional argument (R-RES-3: train gold only, never public text).
RARE, USUAL, READ_CAP = 0.05, 0.95, 3
HAB = {"n": collections.Counter(), "fill": collections.Counter(), "pool": collections.Counter(),
       "pool_n": collections.Counter()}


def optional(t):
    """Optional parameters of a tool, without the x-one-of alternatives (one of those is always needed)."""
    P = t.get("parameters") or {}
    skip = set(P.get("required") or []) | {k for g in P.get("x-one-of") or [] for k in g}
    return [k for k in (P.get("properties") or {}) if k not in skip]


def habits(rows):
    # The Colab bundle has train.jsonl but no train catalog: then pool over the tools this split's catalog knows.
    # Checked on public: the same drop/bounce/hint decision for all 76 optional arguments either way.
    ttools = dict(TOOLS)
    if (DATA / "train/train_tools_catalog.json").is_file():
        cat = json.load(open(DATA / "train/train_tools_catalog.json", encoding="utf-8"))
        ttools = {t["name"]: t for a in cat["apps"] for t in a["tools"]}
    for h in HAB.values():
        h.clear()
    for t in rows:
        for c in t["label"]["tool_calls"]:
            HAB["n"][c["name"]] += 1
            for k in c["arguments"]:
                HAB["fill"][c["name"], k] += 1
    for name, n in HAB["n"].items():  # the same, pooled by (kind, parameter name), for tools gold rarely or never uses
        for k in optional(ttools.get(name, {})):
            HAB["pool"][ttools[name].get("kind"), k] += HAB["fill"][name, k]
            HAB["pool_n"][ttools[name].get("kind"), k] += n


def rate(name, k):
    """(share of gold calls that fill optional argument k, True if from this tool's own calls) or (None, False)."""
    if HAB["n"][name] >= 8:
        return HAB["fill"][name, k] / HAB["n"][name], True
    kind = TOOLS[name].get("kind")
    if HAB["pool_n"][kind, k] >= 25:
        return HAB["pool"][kind, k] / HAB["pool_n"][kind, k], False
    return None, False


def review(name, args):
    """-> (args to run, rare optional args present, usual optional args missing)."""
    rare, missing = [], []
    for k in optional(TOOLS[name]):
        r, own = rate(name, k)
        if r is not None and r <= RARE and k in args:
            rare.append(k)
        elif own and r >= USUAL and k not in args:
            missing.append(k)
    if rare and TOOLS[name].get("kind") == "read":  # a read without the filter returns a superset: just drop it
        return {k: v for k, v in args.items() if k not in rare}, [], missing
    return args, rare, missing


def arg_hints(allowed):
    lines = []
    for name in allowed:
        rs = [(k, rate(name, k)[0]) for k in optional(TOOLS[name])]
        lo = [k for k, r in rs if r is not None and r < 0.25]
        hi = [k for k, r in rs if r is not None and r >= 0.9]  # 0.6 made the model add calendar end where gold has none
        if lo or hi:
            lines.append(f"- {name}: " + "; ".join(x for x in (
                "hiếm khi truyền " + ", ".join(lo) if lo else "", "thường truyền " + ", ".join(hi) if hi else "") if x))
    return "\n".join(lines) or "(không có)"
TRAIN_TOK = [(_tokens(t["audio_text"]), t) for t in TRAIN]


def fmt_history(item, texts, max_chars=None):
    lines = []
    for i, h in enumerate(item.get("history", [])):
        if h["role"] == "user":
            lines.append(f"USER: {texts.get(i) or '(không nghe rõ)'}")
        else:
            lines.append(f"ASSISTANT: {h.get('text', '')}")
            res = h.get("tool_results") or []
            for j, c in enumerate(h.get("tool_calls") or []):
                r = f" -> {json.dumps(res[j], ensure_ascii=False)}" if j < len(res) else ""
                lines.append(f"   [đã gọi] {c['name']} {json.dumps(c.get('arguments', {}), ensure_ascii=False)}{r}")
    s = "\n".join(lines) or "(không có)"
    return s if not max_chars or len(s) <= max_chars else s[:max_chars] + " ..."


def gold_texts(t):
    return {i: h.get("audio_text", "") for i, h in enumerate(t.get("history", [])) if h["role"] == "user"}


def examples(item, utt):
    q, apps = _tokens(utt), set(item["context"]["installed_apps"])
    scored = []
    for toks, t in TRAIN_TOK:
        if t["id"] == item["id"]:
            continue
        used = {c["name"].split(".")[0] for c in t["label"]["tool_calls"]}
        scored.append((len(q & toks) / (len(q | toks) or 1) + (0.05 if used and used <= apps else 0), t))
    scored.sort(key=lambda x: -x[0])
    blocks = []
    for _, t in scored[:A.shots]:
        calls = "\n".join(f"  {c['name']} {json.dumps(c['arguments'], ensure_ascii=False)}"
                          for c in t["label"]["tool_calls"]) or "  (không gọi tool nào)"
        blocks.append(f"--- Ví dụ (thời điểm {t['context']['current_time']})\nLịch sử:\n"
                      f"{fmt_history(t, gold_texts(t), 1200)}\nYêu cầu cuối: {t['audio_text']}\nChuỗi lệnh đúng:\n{calls}")
    return "\n\n".join(blocks)


SYSTEM = """Bạn là trợ lý giọng nói tiếng Việt điều khiển ứng dụng qua tool. Bạn được chấm điểm CHỈ dựa trên chuỗi tool call bạn gọi (so với đáp án) và trạng thái cuối của dữ liệu ứng dụng.

Cách chấm:
- Trạng thái cuối phải khớp đáp án, nếu không mẫu được 0 điểm.
- Mỗi tool call khớp khi đúng tên và đúng tham số (văn bản tự do chỉ cần gần đúng). Gọi THỪA tool (kể cả tool đọc không cần) hoặc THIẾU tool đều bị trừ điểm.

Quy tắc:
1. Chỉ xử lý YÊU CẦU CUỐI. Thao tác trong lịch sử đã làm rồi, KHÔNG làm lại. Dùng lịch sử để hiểu ngữ cảnh/đại từ và lấy ID đã có; ID đã có trong lịch sử thì không đọc lại. Lời người dùng trong lịch sử là bản nhận dạng giọng nói, có thể sai chính tả — đối chiếu với câu trả lời của trợ lý.
2. Không bịa ID. Cần ID chưa biết thì gọi tool đọc tương ứng trước, rồi dùng đúng ID trong kết quả.
3. Tool đọc gọi với tham số tối thiểu (thường là {}), trừ khi cần tham số để tra (vd. contacts.search query=tên). KHÔNG tự thêm bộ lọc không bắt buộc (vd. calendar.list_events date_from/date_to, email.list unread_only, list_transactions limit/account_id, list_alerts since) — đọc hết rồi tự lọc trong đầu. Không đọc dữ liệu không cần.
4. Câu điều kiện ("nếu mai mưa thì..."): đọc dữ liệu cần (hoặc dùng số liệu đã có trong lịch sử), so sánh cẩn thận, rồi CHỈ làm nhánh đúng.
5. Người dùng tự sửa/đổi ý ("à mà thôi", "không phải..."): chỉ làm ý cuối cùng.
6. Thiếu thông tin bắt buộc hoặc mơ hồ: gọi sys.ask_missing_info(tool_name, params). Phần việc khác đủ thông tin thì vẫn làm.
7. Ngoài phạm vi app đã cài, yêu cầu lừa đảo/nguy hiểm, hoặc chỉ trò chuyện: KHÔNG gọi tool nào. Cũng KHÔNG gọi tool khi: người dùng rút lại yêu cầu ("à mà thôi khỏi"); giá trị yêu cầu không hợp lệ hoặc không thể làm (số âm, thời điểm đã qua); hoặc câu trả lời đã có sẵn trong lịch sử.
8. Tham số KHÔNG bắt buộc chỉ truyền khi người dùng NÓI RÕ trong lời yêu cầu. KHÔNG tự điền từ hồ sơ người dùng hay tự suy đoán (vd. place_order address/payment, banking.transfer note, alarm.create label, reminders.create repeat, robot_vacuum.start mode). Mỗi tham số thừa làm cả lệnh bị tính sai. Thời gian ISO 8601 +07:00, suy ra từ thời điểm hiện tại. Văn bản tự do viết ngắn gọn, sát ý người dùng. "Thói quen tham số" ở dưới cho biết đáp án thường/hiếm khi truyền tham số nào.
9. Kiểm tra điều kiện trước lệnh ghi (đủ số dư, còn chỗ, chưa làm rồi...). Lệnh chắc chắn thất bại thì đừng gọi.
10. Tìm không ra (kết quả rỗng): thử nhiều nhất MỘT lần nữa với từ khóa khác (tên có thể bị nghe sai), rồi làm tiếp với dữ liệu đã có hoặc hỏi lại. Không gọi cùng một tool đọc quá 3 lần.
11. Theo phong cách các ví dụ mẫu. Gọi được nhiều tool độc lập cùng lúc thì gọi cùng lúc.
Xong việc thì trả lời một câu ngắn, không gọi thêm tool."""
WHY_RULE = ("Mỗi lượt gọi đúng MỘT tool. Ngay trước tool call, viết MỘT câu ngắn tiếng Việt (tối đa 30 từ) nêu lý do: "
            "dựa vào dữ liệu nào, vì sao chọn tool và tham số này.")
if A.why:  # visible reasons for student SFT data: the gwai gateway now returns only a title as the thinking
    if A.split != "train":
        sys.exit("--why makes training data: train split only (R-RES-3)")
    SYSTEM = SYSTEM.replace("Gọi được nhiều tool độc lập cùng lúc thì gọi cùng lúc.", WHY_RULE)
    assert WHY_RULE in SYSTEM


def user_prompt(item, utt, hist, allowed):
    c = item["context"]
    apps = "\n".join(f"- {a}: {APP_DESC.get(a, '')}" for a in c["installed_apps"])
    return (f"# Ví dụ mẫu (dữ liệu huấn luyện, có thể dùng app khác)\n{examples(item, utt)}\n\n"
            f"# Mẫu hiện tại\nThời điểm hiện tại: {c.get('current_time')} ({c.get('timezone')})\n"
            f"Vị trí: {json.dumps(c.get('location'), ensure_ascii=False)}\n"
            f"Hồ sơ người dùng: {json.dumps(c.get('user_profile'), ensure_ascii=False)}\n"
            f"App đã cài:\n{apps}\n\nThói quen tham số (thống kê từ đáp án huấn luyện):\n{arg_hints(allowed)}\n\n"
            f"Lịch sử hội thoại:\n{fmt_history(item, hist)}\n\n"
            f"{final_request(utt)}\n\nHãy thực hiện bằng cách gọi tool.")


# Measured on train (gold text 98.4 vs transcripts 92.3): lost punctuation made a hesitation or a second request
# part of the first one. A second note on names ("search with the title word") was tried and removed: it fixed
# nothing and made the model search "anh Tuấn" where gold searches "Tuấn".
ASR_NOTE = ("Lưu ý: bản nhận dạng giọng nói thường KHÔNG có dấu câu, hãy tự ngắt câu. 'mà thôi', 'à thôi khỏi' ở cuối "
            "= người dùng rút lại việc vừa nói. 'rồi ...', 'xong ...', 'với ...' thường mở đầu một yêu cầu KHÁC, "
            "không phải phần nội dung của tin nhắn/ghi chú đứng trước nó.")


def final_request(utt):
    if not utt.startswith(MULTI):
        return f"YÊU CẦU CUỐI CỦA NGƯỜI DÙNG: \"{utt}\"\n{ASR_NOTE}"
    vers = utt[len(MULTI):].split("\n")
    return ("YÊU CẦU CUỐI CỦA NGƯỜI DÙNG — có " + str(len(vers)) + " bản nhận dạng giọng nói khác nhau của CÙNG MỘT câu "
            "nói. Bản nào cũng có thể nghe sai, nhất là tên riêng, tên app, số, giờ, số tiền. Hãy đối chiếu các bản với "
            "nhau và với ngữ cảnh (app đã cài, lịch sử, dữ liệu đọc được) để suy ra câu gốc rồi thực hiện:\n"
            + "\n".join(f"- bản {k}: \"{v}\"" for k, v in enumerate(vers, 1)) + f"\n{ASR_NOTE}")


MULTI = "\x00multi\x00"


def merge_versions(texts):
    """Distinct non-empty versions (case/punctuation-insensitive) -> one string (plain if only one is distinct)."""
    seen, out = set(), []
    for t in texts:
        k = " ".join("".join(c if c.isalnum() else " " for c in (t or "").lower()).split())
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip().replace("\n", " "))
    return (MULTI + "\n".join(out)) if len(out) > 1 else (out[0] if out else "")

# ------------------------------------------------------------------------------------------ LLM call
class DailyQuotaExhausted(Exception):
    """Every later request would fail too (quota used up, no balance, key rejected): stop starting samples."""
    pass


class EmptyReply(Exception):
    """The API answered HTTP 200 but the message has neither tool calls nor text (seen on NIM under load:
    532/633 samples of a Colab run, 2026-11-12). Never a real decision, so the sample must not be cached."""


STATS = {"requests": 0, "empty": 0, "usd": 0.0}
SPEND_LOCK = threading.Lock()


def spent():
    """USD spent so far: the --spend-file total at start (earlier runs sharing it) + this run's requests.
    The file is read only once: a read while another thread replaces it fails on Windows (WinError 5)."""
    return SPENT0 + STATS["usd"]


try:
    SPENT0 = float(Path(A.spend_file).read_text()) if A.spend_file else 0.0
except (OSError, ValueError):  # no file yet
    SPENT0 = 0.0


def charge(usage):
    """Adds one request's cost (backend price table x the API's usage counts) to the ledger; returns it in USD."""
    p = BK.get("price")
    if not p or not usage:
        return 0.0
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    usd = (hit * p["hit"] + miss * p["miss"] + usage.get("completion_tokens", 0) * p["out"]) / 1e6
    with SPEND_LOCK:
        STATS["usd"] += usd
        if A.spend_file:
            try:  # atomic: a kill never leaves a half-written total
                Path(A.spend_file + ".tmp").write_text(f"{spent():.6f}")
                os.replace(A.spend_file + ".tmp", A.spend_file)
            except OSError as e:  # e.g. the notebook's status cell reading it: the next request writes the total again
                print(f"  (spend file not updated: {e})", flush=True)
    return usd


EMPTY_BACKOFF = (5, 10, 20, 40)  # re-ask the same step this many times before giving up
UNSUPPORTED = set()  # request fields the endpoint answered HTTP 400 to


def is_empty(m):
    return not m.get("tool_calls") and not (m.get("content") or "").strip()


def to_anthropic(messages, tools):
    """The OpenAI-style chat solve() keeps -> an Anthropic Messages request body. Assistant turns are sent back as the
    API returned them (`_blocks`, thinking blocks included): a tool loop must carry its thinking along."""
    out = []
    for m in messages:
        if m["role"] == "tool":  # all results of one assistant turn go into one user message
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m.get("_blocks") or [{"type": "text", "text": m.get("content") or "."}]})
        elif m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
    body = {"model": MODEL, "max_tokens": 8000, "messages": out,
            "system": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
            "tools": [{"name": t["function"]["name"], "description": t["function"]["description"],
                       "input_schema": t["function"]["parameters"]} for t in tools]}
    if A.reasoning == "off":
        body["thinking"] = {"type": "disabled"}
    else:
        body["output_config"] = {"effort": A.reasoning}
    return body


def from_anthropic(d):
    """Anthropic Messages response -> (message in the OpenAI shape solve() reads, usage in DeepSeek's field names)."""
    blocks = d.get("content") or []
    m = {"content": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
         "reasoning_content": "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking"),
         "tool_calls": [{"id": b["id"], "type": "function",
                         "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                        for b in blocks if b.get("type") == "tool_use"],
         "_blocks": blocks}
    u = d.get("usage")
    usage = u and {"prompt_cache_hit_tokens": u.get("cache_read_input_tokens") or 0,
                   "prompt_cache_miss_tokens": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0),
                   "completion_tokens": u.get("output_tokens") or 0}
    return m, usage


def chat(messages, tools):
    anth = BK.get("format") == "anthropic"
    body = {"model": MODEL, "messages": messages, "tools": tools, "temperature": 0, "seed": 0, "max_tokens": 8000}
    if anth:
        body = to_anthropic(messages, tools)
    if A.backend == "openrouter" and A.reasoning != "off":
        body["reasoning"] = {"effort": A.reasoning}
    if A.backend == "deepseek":  # seed is not documented by DeepSeek; temperature is ignored in thinking mode
        body.pop("seed")
        body["thinking"] = {"type": "disabled" if A.reasoning == "off" else "enabled"}
        if A.reasoning != "off":
            body["reasoning_effort"] = A.reasoning
    if A.backend == "ifm":  # always thinks at "high"; room for the thought plus the answer
        body["max_tokens"] = 16000
    for k in UNSUPPORTED:
        body.pop(k, None)
    for attempt in range(7 + len(KEYS)):  # a key switch also takes an attempt
        ki = KEY_AT[0]
        headers = ({"x-api-key": KEYS[ki], "anthropic-version": "2023-06-01"} if anth
                   else {"Authorization": f"Bearer {KEYS[ki]}"})
        try:
            r = requests.post(BK["url"], headers=headers, json=body, timeout=300)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        STATS["requests"] += 1
        if r.status_code == 200:
            d = r.json()
            if d.get("error"):
                time.sleep(5 * (attempt + 1))
                continue
            m, usage = from_anthropic(d) if anth else (d["choices"][0]["message"], d.get("usage"))
            m["_usd"] = charge(usage)
            return m
        txt = r.text[:400]
        # 401/403 = key rejected or endpoint blocked, 402 = no balance: every later request would fail the same way
        if r.status_code in (401, 402, 403) or (r.status_code == 429 and (
                "per-day" in txt or "per day" in txt or "free-models-per-day" in txt or "daily" in txt.lower()
                or r.headers.get("x-ratelimit-remaining-tokens-daily") == "0")):  # IFM: 10M tokens/day per key
            # shortcut: a spent key stays skipped until the run ends, even past the 00:00 UTC reset; rerun to start over
            with KEY_LOCK:
                if KEY_AT[0] == ki and ki + 1 < len(KEYS):  # only the first thread to see key ki fail moves on
                    KEY_AT[0] = ki + 1
                    print(f"  key {ki + 1} of {len(KEYS)} refused ({r.status_code}): switching to key {ki + 2}", flush=True)
                moved = KEY_AT[0] != ki
            if moved:
                continue
            raise DailyQuotaExhausted(f"{r.status_code} on the last key ({len(KEYS)} of {len(KEYS)}): {txt}")
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(90, 8 * 2 ** attempt) + random.random())
            continue
        dropped = next((k for k in ("reasoning", "output_config") if k in body), None)
        if r.status_code == 400 and dropped:  # the endpoint does not take this effort setting: never send it again
            if dropped not in UNSUPPORTED:
                print(f"  (HTTP 400 with '{dropped}': sent without it from now on. {txt[:150]})", flush=True)
            UNSUPPORTED.add(dropped)
            body.pop(dropped)
            continue
        raise RuntimeError(f"{r.status_code}: {txt}")
    raise RuntimeError("retries exhausted")


def solve(item, utt, hist):
    # The ToolEnv is created only once the first real reply arrives: a sample whose first reply stays empty
    # raises EmptyReply without ever creating an env, so a later retry still uses exactly one env (R-ENV-2).
    env = None
    allowed = [t for a in item["context"]["installed_apps"].values() for t in a["tools"] if t in TOOLS]
    allowed.append("sys.ask_missing_info")
    tools = [tool_spec(n) for n in allowed]
    back = {fname(n): n for n in allowed}
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(item, utt, hist, allowed[:-1])}]
    seen, reasoning, final = {}, [], ""
    trace, warned, reads, stop, usd = [], set(), collections.Counter(), "max_steps", 0.0
    for step in range(A.max_steps):
        m = chat(msgs, tools)
        usd += m["_usd"]
        for wait in EMPTY_BACKOFF:
            if not is_empty(m):
                break
            STATS["empty"] += 1
            time.sleep(wait)
            m = chat(msgs, tools)
            usd += m["_usd"]
        if is_empty(m) and step == 0:
            raise EmptyReply(f"empty reply {len(EMPTY_BACKOFF) + 1}x in a row")
        # an empty reply after tool calls ends the turn: the calls already made stay (no redo on a new env)
        if env is None:
            env = ToolEnv(A.split, item["id"], root=DATA)  # exactly one env per sample
        r = m.get("reasoning") or m.get("reasoning_content")  # NIM uses reasoning_content
        if r:
            reasoning.append(r)
        tcs = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs} if tcs
                    else {"role": "assistant", "content": m.get("content") or ""})
        if A.backend == "deepseek" and r:  # DeepSeek thinking + tools: HTTP 400 unless the reasoning is sent back
            msgs[-1]["reasoning_content"] = r
        if A.backend == "ifm":  # IFM: replay the reasoning on every assistant turn, "" included
            msgs[-1]["reasoning_content"] = r or ""
        if "_blocks" in m:  # Anthropic format: the turn goes back verbatim (thinking blocks with their signatures)
            msgs[-1]["_blocks"] = m["_blocks"]
        trace.append({"thought": r or "", "text": m.get("content") or "", "calls": []})
        if not tcs:
            final = m.get("content") or ""
            stop = "done"
            break
        for tc in tcs:
            fn = tc.get("function", {})
            name = back.get(fn.get("name"), (fn.get("name") or "").replace("__", "."))
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError
            except ValueError:
                args, status = fn.get("arguments"), "bad_json"
                res = {"ok": False, "error": "arguments không phải JSON object hợp lệ; gọi lại."}
            else:
                asked = dict(args)
                v = env.validate(name, args)
                rare = missing = []
                if v.get("ok") and name in TOOLS:
                    args, rare, missing = review(name, args)
                key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
                if key in warned:  # the same call sent again after a warning: the model's answer stands
                    rare = missing = []
                if not v.get("ok"):
                    status = "schema"
                    res = {"ok": False, "error": f"KHÔNG thực thi (sai schema): {v.get('error')}. Sửa tham số."}
                elif key in seen:
                    status, res = "repeat", seen[key]
                elif rare:
                    status = "rare_arg"
                    warned.add(key)
                    res = {"ok": False, "error": f"CHƯA thực thi. Tham số tùy chọn {rare} của {name} hầu như không bao giờ "
                           "được truyền. GIỮ nếu người dùng yêu cầu rõ giá trị đó trong yêu cầu cuối (vd. 'nội dung ghi là ...', "
                           "'trả bằng ví', 'giao tới ...'): gọi lại y như cũ. BỎ nếu bạn tự lấy từ hồ sơ, lịch sử, kết quả "
                           "tool hay tự đặt theo lý do/mục đích: gọi lại KHÔNG có tham số đó."}
                elif missing:
                    status = "missing_arg"
                    warned.add(key)
                    res = {"ok": False, "error": f"CHƯA thực thi. {name} thường phải có {missing} (vd. dời lịch thì truyền "
                           "cả start và end, giữ nguyên thời lượng). Gọi lại với đủ tham số nếu xác định được."}
                elif TOOLS.get(name, {}).get("kind") == "read" and reads[name] >= READ_CAP:
                    status = "read_cap"
                    res = {"ok": False, "error": f"KHÔNG thực thi: đã gọi {name} {READ_CAP} lần. Đừng tìm nữa; làm tiếp với "
                           "dữ liệu đã có, hoặc sys.ask_missing_info nếu vẫn thiếu."}
                else:
                    status, res = "ok", env.call(name, args)
                    seen[key] = res
                    reads[name] += 1
                if args != asked:
                    trace[-1].setdefault("dropped", []).append({"name": name, "args": sorted(set(asked) - set(args))})
            trace[-1]["calls"].append({"name": name, "arguments": args, "status": status, "result": res})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(res, ensure_ascii=False)})
    if env is None:  # only when --max-steps 0
        env = ToolEnv(A.split, item["id"], root=DATA)
    calls = env.get_tool_calls()
    if any(c["name"] == "sys.ask_missing_info" for c in calls):
        stop = "ask"
    return calls, env.get_state_hash(), {"reasoning": reasoning, "final": final, "stop": stop, "trace": trace,
                                         "usd": round(usd, 6)}

# ------------------------------------------------------------------------------------------ scoring (train)
def args_match(name, a, g):
    if set(a) != set(g):
        return False
    for k in g:
        if (name, k) in FUZZY and isinstance(a[k], str) and isinstance(g[k], str):
            ta, tg = _tokens(a[k]), _tokens(g[k])
            i = len(ta & tg)
            if not i:
                if ta != tg:
                    return False
                continue
            p, r = i / len(ta), i / len(tg)
            if 2 * p * r / (p + r) <= 0.6:
                return False
        elif a[k] != g[k] and _canon_scalar(a[k]) != _canon_scalar(g[k]):
            return False
    return True


def score(item, calls, h):
    gold = item["label"]["tool_calls"]
    genv = ToolEnv(A.split, item["id"], root=DATA)
    for c in gold:
        genv.call(c["name"], c["arguments"])
    if not gold:
        return float(not calls)
    if h != genv.get_state_hash():
        return 0.0
    used, m = set(), 0
    for c in calls:
        for j, g in enumerate(gold):
            if j not in used and c["name"] == g["name"] and args_match(c["name"], c.get("arguments") or {}, g["arguments"]):
                used.add(j)
                m += 1
                break
    if not m:
        return 0.0
    p, r = m / len(calls), m / len(gold)
    return 2 * p * r / (p + r)

# ------------------------------------------------------------------------------------------ main
def main():
    items = [json.loads(l) for l in open(DATA / A.split / f"{A.split}.jsonl", encoding="utf-8") if l.strip()]
    if A.ids:
        want = set(A.ids.split(","))
        items = [i for i in items if i["id"] in want]
    elif A.sample:
        items = random.Random(0).sample(items, A.sample)
    if A.split == "train":
        if A.text_source == "asr" and not A.transcripts:
            sys.exit("--text-source asr needs --transcripts")
        tr = json.load(open(A.transcripts, encoding="utf-8")) if A.transcripts else {}
        if tr:
            items = [i for i in items if i["id"] in tr]
    else:
        tr = json.load(open(A.transcripts or HERE / "transcripts.json", encoding="utf-8"))
    if A.limit:
        items = items[:A.limit]
    extra = [json.load(open(p, encoding="utf-8")) for p in A.extra_transcripts]
    if A.split == "train" and A.holdout_apps:
        hold = set(A.holdout_apps.split(","))
        uses = lambda t: bool(hold & {c["name"].split(".")[0] for h in t["history"] + [t["label"]]
                                      for c in h.get("tool_calls") or []})
        items = [i for i in items if uses(i)]
        TRAIN_TOK[:] = [x for x in TRAIN_TOK if not uses(x[1])]
    ids = {i["id"] for i in items} if A.split == "train" else set()
    pool = {t["id"] for _, t in TRAIN_TOK}
    habits([t for t in TRAIN if t["id"] not in ids and t["id"] in pool])  # never counts the rows it is scored on

    def texts(item):
        if A.split == "train" and A.text_source == "gold":
            return item["audio_text"], gold_texts(item)
        sid, hist, n = item["id"], {}, 0
        for i, h in enumerate(item["history"]):
            if h["role"] == "user":
                n += 1
                k = f"{sid}_u{n}"
                v = merge_versions([tr.get(k, "")] + [x.get(k, "") for x in extra])
                hist[i] = " / ".join(v[len(MULTI):].split("\n")) if v.startswith(MULTI) else v
        return merge_versions([tr.get(sid, "")] + [x.get(sid, "") for x in extra]), hist

    tag = MODEL.split("/")[-1].replace(":", "_")
    suffix = f"_{A.text_source}{len(items)}" if A.split == "train" and A.transcripts else ""
    cache = Path(A.cache or HERE / f"cache_{A.split}_{tag}{suffix}.jsonl")
    traces = Path(A.traces or cache.with_name("traces_" + cache.name.removeprefix("cache_")))
    done = {}
    if cache.is_file():
        bad, stale = 0, set()
        for l in open(cache, encoding="utf-8"):
            if l.strip():
                try:  # a hard kill mid-write can leave a truncated last line: skip it, that sample just reruns
                    o = json.loads(l)
                    oid = o["id"]
                except (json.JSONDecodeError, KeyError):
                    bad += 1
                    continue
                if not o.get("tool_calls") and not (o.get("final") or "").strip():
                    # no calls AND no text = an empty API reply cached by an older solve.py, not a decision
                    if oid not in done:
                        stale.add(oid)
                    continue
                done[oid] = o
                stale.discard(oid)
        if bad:
            print(f"(skipped {bad} incomplete cache line(s); those samples will rerun)", flush=True)
        if stale:
            print(f"(ignored {len(stale)} cached empty API replies; those samples will rerun)", flush=True)
    todo = [i for i in items if i["id"] not in done]
    print(f"{MODEL} via {A.backend} | {len(items)} samples, {len(done)} cached, {len(todo)} to run", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, stop_evt, state = threading.Lock(), threading.Event(), {"stopped": None, "n": 0}

    def run_one(it):
        if stop_evt.is_set():
            return
        if A.stop_file and Path(A.stop_file).exists():  # graceful pause: no new samples, in-flight ones finish
            state["stopped"] = state["stopped"] or f"paused ({A.stop_file} exists)"
            stop_evt.set()
            return
        if A.budget is not None and spent() >= A.budget:
            state["stopped"] = state["stopped"] or f"budget reached (${spent():.2f} of ${A.budget:.2f})"
            stop_evt.set()
            return
        utt, hist = texts(it)
        t0 = time.time()
        try:
            calls, h, dbg = solve(it, utt, hist)
        except DailyQuotaExhausted as e:
            state["stopped"] = f"API refused the run: key, balance or quota ({e.args[0][:120]})"
            stop_evt.set()
            return
        except EmptyReply as e:  # not cached -> the next pass / run retries it; meanwhile filled from base
            with lock:
                state["empty_fail"] = state.get("empty_fail", 0) + 1
            print(f"  !! {it['id']}: {e} (not cached, will retry next pass)", flush=True)
            return
        except Exception as e:
            print(f"  !! {it['id']}: {e}", flush=True)
            return
        steps = dbg.pop("trace")
        rec = {"id": it["id"], "utterance": utt, "tool_calls": calls, "state_hash": h, "model": MODEL, **dbg}
        sc = score(it, calls, h) if A.split == "train" else None
        with lock:
            done[it["id"]] = rec
            with open(traces, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": it["id"], "utterance": utt, "stop": dbg["stop"], "score": sc, "usd": dbg["usd"],
                                    "seconds": round(time.time() - t0, 1), "steps": steps, "final": dbg["final"]},
                                   ensure_ascii=False) + "\n")
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())  # checkpoint is on disk (Drive) before the sample counts as done
            state["n"] += 1
            msg = (f"  [{state['n']}/{len(todo)}] {it['id']} {len(calls)} calls, {time.time() - t0:.0f}s, "
                   f"requests so far {STATS['requests']}")
            if sc is not None:
                msg += f", score {sc:.2f}"
            if BK.get("price"):
                msg += f", ${dbg['usd']:.4f} (spent ${spent():.2f}" + (f" of ${A.budget:.2f})" if A.budget else ")")
            print(msg, flush=True)

    with ThreadPoolExecutor(max(1, A.workers)) as ex:
        list(ex.map(run_one, todo))
    stopped = state["stopped"]
    if stopped:
        print(f"\nSTOPPED: {stopped}. Rerun the same command later to resume.", flush=True)
    kinds = collections.Counter()
    if traces.is_file():
        for l in open(traces, encoding="utf-8"):
            o = json.loads(l)
            kinds.update(["stop=" + o["stop"]] + [c["status"] for s in o["steps"] for c in s["calls"]]
                         + ["dropped_read_arg" for s in o["steps"] for _ in s.get("dropped", [])])
        print(f"\ntraces: {traces} | " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())), flush=True)
    if STATS["empty"] or state.get("empty_fail"):
        print(f"\nempty API replies: {STATS['empty']} re-asked, {state.get('empty_fail', 0)} sample(s) gave up "
              f"(not cached; rerun to retry). Many of these = the API is degraded right now.", flush=True)

    if A.split == "train":
        sc = [score(it, done[it["id"]]["tool_calls"], done[it["id"]]["state_hash"]) for it in items if it["id"] in done]
        print(f"\nTRAIN SCORE: {100 * sum(sc) / max(len(sc), 1):.2f} on {len(sc)} samples")
    else:
        base = {}
        if A.base_submission:
            for l in open(A.base_submission, encoding="utf-8"):
                if l.strip():
                    o = json.loads(l)
                    base[o["id"]] = o
        out, solved, filled = Path(A.out or HERE / "submission.jsonl"), 0, 0
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                o = done.get(it["id"])
                if o:
                    solved += 1
                else:
                    o = base.get(it["id"])
                    filled += bool(o)
                if o:
                    f.write(json.dumps({"id": it["id"], "tool_calls": o["tool_calls"], "state_hash": o["state_hash"]},
                                       ensure_ascii=False) + "\n")
        print(f"\nsubmission.jsonl: {solved} solved here, {filled} from base, {len(items) - solved - filled} missing")


def self_check():
    habits(TRAIN)
    assert review("alarm.create", {"time": "t", "label": "x"}) == ({"time": "t", "label": "x"}, ["label"], [])
    assert review("email.list", {"unread_only": True}) == ({}, [], [])                      # read: dropped silently
    assert review("calendar.update_event", {"event_id": "e", "start": "s"})[2] == ["end"]  # gold always sends end
    assert review("sms.send", {"phone_number": "1", "content": "c"})[1:] == ([], [])        # x-one-of is left alone
    assert rate("contacts.search", "query")[0] > 0.8 and "alarm.create" in arg_hints(["alarm.create"])
    print("self-check ok")


if __name__ == "__main__":
    self_check() if A.self_check else main()
