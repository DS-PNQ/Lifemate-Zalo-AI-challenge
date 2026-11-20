"""Chain-of-Draft version of rationales/gen_rationales.py (AdaHome, arXiv 2607.18034): each reason keeps only the deciding fact.

The long reasons paste the tool description, which the prompt already holds: 12.9 words on average. Here an ID argument keeps its
item label and source tool, an ask names the missing parameters, and every other step gets a short constant: 4.7 words.
Same order and count as the long file, so 02c ver3, 02 ver2 and 02b ver8 take it unchanged with WHY = True.
Attach this rationales.json or the long one, never both: the notebooks take the first glob match.

python gen_rationales_short.py      writes adahome/rationales.json and prints a few samples"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rationales"))
import gen_rationales as g


def reason(call, earlier, catalog):
    """earlier: [(tool name, result)] of every step before this one, history included."""
    name, args = call["name"], call["arguments"]
    if name == "sys.ask_missing_info":
        return f"Thiếu {', '.join(args.get('params', []))}."
    parts = []
    for k, v in args.items():
        if not (k.endswith("_id") or k.endswith("_ids")) or not isinstance(v, (str, list)):
            continue
        for one in v if isinstance(v, list) else [v]:
            src = next(((n, it) for n, r in reversed(earlier) if (it := g.find_item(r, one)) is not None), None)
            label = " ".join(g.describe(src[1]).split()[:6]) if src else ""
            parts.append(f"{one}{f' ({label})' if label else ''} từ {src[0]}" if src else f"{one} đã có trong hội thoại")
    if parts:
        return "; ".join(parts) + "."
    return "Cần tra trước." if catalog.get(name, {}).get("kind") == "read" else "Làm theo yêu cầu."


def sample(x, catalog):
    return long_sample(x, catalog)[:-1] + ["Xong." if x["label"]["tool_calls"] else "Không cần gọi tool."]


# gen_rationales.main and .sample look these names up at call time, so its walk over history and its count check stay shared
long_sample, g.reason, g.sample, g.OUT = g.sample, reason, sample, Path(__file__).with_name("rationales.json")

if __name__ == "__main__":
    g.main()
