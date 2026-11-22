"""Static compliance checker for the ZAC 2026 LifeMate rules (see RULES.md).

Usage:
    python tools/check_rules.py <file-or-dir> [...]          # dev mode: network/API use = WARN
    python tools/check_rules.py --final <file-or-dir> [...]  # final/Docker mode: network/API use = ERROR

Exit code 1 if any ERROR, else 0. Suppress a finding on a line with:  # zac-allow: R-XXX-N <reason>
Heuristic only: a clean run does not prove compliance (R-SIZE, R-RES-1, R-RUN-2 need manual checks).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP_DIRS = {".venv", ".cache", "__pycache__", "models", "train+public_test", ".git", "docx_review_render",
             "student_design_reference", "exp"}
SELF = Path(__file__).resolve()

# (rule id, severity, regex, message). severity "NET" = WARN in dev, ERROR with --final.
LINE_RULES = [
    ("R-ENV-1", "ERROR", re.compile(r"_environment\b|state\.json"),
     "touches <split>_environment/ or state.json — solution must use zaic_tools only"),
    ("R-ENV-2", "ERROR", re.compile(r"\b(?:copy\.)?(?:deep)?copy\s*\(\s*env\b|pickle\.dumps?\s*\(\s*env\b"),
     "copies/pickles a ToolEnv — one env per sample, never snapshot/reset"),
    ("R-ENV-4", "ERROR", re.compile(r"\benv\.get_state\s*\("),
     "env.get_state() in agent code — app data only via tool calls (debug/eval scripts: add zac-allow)"),
    ("R-ENV-4", "ENVATTR",
     re.compile(r"\benv\.(?!call\b|validate\b|get_tool_calls\b|get_state_hash\b|get_state\b"
                r"|allowed\b|sample_id\b|split\b)[A-Za-z_]\w*"),  # allowed/sample_id/split = same info as the jsonl context
     "accesses ToolEnv internals — only call/validate/get_tool_calls/get_state_hash are allowed"),
    ("R-REP-2", "ERROR", re.compile(r"do_sample\s*=\s*True"),
     "sampling enabled — decoding must be deterministic"),
    ("R-REP-2", "ERROR", re.compile(r"\btemperature\s*[=:]\s*(?!0(?:\.0+)?\b)\d"),
     "non-zero temperature — use greedy / temperature=0"),
    ("R-RUN-1", "NET", re.compile(r"^\s*(?:import|from)\s+(requests|httpx|aiohttp|urllib\.request|urllib3|openai|"
                                  r"anthropic|google\.genai|google\.generativeai|google|grpc|riva|litellm)\b"),
     "network/API client import — forbidden at inference"),
    ("R-RUN-1", "NET", re.compile(r"https?://(?!localhost|127\.0\.0\.1)"),
     "external URL — inference must be fully offline"),
    ("R-RUN-1", "NET", re.compile(r"api_keys?\.txt|API_KEY|from_pretrained\([^)]*\bhub\b"),
     "API key / hub download — inference must be fully offline"),
    ("R-RES-3", "WARN", re.compile(r"public_test[^\"'\n]*(?:rationale|transcript|cache|pseudo|label)|"
                                   r"private_test[^\"'\n]*(?:rationale|transcript|cache|pseudo|label)"),
     "uses externally-produced public/private_test artefact — never for training/tuning/few-shot"),
]


def check_file(path: Path, final: bool) -> list[tuple[str, int, str, str, str]]:
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    lines = text.splitlines()
    has_toolenv = "ToolEnv" in text
    in_doc = False
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # skip triple-quoted blocks (docstrings, prompt templates): mentions there are not code access
        n_tq = line.count('"""') + line.count("'''")
        if in_doc or n_tq:
            starts_in_doc = in_doc
            if n_tq % 2:
                in_doc = not in_doc
            if starts_in_doc or n_tq >= 2 or in_doc:
                continue
        allow = re.search(r"zac-allow:\s*([A-Z0-9\- ,]+)", line)
        allowed = set(re.findall(r"R-[A-Z]+-\d+", allow.group(1))) if allow else set()
        for rid, sev, rx, msg in LINE_RULES:
            if rid in allowed or not rx.search(line):
                continue
            if sev == "NET":
                sev = "ERROR" if final else "WARN"
            if sev == "ENVATTR":
                if not has_toolenv:
                    continue
                sev = "ERROR"
            out.append((sev, i, rid, msg, line.strip()[:120]))

    # R-ENV-2: more than one ToolEnv( construction site in a file
    sites = [i for i, l in enumerate(lines, 1) if re.search(r"\bToolEnv\s*\(", l) and not l.lstrip().startswith("#")
             and "zac-allow" not in l]
    if len(sites) > 1:
        out.append(("WARN", sites[1], "R-ENV-2", f"ToolEnv( constructed at {len(sites)} sites {sites} — ensure exactly one per sample", ""))
    # R-ENV-3: tool_calls must come from env.get_tool_calls()
    if sites and "get_tool_calls" not in text:
        out.append(("ERROR" if final else "WARN", sites[0], "R-ENV-3", "creates ToolEnv but never submits env.get_tool_calls()", ""))
    if sites and "get_state_hash" not in text:
        out.append(("ERROR" if final else "WARN", sites[0], "R-ENV-3", "creates ToolEnv but never submits env.get_state_hash()", ""))
    # R-REP-1: seeds when ML libs are used
    uses_ml = re.search(r"^\s*(?:import|from)\s+(torch|numpy|transformers|vllm|llama_cpp|sherpa_onnx)\b", text, re.M)
    if uses_ml and not re.search(r"(manual_seed|random\.seed|np\.random\.seed|set_seed|seed_everything)\s*\(", text):
        out.append(("WARN", 1, "R-REP-1", "ML library imported but no seed is fixed in this file", ""))
    return out


def iter_files(targets: list[str]):
    for t in targets:
        p = Path(t)
        if p.is_file():
            if p.suffix == ".py" and p.resolve() != SELF:
                yield p
        elif p.is_dir():
            for f in p.rglob("*.py"):
                if not SKIP_DIRS.intersection(f.parts) and f.resolve() != SELF:
                    yield f


def main(argv: list[str]) -> int:
    final = "--final" in argv
    targets = [a for a in argv if a != "--final"] or ["."]
    n_err = n_warn = 0
    for f in iter_files(targets):
        for sev, ln, rid, msg, src in check_file(f, final):
            n_err += sev == "ERROR"
            n_warn += sev == "WARN"
            print(f"{sev:5} {f}:{ln} [{rid}] {msg}" + (f"\n        > {src}" if src else ""))
    print(f"-- {n_err} error(s), {n_warn} warning(s) ({'final' if final else 'dev'} mode). See RULES.md.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
