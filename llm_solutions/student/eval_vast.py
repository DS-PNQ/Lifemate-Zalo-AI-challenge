"""The eval notebook (notebooks/neweval_sft.ipynb) as one command for a rented Linux box with 1 GPU (vast.ai, RTX 5090).

On the box (image: PyTorch with a CUDA 12.8 build of torch >= 2.7, which a 5090 needs; ~40 GB disk):
  export KAGGLE_API_TOKEN=...            # or put kaggle.json in ~/.kaggle/
  python eval_vast_onefile.py            # the one file to upload: written here by  python student/eval_vast.py --pack
It installs the packages, downloads the inputs from Kaggle (SOURCES below, kept in --input, so a rerun downloads
nothing), runs held-out train + public_test and writes eval_out/submission.jsonl (validated), pred_*.jsonl, log_*.txt.

  --llm owner/notebook[/versions/N]   another model (its output must hold llm_model/ with lifemate.py beside it)
  --no-eval                           public_test only
  --no-pull                           no Kaggle: everything is already somewhere under --input (found by name)
  --shards N                          processes on the one GPU (4B fp16 is ~9-11 GB each; 2 fit 32 GB, 3 is untested)

The notebook's cells run as they are: infer.py is byte-identical, so scores compare with the Kaggle runs. Changed
are only the input paths and SMOKE / EVAL / GOLD / SHARDS. Added: the score on the gate rows of qwen_k2_rl.

Rules: eval only, not the Docker entry point. The network is used for pip and the Kaggle download, never by the
agent (R-RUN-1). Seeds and greedy decoding are infer.py's (R-REP-1/2). Times here are not the A30 latency (R-RUN-2).
The data package on a rented machine must stay private: destroy the instance afterwards (R-RES-4). The Kaggle token
is read by kagglehub from the environment and never printed or written (CLAUDE.md ground rule 5).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

PACKED = None  # --pack fills this in a copy of this file

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="eval_in")
ap.add_argument("--out", default="eval_out")
ap.add_argument("--llm", default="hoangducbao/k2-rl", help="Kaggle notebook whose output holds llm_model/ + lifemate.py")
ap.add_argument("--shards", type=int, default=2, help="processes on the one GPU")
ap.add_argument("--no-eval", action="store_true", help="public_test only")
ap.add_argument("--no-pull", action="store_true", help="inputs are already under --input")
ap.add_argument("--gold", action="store_true", help="also run gold text on the held-out rows")
ap.add_argument("--smoke", action="store_true", help="8 samples per shard")
ap.add_argument("--notebook", default=str(Path(__file__).resolve().parent / "notebooks/neweval_sft.ipynb"))
ap.add_argument("--pack", action="store_true", help="write eval_vast_onefile.py (this file + the notebook's cells) and stop")
args = ap.parse_args()

# folder name: (kagglehub function, handle, single file). The data package and the raw transcripts are private datasets
SOURCES = {"data": ("dataset_download", "hoangducbao/zola-deepseek-bundle", None),
           "llm": ("notebook_output_download", args.llm, None),
           "asr": ("notebook_output_download", "png0806/asrtest1", "asr_transcripts.json"),  # 01c output, only this file
           "raw": ("dataset_download", "hoangducbao/eval-raw-asr", None)}  # held-out train clips, stock Gipformer

CELLS = PACKED or ["".join(c["source"]) for c in json.load(open(args.notebook, encoding="utf-8"))["cells"]]
if args.pack:
    marker = "PACKED = " + "None  #"
    src = Path(__file__).read_text(encoding="utf-8")
    assert src.count(marker) == 1
    one = Path(__file__).with_name("eval_vast_onefile.py")
    one.write_text(src.replace(marker, f"PACKED = {CELLS!r}  #"), encoding="utf-8")
    sys.exit(f"wrote {one}")

inp, out = Path(args.input).resolve(), Path(args.out).resolve()
assert inp not in (out, *out.parents), "--out inside --input: the next run would find two asr_transcripts.json"
inp.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)


def cell(start, *pairs):
    """The one cell that starts with `start`, with exact-string patches that must each match once."""
    (src,) = [c for c in CELLS if c.startswith(start)]
    for old, new in pairs:
        assert src.count(old) == 1, f"must match once, found {src.count(old)}: {old[:60]!r}"
        src = src.replace(old, new)
    return src.split("\n", 1)[1] if src.startswith("%%") else src


def py(*argv, wait=True, log=None, cwd=out):
    p = subprocess.Popen([sys.executable, *argv], cwd=cwd, stdout=log, stderr=subprocess.STDOUT if log else None)
    if wait:
        assert p.wait() == 0, f"{argv[0]} failed"
    return p


def write(name, src):
    assert "/kaggle/" not in src, f"{name} still has a Kaggle path"
    (out / name).write_text(src, encoding="utf-8")


import torch  # noqa: E402  the image's own build

# before any download: infer.py falls back to the CPU without a word, and an old torch build cannot run on a 5090 (sm_120)
assert torch.cuda.is_available(), "no CUDA: this box needs a torch build for its GPU"
torch.zeros(8, device="cuda").sum().item()
print(torch.__version__, torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB", flush=True)

root = dict.fromkeys(SOURCES, inp)  # where each input is looked for
if not args.no_pull:
    py("-m", "pip", "install", "-q", "kagglehub")
    os.environ["KAGGLEHUB_CACHE"] = str(inp / "kagglehub")  # not ~/.cache: on the big disk, and a rerun finds it. No leading dot: glob ** skips hidden folders
    import kagglehub

    def get(fn, handle, file=None):
        """One download, retried: a single file resumes where a broken connection left it."""
        for attempt in range(5):
            try:
                return Path(getattr(kagglehub, fn)(handle, **({"path": file} if file else {})))
            except Exception as e:
                print(f"  attempt {attempt + 1} failed: {e!r}", flush=True)
                err = e
        sys.exit(f"could not download {handle} {file or ''}: {err!r}\nIs KAGGLE_API_TOKEN set (or ~/.kaggle/kaggle.json), and can that "
                 "account open it? A teammate's private notebook must be shared with you.")

    def model_files(handle):
        """llm_model/ + lifemate.py of the notebook's latest output. The whole output comes as one zip built on the fly,
        which cannot resume: a broken connection lost the ~8 GB model every time."""
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        names, token = [], ""
        with build_kaggle_client() as client:
            while True:
                r = ApiListKernelSessionOutputRequest()
                r.user_name, r.kernel_slug, r.page_size = *handle.split("/")[:2], 200
                if token:
                    r.page_token = token
                resp = client.kernels.kernels_api_client.list_kernel_session_output(r)
                names += [f.file_name for f in resp.files or []]
                token = resp.next_page_token
                if not token:
                    break
        want = [n for n in names if n.startswith("llm_model/") or n == "lifemate.py"]
        assert {"llm_model/config.json", "lifemate.py"} <= set(want), f"no llm_model/ + lifemate.py in the output of {handle}: {names[:20]}"
        return want

    for name, (fn, handle, file) in SOURCES.items():
        if name == "raw" and args.no_eval:
            continue
        print("Kaggle:", handle, file or "", flush=True)
        if name == "llm":  # shortcut: lists the latest version's files, also for a .../versions/N handle (same names expected)
            files = model_files(handle)
            for f in files:
                print("  ", f, flush=True)
                p = get(fn, handle, f)
            root[name] = p.parents[len(Path(files[-1]).parts) - 1]  # the output's top folder
        else:
            root[name] = get(fn, handle, file)
    root["asr"], root["data"] = root["asr"].parent, inp
    if not list(inp.rglob("train/train.jsonl")):  # the bundle was uploaded as a zip and Kaggle kept it zipped
        for z in inp.rglob("*.zip"):
            zipfile.ZipFile(z).extractall(inp / "unzipped")
g = {k: v.as_posix() for k, v in root.items()}
for name, pattern in (("data", "train/train.jsonl"), ("llm", "llm_model/config.json"), ("asr", "asr_transcripts.json")) + (() if args.no_eval else (("raw", "raw_asr*.json"),)):
    found = list(root[name].rglob(pattern))
    assert len(found) == 1 or (name == "raw" and (found or args.no_pull)), (
        f"need exactly one {pattern} under {root[name]}, found {[str(p) for p in found]}; files there: {[p.name for p in root[name].rglob('*') if p.is_file()][:20]}")
(config,) = root["llm"].rglob("llm_model/config.json")
assert (config.parents[1] / "lifemate.py").is_file(), f"no lifemate.py beside {config.parent}"

# the notebook's install cell: its transformers version for this model type, its pip line
setup = {line.split(" ", 1)[0]: line for line in cell("import glob\nimport json\n\nimport torch").split("\n")}
ns = {"_config": json.load(open(config, encoding="utf-8"))}
exec(setup["TRANSFORMERS"], ns)
py("-m", *shlex.split(setup["!pip"][1:].replace("{TRANSFORMERS}", ns["TRANSFORMERS"])))

write("settings.py", cell("%%writefile settings.py",
                          ("SMOKE = False", f"SMOKE = {args.smoke}"), ("EVAL = True", f"EVAL = {not args.no_eval}"),
                          ("GOLD = False", f"GOLD = {args.gold}"), ("SHARDS = 2", f"SHARDS = {args.shards}"),
                          ('"/kaggle/input/**/train/train.jsonl"', f'"{g["data"]}/**/train/train.jsonl"'),
                          ('"/kaggle/input/**/llm_model/config.json"', f'"{g["llm"]}/**/llm_model/config.json"'),
                          ('"/kaggle/input/**/raw_asr*.json"', f'"{g["raw"]}/**/raw_asr*.json"')))
write("vi_itn.py", cell("%%writefile vi_itn.py"))
write("infer.py", cell("%%writefile infer.py"))  # unchanged
write("transcripts.py", cell("import glob\nimport json\nimport re\nimport sys\nimport unicodedata",
                             ('"/kaggle/input/**/asr_transcripts.json"', f'"{g["asr"]}/**/asr_transcripts.json"'),
                             ('"/kaggle/input/**/asr_xfit_train.json"', f'"{inp.as_posix()}/**/asr_xfit_train.json"')))
write("report.py", "from settings import *\n" + cell("import collections") + "\n" + cell(
    'pred = load("public_test", "asr")',
    ('!cd "{ROOT}/public_test" && PYTHONPATH=. python -m zaic_tools --split public_test --root .. --validate /kaggle/working/submission.jsonl',
     'import subprocess, sys\nsubprocess.run([sys.executable, "-m", "zaic_tools", "--split", "public_test", "--root", "..", "--validate", '
     f'{(out / "submission.jsonl").as_posix()!r}], cwd=ROOT / "public_test", check=True)')) + '''
if EVAL:  # qwen_k2_rl trains on the other held-out rows, so for its models only this line is a clean score
    sys.path.insert(0, LIFEMATE_DIR)
    import lifemate as lm
    gate = {x["id"] for x in sum(lm.split_rows(lm.load_split(ROOT, "train"))[1:], [])[::5]}
    report("asr gate rows", [x for x in load("train", "asr") if x["id"] in gate])
print("submission:", ''' + repr((out / "submission.jsonl").as_posix()) + ")\n")

py("vi_itn.py")  # self-check
py("transcripts.py")
runs = ([("train", "gold")] if args.gold and not args.no_eval else []) + ([] if args.no_eval else [("train", "asr")]) + [("public_test", "asr")]
for split, mode in runs:
    print("running", split, mode, "- follow with: tail -f", out / f"log_{split}_{mode}_0.txt", flush=True)
    logs = [open(out / f"log_{split}_{mode}_{i}.txt", "w", encoding="utf-8") for i in range(args.shards)]
    procs = [py("infer.py", split, mode, str(i), wait=False, log=f) for i, f in enumerate(logs)]
    assert not any([p.wait() for p in procs]), f"a shard of {split} {mode} failed, see log_{split}_{mode}_*.txt"
py("report.py")

failed = sum(json.loads(line)["failed"] for f in out.glob("pred_*.jsonl") for line in open(f, encoding="utf-8"))
if failed:  # infer.py keeps a crashed sample's line, so an out-of-memory error only shows up here
    sys.exit(f"{failed} samples crashed (see log_*.txt). If it is CUDA out of memory, rerun with --shards {max(1, args.shards - 1)}")
