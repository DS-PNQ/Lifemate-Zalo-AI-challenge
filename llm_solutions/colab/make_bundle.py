"""Builds zola_bundle.zip for the Colab notebook (run: py make_bundle.py). Upload it to MyDrive/ZOLA_colab/.
Contains competition data -> keep the Drive folder private. No API keys are included."""
import zipfile
from pathlib import Path

ZOLA = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "zola_bundle.zip"
CODE = "api_agent"
FILES = {  # path in zip: source
    f"{CODE}/solve.py": ZOLA / CODE / "solve.py",
    f"{CODE}/transcripts_gemini_public_test.json": ZOLA / CODE / "transcripts_gemini_public_test.json",
    f"{CODE}/transcripts_gipformer_reference.json": ZOLA / CODE / "transcripts_gipformer_public_test.json",
    f"{CODE}/submission_68.42.jsonl": ZOLA / CODE / "submission_68.42.jsonl",
    "asr_zipformer/vi_itn.py": ZOLA / "asr_zipformer/vi_itn.py",
    "train+public_test/train/train.jsonl": ZOLA / "train+public_test/train/train.jsonl",
    "train+public_test/train/train_tools_catalog.json": ZOLA / "train+public_test/train/train_tools_catalog.json",
}
for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
    FILES[f"asr_zipformer/asr_model/{f}"] = ZOLA / "asr_zipformer/asr_model" / f
pub = ZOLA / "train+public_test/public_test"
for p in pub.rglob("*"):
    if p.is_file() and "__pycache__" not in p.parts:
        FILES[f"train+public_test/public_test/{p.relative_to(pub).as_posix()}"] = p

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for arc, src in FILES.items():
        z.write(src, arc)
print(f"{len(FILES)} files -> {OUT} ({OUT.stat().st_size / 2**20:.0f} MB)")
