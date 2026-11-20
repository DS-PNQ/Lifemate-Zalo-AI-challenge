#!/bin/bash
# Entry point: reads /data, writes /result/submission.jsonl (and /result/details.jsonl with stop reasons, timings and transcripts).
set -euo pipefail
mkdir -p /result
python3 /code/predict.py
