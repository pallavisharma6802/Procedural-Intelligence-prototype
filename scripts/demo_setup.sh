#!/usr/bin/env bash
# Rebuild the demo runs from the committed sample audio, then bake the web bundle.
# Needs an LLM provider configured (see README) and ffmpeg on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-./.venv/bin/python}

run() { echo "== $1"; "$PY" -m pi.cli run "data/synthetic/audio/$1.mp3" --case-id "$1" --profile "$2"; }

run case01_lapchole       default_or
run case02_tka_uneventful default_or
run case03_trauma_exlap   default_or
run case04_cath_pci       cath_lab
"$PY" -m pi.cli run data/synthetic/case01_lapchole.srt --case-id case01_uk --profile uk_or

pm() { echo "== $1"; "$PY" -m pi.cli run "data/primock/$1.mp3" --case-id "$1" --profile primary_care; }
pm primock_d1c01
pm primock_d2c01

"$PY" web/build.py
echo "done - ./.venv/bin/python -m pi.cli serve"
