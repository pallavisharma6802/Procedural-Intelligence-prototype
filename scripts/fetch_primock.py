#!/usr/bin/env python3
"""Fetch a few PriMock57 consultations and prepare them for the pipeline.

PriMock57 (Papadopoulos Korfiatis et al., ACL 2022) is 57 mock primary-care consultations
released under CC-BY-4.0: https://github.com/babylonhealth/primock57

For each requested consultation this:
  - shallow-clones the repo (metadata only) and `git lfs pull`s just that consultation's audio
  - mixes the separate doctor / patient channels into one mono mp3 in data/primock/
  - copies the clinician's reference note alongside it

    python scripts/fetch_primock.py day1_consultation01 day2_consultation01

Then:  pi run data/primock/primock_d1c01.mp3 --case-id primock_d1c01 --profile primary_care
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/babylonhealth/primock57.git"
OUT = Path(__file__).resolve().parent.parent / "data" / "primock"


def case_id(name: str) -> str:
    m = re.match(r"day(\d+)_consultation(\d+)", name)
    return f"primock_d{int(m.group(1))}c{int(m.group(2)):02d}" if m else name


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["git", "clone", "--depth", "1", REPO, td], check=True,
                       env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": __import__("os").environ["PATH"]})
        for name in names:
            inc = [f"--include=audio/{name}_doctor.wav", f"--include=audio/{name}_patient.wav"]
            subprocess.run(["git", "-C", td, "lfs", "pull", *inc], check=True)
            doc, pat = (Path(td) / "audio" / f"{name}_{who}.wav" for who in ("doctor", "patient"))
            cid = case_id(name)
            dest = OUT / f"{cid}.mp3"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(doc), "-i", str(pat),
                "-filter_complex",
                "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,"
                "loudnorm=I=-19:TP=-2,aformat=sample_rates=16000:channel_layouts=mono[out]",
                "-map", "[out]", "-c:a", "libmp3lame", "-q:a", "5", str(dest),
            ], check=True)
            note = Path(td) / "notes" / f"{name}.json"
            if note.exists():
                (OUT / f"{cid}_note.json").write_text(json.dumps(json.loads(note.read_text()), indent=1))
            print(f"{dest.name}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["day1_consultation01", "day1_consultation02", "day2_consultation01"])
