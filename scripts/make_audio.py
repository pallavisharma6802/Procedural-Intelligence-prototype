#!/usr/bin/env python3
"""Render a role-labelled .srt transcript to multi-voice OR audio (macOS `say` + ffmpeg).

    python scripts/make_audio.py data/synthetic/case01_lapchole.srt MM-OR_data/clips/case01_lapchole.mp3

Speakers are read from the "SPEAKER: text" prefix on each cue and mapped to a distinct
system voice. Inter-cue silence is capped so a 40-minute case becomes a few minutes of
audio; the pipeline re-transcribes it, so only relative order and rough pacing matter.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

VOICE = {
    "SURGEON": "Alex", "CARDIOLOGIST": "Alex", "ATTENDING": "Alex",
    "ANESTHESIA": "Daniel", "ANAESTHETIST": "Daniel", "ANESTHESIOLOGIST": "Daniel", "CRNA": "Moira",
    "NURSE": "Samantha", "CIRCULATOR": "Samantha", "CIRCULATING NURSE": "Samantha",
    "SCRUB": "Karen", "SCRUB NURSE": "Karen", "SCRUB TECH": "Karen",
    "ASSISTANT": "Fred", "FELLOW": "Fred", "RESIDENT": "Fred", "PA": "Fred",
    "PERFUSIONIST": "Ralph", "TECH": "Ralph", "RADIOLOGY": "Ralph", "REP": "Ralph",
}
DEFAULT_VOICE = "Tom"
RATE = 178              # words/min for `say`
MAX_GAP = 2.4          # seconds of silence allowed between cues
LEAD_IN = 0.4

CUE = re.compile(r"(\d\d):(\d\d):(\d\d)[,.](\d\d\d)\s*-->", re.M)


def parse_srt(path: Path):
    blocks = re.split(r"\n\s*\n", path.read_text().strip())
    out = []
    for b in blocks:
        m = CUE.search(b)
        if not m:
            continue
        h, mn, s, ms = map(int, m.groups())
        start = h * 3600 + mn * 60 + s + ms / 1000
        text = b[m.end():].strip()
        text = text.split("-->", 1)[-1].strip() if "-->" in text else text
        text = re.sub(r"\s+", " ", text.splitlines()[-1] if "\n" in text else text).strip()
        who, _, rest = text.partition(":")
        if rest and who.isupper() and len(who) < 24:
            speaker, line = who.strip(), rest.strip()
        else:
            speaker, line = "", text
        if line:
            out.append((start, speaker, line))
    return out


def dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def main() -> None:
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    cues = parse_srt(src)
    if not cues:
        sys.exit(f"no cues parsed from {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clips: list[tuple[float, Path]] = []
        play_head = LEAD_IN
        prev_src = cues[0][0]
        for i, (t_src, speaker, line) in enumerate(cues):
            gap = min(MAX_GAP, max(0.25, t_src - prev_src)) if i else 0.0
            prev_src = t_src
            play_head += gap
            voice = VOICE.get(speaker.upper(), DEFAULT_VOICE)
            aiff = tmp / f"{i:03d}.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(RATE), "-o", str(aiff), line],
                           check=True)
            clips.append((play_head, aiff))
            play_head += dur(aiff)

        total = play_head + 1.0
        inputs: list[str] = []
        filt: list[str] = []
        for j, (at, aiff) in enumerate(clips):
            inputs += ["-i", str(aiff)]
            filt.append(f"[{j}]adelay={int(at*1000)}:all=1[a{j}]")
        mix = "".join(f"[a{j}]" for j in range(len(clips)))
        filt.append(f"{mix}amix=inputs={len(clips)}:normalize=0:dropout_transition=0,"
                    f"apad,atrim=0:{total:.2f},asetpts=N/SR/TB,"
                    f"loudnorm=I=-19:TP=-2,aformat=sample_rates=22050:channel_layouts=mono[out]")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex", ";".join(filt), "-map", "[out]",
             "-c:a", "libmp3lame", "-q:a", "5", str(dest)],
            check=True,
        )
    print(f"{dest}  ({total:.0f}s, {len(clips)} lines)")


if __name__ == "__main__":
    main()
