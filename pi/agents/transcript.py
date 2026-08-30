"""Front end: source file -> Turn[].

- `.srt` / `.vtt`  -> parse captions (deterministic)
- audio / video    -> transcribe with Whisper (see pi/stt.py), emit the same shape
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import srt

from ..casefile import CaseFile
from ..schemas import Turn
from ..stt import is_audio, transcribe
from .base import Agent


class TranscriptAgent(Agent):
    name = "transcript"
    requires = ()
    produces = "turns"

    async def run(self, cf: CaseFile) -> CaseFile:
        path = Path(cf.source_path)
        if is_audio(path):
            cf.turns = await asyncio.to_thread(transcribe, path)
            cf.log(self.name, f"transcribed {len(cf.turns)} turns from {path.name} (whisper)")
            return cf

        subs = list(srt.parse(path.read_text()))
        cf.turns = []
        for i, sub in enumerate(subs):
            speaker, text = _split_speaker(" ".join(sub.content.split()))
            cf.turns.append(
                Turn(
                    id=f"t{sub.index or i:04d}",
                    start_s=sub.start.total_seconds(),
                    end_s=sub.end.total_seconds(),
                    text=text,
                    speaker=speaker,
                    source="srt",
                )
            )
        cf.log(self.name, f"parsed {len(cf.turns)} turns from {path.name}")
        return cf


def _split_speaker(content: str) -> tuple[str | None, str]:
    """Pick up an explicit 'SURGEON:' style prefix if the transcript has one."""
    head = content.split(":", 1)
    if len(head) == 2 and 0 < len(head[0]) <= 20 and head[0].isupper():
        return head[0].strip(), head[1].strip()
    return None, content
