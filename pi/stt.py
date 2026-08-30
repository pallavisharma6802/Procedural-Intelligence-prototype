"""Audio/video -> timed Turn[] . Same output shape as the .srt parser, so the rest of the
pipeline never knows whether it started from a caption file or a recording.

Backends:
  PI_STT=groq   (default)  -> Groq whisper-large-v3, hosted, free tier
  PI_STT=local             -> faster-whisper on this machine (offline; `pip install faster-whisper`)

No diarization yet — every turn comes back with speaker=None. (pyannote is a later add.)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from .llm import _post_with_retry  # reuse the throttle + retry
from .schemas import Turn

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".ogg", ".oga", ".flac", ".webm", ".aac", ".opus"}
_GROQ_LIMIT_MB = 24  # free-tier upload ceiling; larger files are segmented if ffmpeg is present
_CHUNK_SECONDS = 900


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXT


def transcribe(path: str | Path) -> list[Turn]:
    path = Path(path)
    backend = os.environ.get("PI_STT", "groq")
    if backend == "local":
        segments = _local(path)
    else:
        segments = _groq(path)
    turns: list[Turn] = []
    for i, (start, end, text) in enumerate(segments):
        text = " ".join(text.split())
        if not text:
            continue
        turns.append(Turn(id=f"t{i:04d}", start_s=float(start), end_s=float(end), text=text, source="whisper"))
    return turns


# --- Groq whisper-large-v3 ------------------------------------------------
def _groq(path: Path) -> list[tuple[float, float, str]]:
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("PI_WHISPER_MODEL", "whisper-large-v3")
    size_mb = path.stat().st_size / 1e6

    if size_mb <= _GROQ_LIMIT_MB:
        return _groq_one(path, key, model, offset=0.0)

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{path.name} is {size_mb:.0f} MB, over Groq's ~{_GROQ_LIMIT_MB} MB free-tier limit, "
            "and ffmpeg isn't installed to split it. Install ffmpeg (`brew install ffmpeg`), "
            "trim the file, or set PI_STT=local."
        )
    out: list[tuple[float, float, str]] = []
    for chunk_path, offset in _ffmpeg_chunks(path):
        out.extend(_groq_one(chunk_path, key, model, offset))
    return out


def _groq_one(path: Path, key: str, model: str, offset: float) -> list[tuple[float, float, str]]:
    with open(path, "rb") as fh:
        r = _post_with_retry(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": model, "response_format": "verbose_json", "temperature": "0"},
            files={"file": (path.name, fh, "application/octet-stream")},
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
    body = r.json()
    segs = body.get("segments") or []
    if not segs and body.get("text"):  # no timestamps came back — one big turn
        return [(offset, offset, body["text"])]
    return [(s["start"] + offset, s["end"] + offset, s["text"]) for s in segs]


# --- ffmpeg segmentation (only used for oversized files) ----------------
def _ffmpeg_chunks(path: Path):
    tmp = Path(tempfile.mkdtemp(prefix="pi_stt_"))
    pattern = str(tmp / "chunk_%03d.m4a")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "segment", "-segment_time", str(_CHUNK_SECONDS),
         "-ac", "1", "-ar", "16000", "-c:a", "aac", pattern],
        check=True,
    )
    for i, chunk in enumerate(sorted(tmp.glob("chunk_*.m4a"))):
        yield chunk, i * _CHUNK_SECONDS


# --- faster-whisper (local, offline) -----------------------------------
def _local(path: Path) -> list[tuple[float, float, str]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError("PI_STT=local needs `pip install faster-whisper`") from exc
    name = os.environ.get("PI_WHISPER_MODEL", "base")
    model = WhisperModel(name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), vad_filter=True)
    return [(s.start, s.end, s.text) for s in segments]
