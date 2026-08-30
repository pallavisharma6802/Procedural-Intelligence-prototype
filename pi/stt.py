"""Audio/video -> timed Turn[]. Same output shape as the .srt parser, so the rest of the
pipeline never knows whether it started from a caption file or a recording.

Backends (PI_STT=...):
  groq        (default)  Groq whisper-large-v3 — transcription only, no speakers
  assemblyai            AssemblyAI — transcription + speaker diarization in one call
  deepgram             Deepgram nova — transcription + diarization in one call
  local                faster-whisper on this machine (offline; `pip install '.[local-stt]'`)

The `roles` agent maps raw diarization speaker labels (A/B/…, SPEAKER_01) to clinical roles.
Local pyannote / NeMo Sortformer diarization is a future backend.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from .llm import _post_with_retry  # reuse the throttle + retry
from .schemas import Turn

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".ogg", ".oga", ".flac", ".webm", ".aac", ".opus"}
_GROQ_LIMIT_MB = 24
_CHUNK_SECONDS = 900
_LONG = httpx.Timeout(900.0, connect=15.0)


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXT


# Whisper's stock hallucinations over silence / music
_JUNK = {"you", "thank you", "thanks for watching", "bye", ".", "。", "you.", "thank you.",
         "please subscribe", "the end", "okay", "ok", "so", "um", "uh"}


def _is_junk(text: str) -> bool:
    t = text.strip().lower().strip(".!? ")
    return not t or t in _JUNK or (len(set(t.split())) == 1 and len(t.split()) > 1)


def transcribe(path: str | Path) -> list[Turn]:
    path = Path(path)
    backend = os.environ.get("PI_STT", "groq")
    fn = {"groq": _groq, "assemblyai": _assemblyai, "deepgram": _deepgram, "local": _local}.get(backend)
    if fn is None:
        raise ValueError(f"unknown PI_STT={backend!r}")
    turns = [t for t in fn(path) if t.text.strip() and not _is_junk(t.text)]
    for i, t in enumerate(turns):  # renumber ids consistently
        t.id = f"t{i:04d}"
        t.text = " ".join(t.text.split())
    return turns


def _seg_turns(segments, *, speaker=None, source="whisper") -> list[Turn]:
    return [
        Turn(id="t", start_s=float(s), end_s=float(e), text=txt, speaker=speaker, source=source)
        for (s, e, txt) in segments
    ]


# --- Groq whisper-large-v3 (transcription only) -------------------------
def _groq(path: Path) -> list[Turn]:
    key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("PI_WHISPER_MODEL", "whisper-large-v3")
    size_mb = path.stat().st_size / 1e6
    if size_mb <= _GROQ_LIMIT_MB:
        return _seg_turns(_groq_one(path, key, model, 0.0), source="whisper")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{path.name} is {size_mb:.0f} MB, over Groq's ~{_GROQ_LIMIT_MB} MB free-tier limit, "
            "and ffmpeg isn't installed to split it. `brew install ffmpeg`, trim the file, "
            "or use PI_STT=local / assemblyai."
        )
    out: list[Turn] = []
    for chunk_path, offset in _ffmpeg_chunks(path):
        out.extend(_seg_turns(_groq_one(chunk_path, key, model, offset), source="whisper"))
    return out


def _groq_one(path: Path, key: str, model: str, offset: float):
    # PI_WHISPER_TASK=translate -> force English output for non-English audio (e.g. MM-OR is German)
    endpoint = "translations" if os.environ.get("PI_WHISPER_TASK") == "translate" else "transcriptions"
    with open(path, "rb") as fh:
        r = _post_with_retry(
            f"https://api.groq.com/openai/v1/audio/{endpoint}",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": model, "response_format": "verbose_json", "temperature": "0"},
            files={"file": (path.name, fh, "application/octet-stream")},
            timeout=_LONG,
        )
    body = r.json()
    segs = body.get("segments") or []
    if not segs and body.get("text"):
        return [(offset, offset, body["text"])]
    return [(s["start"] + offset, s["end"] + offset, s["text"]) for s in segs]


# --- AssemblyAI (transcription + diarization) --------------------------
def _assemblyai(path: Path) -> list[Turn]:
    key = os.environ["ASSEMBLYAI_API_KEY"]
    h = {"authorization": key}
    with open(path, "rb") as fh:
        up = httpx.post("https://api.assemblyai.com/v2/upload", headers=h, content=fh.read(), timeout=_LONG)
    up.raise_for_status()
    job = httpx.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=h,
        json={"audio_url": up.json()["upload_url"], "speaker_labels": True},
        timeout=60,
    )
    job.raise_for_status()
    tid = job.json()["id"]
    while True:
        time.sleep(4)
        st = httpx.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=h, timeout=60).json()
        if st["status"] == "completed":
            break
        if st["status"] == "error":
            raise RuntimeError(f"AssemblyAI: {st.get('error')}")
    return [
        Turn(id="t", start_s=u["start"] / 1000, end_s=u["end"] / 1000, text=u["text"],
             speaker=f"SPEAKER_{u['speaker']}", source="assemblyai")
        for u in (st.get("utterances") or [])
    ]


# --- Deepgram (transcription + diarization) ---------------------------
def _deepgram(path: Path) -> list[Turn]:
    key = os.environ["DEEPGRAM_API_KEY"]
    with open(path, "rb") as fh:
        r = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": "nova-2", "diarize": "true", "punctuate": "true", "utterances": "true"},
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/*"},
            content=fh.read(),
            timeout=_LONG,
        )
    r.raise_for_status()
    utts = r.json()["results"].get("utterances") or []
    return [
        Turn(id="t", start_s=u["start"], end_s=u["end"], text=u["transcript"],
             speaker=f"SPEAKER_{u.get('speaker', 0)}", source="deepgram")
        for u in utts
    ]


# --- ffmpeg segmentation (oversized files) ---------------------------
def _ffmpeg_chunks(path: Path):
    tmp = Path(tempfile.mkdtemp(prefix="pi_stt_"))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "segment",
         "-segment_time", str(_CHUNK_SECONDS), "-ac", "1", "-ar", "16000", "-c:a", "aac",
         str(tmp / "chunk_%03d.m4a")],
        check=True,
    )
    for i, chunk in enumerate(sorted(tmp.glob("chunk_*.m4a"))):
        yield chunk, i * _CHUNK_SECONDS


# --- faster-whisper (local, offline, transcription only) -------------
def _local(path: Path) -> list[Turn]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError("PI_STT=local needs `pip install 'procedural-intelligence[local-stt]'`") from exc
    name = os.environ.get("PI_WHISPER_MODEL", "base")
    model = WhisperModel(name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(path), vad_filter=True)
    return _seg_turns([(s.start, s.end, s.text) for s in segments], source="whisper")
