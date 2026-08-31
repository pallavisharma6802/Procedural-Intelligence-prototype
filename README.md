# Procedural Intelligence

Reconstruct a procedural state timeline from an operating-room transcript or recording, then
generate an OR-to-ICU/PACU handoff, an operative-note draft, and a family update.

The source of truth is an append-only `ProceduralEvent` log. Everything downstream is derived
from the folded `CaseState`, and every state field keeps provenance back to the events and
transcript turns that produced it.

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

An LLM provider is required. By default the pipeline uses the Groq API when `GROQ_API_KEY` is
set (`qwen/qwen3.8-27b`), and falls back to a local Ollama model otherwise.

```bash
cp .env.example .env    # add GROQ_API_KEY, or configure Ollama
```

## Run

```bash
./.venv/bin/python -m pi.cli serve                                 # web UI + API on :8000
./.venv/bin/python -m pi.cli run data/synthetic/case01_lapchole.srt
./.venv/bin/python -m pi.cli run recording.m4a -c my_case          # audio/video -> Whisper -> pipeline
./.venv/bin/python -m pi.cli run case.srt --profile uk_or          # pick a site profile

./.venv/bin/python -m pi.cli profiles                              # list site profiles
./.venv/bin/python -m pi.cli graph                                 # print the graph as mermaid
./.venv/bin/python -m pi.cli show <case_id> events|state|provenance|handoff|log
./.venv/bin/python -m pi.cli stage events <case_id>                # re-run one node
./.venv/bin/python -m pi.cli evaluate <case_id>                    # score against ground truth
```

Config via env or `.env`: `PI_PROFILE`, `PI_PROVIDER` (`groq`|`ollama`|`gemini`), `PI_MODEL`,
`PI_CRITIC_MODEL`, `GROQ_API_KEY`, `PI_STT` (`groq`|`assemblyai`|`deepgram`|`local`),
`PI_WHISPER_TASK` (`transcribe`|`translate`), `PI_MIN_SPACING`, `PI_WINDOW_TURNS`, `OLLAMA_HOST`.

Run artifacts land in `runs/<case_id>/`: `casefile.json`, `turns.json`, `events.json`,
`state.jsonl`, `handoff.md`, `opnote.md`, `family.md`.

## Pipeline

A LangGraph graph (`pi/graph.py`) runs the agents over one shared `CaseFile`:

```
ingest -> roles -> context -> extract -> reduce -> handoff -> opnote -> family -> critic_check
                                                                                     |
                                                        flagged, round 0 ------------ +
                                                                v
                                                         critic_revise --------------- +
                                                                | else
                                                                v
                                                         critic_finalize -> END
```

- `ingest` parses a caption file (`.srt`/`.vtt`) or transcribes audio/video. Backends: Groq
  Whisper (default), `faster-whisper` (`PI_STT=local`), AssemblyAI or Deepgram
  (transcription plus diarization). Non-English audio: `PI_WHISPER_TASK=translate`.
- `roles` maps speaker labels to clinical roles. `.srt` prefixes map directly; diarized ids
  use one LLM call. Events are then attributed to the dominant role of their evidence turns.
- `context` extracts patient descriptor, planned procedure, indication, anaesthesia.
- `extract` runs windowed workers plus a whole-transcript safety sweep, merged and deduped.
- `reduce` folds events into `CaseState` snapshots (deterministic), advancing through the
  active profile's phase order and recording provenance per field.
- `handoff` / `opnote` / `family` build their prompts from the active site profile.
- `critic_check` checks the drafts for fabricated facts only (invented vitals, labs, doses,
  events), verifies each flagged quote appears in the draft, and drives at most one revision.

Hosted LLM calls pass through a global spacing throttle with bounded retry on 429/5xx.

## Site profiles

The pipeline is fixed. Anything that differs between hospitals lives in a JSON profile
(`pi/profile.py`, `pi/profiles/*.json`):

| field | controls |
|---|---|
| `phases`, `phase_synonyms`, `procedure_start_phase` | the phase vocabulary the reducer uses |
| `handoff` | format name (I-PASS, SBAR), ordered sections, guidance |
| `opnote_sections` | note headings, in order |
| `family` | language, reading level, closing line, notes |
| `terminology` | canonical to local term (`OR` -> `theatre`, `surgeon` -> `cardiologist`) |
| `event_focus` | event types the safety sweep emphasises |

Four profiles ship: `default_or` (US, I-PASS), `uk_or` (NHS, SBAR), `cath_lab` (percutaneous,
SBAR), `mmor_robotic` (robot phases). Add a hospital by adding one JSON file. The profile used
is recorded in the run's `casefile.json`; `pi stage` reuses it.

## Web UI

`pi serve` starts a FastAPI backend and a single-page UI (`web/index.html`): transcript,
procedural timeline, and case state plus documents. A draggable playhead replays the
`CaseState`; hovering an event, a state chip, or a linked phrase in a document highlights the
chain across all three panels. The profile dropdown re-runs the case under another profile.

`./.venv/bin/python web/build.py` bakes the current runs into `web/standalone.html`, a single
self-contained file that runs without the server.

## Data and PHI

- `data/synthetic/` holds hand-authored cases with ground-truth JSON. This is the primary
  driver for development and `pi evaluate`.
- MM-OR transcripts and audio (see `scripts/download_mm-or.sh`) are a real-world stress test.
  They are not committed; the dataset requires a form at
  <https://github.com/egeozsoy/MM-OR>.

Use synthetic, MM-OR, or properly de-identified research data only. Never a real patient
recording obtained yourself. `PI_STT=groq` uploads audio to Groq; use `PI_STT=local` for
anything sensitive.

## Status

Seven demo cases run end to end with all drafts accepted by the critic. Four synthetic cases
pass `pi evaluate` (11/11, 7/7, 10/10, 6/6 event types).

- Running `case01_lapchole` under `uk_or` produces an SBAR handover, phase `handover`,
  British terminology, and UK op-note headings, with no code changes.
- `case04_cath_pci` under `cath_lab` produces SBAR with catheterisation vocabulary and a
  family note framed around a percutaneous procedure.
- `case03_trauma_exlap` reconciles running transfusion counts to a single figure per product
  (8 PRBC / 6 FFP / 2 platelets / 1 cryo) rather than summing the running totals.
- MM-OR `007_TKA` runs from both the machine-translated transcript and the raw German audio
  (`PI_WHISPER_TASK=translate`, ffmpeg-chunked). On the raw audio the family draft misframed
  the case as a training simulation, which the transcript's mention of a Mako simulator step
  made plausible.

Known limitations: diarizing STT backends are wired but only exercised on labelled `.srt`;
Groq's free tier caps throughput at 8,000 tokens per minute, so long transcripts need
`PI_MIN_SPACING=19 PI_WINDOW_TURNS=90`; the shipped profiles are starting points, not
validated against any hospital's real templates.
