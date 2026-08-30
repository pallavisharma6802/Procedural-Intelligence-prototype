# Procedural Intelligence

Reconstruct a **procedural state timeline** from an operating-room recording **or** transcript,
then project it into an OR→ICU/PACU handoff, an operative-note draft, and a family-safe update.

Not a transcription tool. The source of truth is an append-only `ProceduralEvent` log; everything
downstream is derived from the folded `CaseState`, with provenance from every state field back to
the events and transcript turns that produced it.

## The interface

`pi serve` runs a web UI (`web/`, FastAPI backend) — three columns that make the thesis legible:

```
   transcript            procedural timeline            case state  +  documents
 (what the room said)   (the reconstructed spine)     (as-of any moment)  (handoff · op note · family)
        mono          phase-banded event stream          chip board         rendered per site profile
         └──────────────── hover anything → it lights up in all three columns ────────────────┘
```

- A **draggable playhead** replays the `CaseState` — scrub back and the EBL, phase, counts and
  disposition rewind to what was known at that minute.
- Hover an event, a state chip, or a **linked phrase in a document** and the chain lights up:
  document sentence → the event that produced it → the transcript line that was heard.
- The **profile dropdown** re-runs the same case under another hospital's conventions
  (I-PASS ⇄ SBAR, US ⇄ NHS terms) — one transcript, different record.
- 0-event / thin recordings are shown as such, not papered over.

Interactive demo (7 real reconstructed cases, no install): **[claude.ai artifact](https://claude.ai/code/artifact/540b5b44-e395-41fb-a50d-9e7fd5a5fc9b)**

## Hospital-agnostic by design

The pipeline — the agent graph, the `ProceduralEvent` vocabulary, the `CaseState` fields — is
**fixed**. Everything that differs between a US OR, an NHS theatre, and a cath lab lives in one
declarative **site profile** (`pi/profile.py`, `pi/profiles/*.json`):

| field | what it controls |
|---|---|
| `phases` / `phase_synonyms` / `procedure_start_phase` | the phase vocabulary the reducer folds through |
| `handoff` | format name (I-PASS, SBAR, …) + ordered sections + guidance |
| `opnote_sections` | the note headings, in order |
| `family` | language, reading level, closing line, setting-specific notes |
| `terminology` | canonical → local term (`OR`→`theatre`, `surgeon`→`cardiologist`, `epinephrine`→`adrenaline`) |
| `event_focus` | which event types the safety sweep emphasises |

Four ship in-box: `default_or` (US, I-PASS), `uk_or` (NHS, SBAR, British terms), `cath_lab`
(percutaneous — no "incision", SBAR, cardiologist), `mmor_robotic` (robot phases). Add a hospital
by dropping in one JSON file. Select with `pi run --profile <name>` or `PI_PROFILE=<name-or-path>`;
the profile used is recorded in the run's `casefile.json`.

## Architecture

A **LangGraph** graph (`pi/graph.py`) drives specialist agents over one shared `CaseFile`.
`pi graph` prints it as mermaid:

```
START → ingest → roles → context → extract → reduce → handoff → opnote → family → critic_check
   │             (LLM?)  (LLM)     (LLM)    (fold,                                     │
   │                                        determ.)          flagged & round 0 ─────┤
 .srt/.vtt → parse captions                                          ▼                │
 audio/video → Whisper / AssemblyAI (pi/stt.py)               critic_revise ──────────┘
                → same Turn[] shape                                  │ else
                                                                     ▼
                                                             critic_finalize → END
```

- `ingest` takes a caption file (`.srt`/`.vtt`, parsed deterministically) **or** an audio/video
  file — transcribed by Groq `whisper-large-v3` (default, transcription only), `faster-whisper`
  locally (`PI_STT=local`), or **AssemblyAI / Deepgram** for transcription **+ speaker
  diarization** in one call (`PI_STT=assemblyai`). Non-English audio: `PI_WHISPER_TASK=translate`.
  All paths emit the same timed `Turn[]`.
- `roles` maps raw speaker labels → clinical roles (surgeon / anesthesia / circulating nurse / …).
  `.srt` prefixes (`SURGEON:`) map directly; diarized ids get one LLM call. No labels → no-op.
  Events are then attributed to the dominant role of their evidence turns (`by_role`).
- `reduce` is a plain function — not LLM-wrapped.
- `context` (LLM) extracts patient descriptor / planned procedure / indication / anesthesia.
- `extract` (LLM) = windowed workers **plus** a chunked whole-transcript "safety sweep". Both
  prompts inject the active profile's phase vocabulary and `event_focus`. Passes are merged and
  deduped (fuzzy for narrative event types).
- `reduce` folds events into `CaseState` snapshots using the profile's phase order; every field
  records provenance back to the events (and transcript turns) that set it — `pi show <id> provenance`.
- `handoff` / `opnote` / `family` build their system prompt entirely from the profile.
- The three projections run as sequential nodes (3 concurrent hosted calls reliably trip
  free-tier token/min limits).
- `critic_check` fact-checks all drafts in **one combined call** (splits to per-draft if the
  payload is too big) **only for fabricated facts** — invented vitals/labs/doses/events, not
  style, not standard-of-care guidance. It verifies each flagged quote actually appears in the
  draft and drops self-negated flags. The `critic_check → critic_revise → critic_check` cycle is
  the graph's one real branch, guarded to at most one round. A draft whose generation failed, or
  that the critic couldn't check, is marked NOT accepted (rerun its stage).
- All hosted LLM calls pass through a global spacing throttle + bounded 429/5xx retry, so a
  blown daily quota fails fast instead of hanging and one lost draft doesn't crash the run.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
```

LLM: **Groq free tier** by default when `GROQ_API_KEY` is set — model `qwen/qwen3.8-27b`
(no card, fast, doesn't train on API data; use with public/synthetic data). Falls back to
local Ollama otherwise.

```bash
cp .env.example .env        # then paste a key from https://console.groq.com/keys
# or stay local:
ollama pull qwen2.5:3b      # fast, weak extraction
```

**Medical models?** Groq hosts none as of 2026-08 — no Meditron / MedGemma / BioMistral,
only general models (`qwen/qwen3.x-27b`, `openai/gpt-oss-20b|120b`) plus `whisper-large-v3`
for the future audio front-end. `gpt-oss-120b` is stronger but its free tier is only
200k tokens/**day**, so `qwen3.8-27b` is the default. If a medical model shows up, set
`PI_MODEL` / `PI_CRITIC_MODEL` — nothing else changes.

Force a provider: `PI_PROVIDER=ollama|groq|gemini`, pick a model with `PI_MODEL`.

## Run

```bash
./.venv/bin/python -m pi.cli serve                              # web UI at http://127.0.0.1:8000
./.venv/bin/python -m pi.cli run data/synthetic/case01_lapchole.srt
./.venv/bin/python -m pi.cli run recording.m4a -c my_case            # audio/video → Whisper → pipeline
./.venv/bin/python -m pi.cli run case.srt --profile uk_or            # pick a site profile
./.venv/bin/python -m pi.cli run <file> --upto understand            # stop early

./.venv/bin/python -m pi.cli profiles                           # list site profiles
./.venv/bin/python -m pi.cli graph                              # print the LangGraph as mermaid
./.venv/bin/python -m pi.cli show case01_lapchole events|context|state|provenance|handoff|log
./.venv/bin/python -m pi.cli stage events case01_lapchole       # re-run one node (reuses run's profile)
./.venv/bin/python -m pi.cli evaluate case01_lapchole           # score vs ground truth
```

Config via env (or `.env`): `PI_PROFILE`, `PI_PROVIDER` (`groq`|`ollama`|`gemini`), `PI_MODEL`,
`PI_CRITIC_MODEL`, `GROQ_API_KEY`, `PI_STT` (`groq`|`assemblyai`|`deepgram`|`local`),
`PI_WHISPER_TASK` (`transcribe`|`translate`), `PI_MIN_SPACING`, `PI_WINDOW_TURNS`,
`OLLAMA_HOST`. Startup warnings silenced unless `PI_WARNINGS=1`.

**Web UI** (`pi serve`, `web/`): drop a `.srt` or recording on the page, pick a profile, watch
the graph run, then explore the reconstruction. `web/standalone.html` bakes the current runs
into one self-contained file (that's what the artifact link above is).

Artifacts land in `runs/<case_id>/`: `casefile.json` (full blackboard), `turns.json`,
`events.json`, `state.jsonl`, `handoff.md`, `opnote.md`, `family.md`.

## Data

- `data/synthetic/` — hand-authored cases with ground truth. Primary driver.
- `MM-OR_data/` — MM-OR `.srt` transcripts for a real-world stress test (secondary).

No PHI — for transcripts **and** audio. Synthetic / MM-OR / properly de-identified research
data only; never a real patient recording you obtained yourself. Note: `PI_STT=groq` uploads
the audio to Groq — keep hosted STT for non-PHI, use `PI_STT=local` otherwise.

## Status

LangGraph pipeline, running end to end on Groq (`qwen/qwen3.8-27b`).

**Synthetic cases** — all passing `pi evaluate`:

| case | profile | what it tests | event types | state |
|---|---|---|---|---|
| `case01_lapchole` | `default_or` | lap chole converted to open; bleeding, drain | 11/11 | 5/5 |
| `case02_tka_uneventful` | `default_or` | routine TKA — critic must NOT invent problems | 7/7 | 4/4 |
| `case03_trauma_exlap` | `default_or` | damage-control trauma — unstable, ICU, intentional retained packs, massive transfusion | 10/10 | 3/3 |
| `case04_cath_pci` | `cath_lab` | percutaneous PCI — SBAR, no "incision", cardiologist | 6/6 | 2/2 |

- case02 stays plainly reassuring (no invented complications, severity "stable").
- case03 handoff flags severity **unstable**, explains the intentional incorrect count, reconciles
  running transfusion totals to 8 PRBC / 6 FFP / 2 plt / 1 cryo (naive summing gives 24/18/…).
- case04 handoff comes out as **SBAR** with cath vocabulary (radial 6F sheath, radial band,
  contrast 140 mL, access-site + distal-pulse checks q15min); family note = "a procedure through
  a small tube in an artery… lie still so the puncture site can heal… **the cardiologist** will
  come speak with you".

**Profile swing** — the same `case01_lapchole` transcript run with `--profile uk_or` produces an
**SBAR** handover instead of I-PASS, phase `handover` not `handoff`, "swab count" / "anaesthesia"
/ "consultant", UK op-note headings, and the family closing line "come and speak with you as soon
as they can". One transcript, zero code change.

**Real MM-OR data** — four machine-translated German knee cases, all run clean end to end
through LangGraph, every draft critic-accepted:

| take | turns | events | picked up |
|---|---|---|---|
| `002_PKA` | 382 | 10 | femoral plate implant, plate equipment issue, surgeon leaves before closure |
| `003_TKA` | 502 | 8  | tibia implant, antibiotic, equipment issue |
| `007_TKA` | 537 | 12 | Mako robotic TKA, tibial + femoral components + cement, 3 equipment issues, counts correct |
| `006_PKA` | 631 | 11 | TKA components, EBL 100, 4 device steps |

`context` (patient / procedure / indication) and disposition come back empty because these
transcripts genuinely never state them — that's the data, not a bug. Phase sometimes stalls
mid-case where the translated closing chatter is too vague to classify.

Groq's free tier is **8,000 tokens/minute** per model, so long transcripts need
`PI_MIN_SPACING=19 PI_WINDOW_TURNS=90 PI_SWEEP_CHUNK=300` (≈8–10 min/case). The throttle+retry
absorbs the rest; a single 413/429 on one chunk just drops a few events.

**Real MM-OR audio** — the 1 GB `take_audios.zip` is German MP3s. `007_TKA.mp3` (58 min, ffmpeg-
chunked) → Groq Whisper **translate** → 446 turns → pipeline extracts ~10 events (robot camera
fault, implant-sizing issue resolved with a size-4 tibia, cement, insertion, meniscus specimen).
The handoff/op-note hold up; the **family note misframed it as a "training simulation"** — the
translated OR chatter genuinely mentions a Mako *simulator* step. Good illustration of why a
human signs the draft, and of the pipeline surfacing rather than hiding its uncertainty.

**Speaker roles** — synthetic `.srt` prefixes map to roles directly; on `case01` that's
20 surgeon / 8 anesthesia / 2 circulating-nurse turns, and events get attributed
("cefazolin — said by anesthesia", "counts correct — said by circulating nurse"). Diarized
audio (AssemblyAI / Deepgram) feeds the same `roles` step — wired, not yet run on real OR audio.

~1.5–3 min/synthetic case; ~2 min for a short audio clip; ~8–10 min for a 500–630-turn MM-OR
case under the free-tier token budget.

Known gaps: diarization is coded but only exercised on labelled `.srt` so far (need an
AssemblyAI/Deepgram key + real audio); Groq free tier 429s the occasional draft on long cases
(rerun that one `pi stage <name>`); the shipped profiles are illustrative, not validated against
any hospital's real templates. Next: run diarization on MM-OR audio; validate profiles with a
clinical advisor; profile authoring guide.
