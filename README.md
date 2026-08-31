# Procedural Intelligence

Reconstruct a procedural state timeline from an operating-room transcript or recording, then
generate an OR-to-ICU/PACU handoff, an operative-note draft, and a family update.

This is not a transcription tool. The source of truth is an append-only `ProceduralEvent` log;
everything downstream is derived from a folded `CaseState`, and every state field keeps
provenance back to the events and transcript lines that produced it.

## Features

- **One reconstruction, three documents.** Handoff, operative note, and family update are all
  projections of the same `CaseState`, so they cannot disagree with each other.
- **Provenance everywhere.** `pi show <case> provenance` traces each state field to its events
  and their verbatim transcript quotes; the web UI makes the same links clickable.
- **Hospital-agnostic.** Phase vocabulary, handoff format (I-PASS / SBAR), note headings,
  terminology, and family-letter style live in a JSON site profile. New hospital, one file.
- **Real context over MCP.** The `context` step pulls procedure, indication, home meds,
  allergies, and problem list from a connected clinical-context MCP server; the transcript only
  fills gaps.
- **Audio or text in.** `.srt`/`.vtt`, or audio/video via Whisper (Groq, local, or a diarizing
  vendor). German MM-OR audio is handled with Whisper translate.
- **Fact-checked drafts.** A critic pass flags only invented facts (vitals, labs, doses,
  events), verifies each flagged quote, and drives one revision.
- **Degrades, does not crash.** A rate-limited draft becomes a placeholder; a silent recording
  yields zero events rather than a hallucinated timeline.

## Architecture

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

| node | what it does |
|---|---|
| `ingest` | parse a caption file, or transcribe audio/video. Backends: Groq Whisper (default), `faster-whisper` local, AssemblyAI/Deepgram (with diarization). `PI_WHISPER_TASK=translate` for non-English audio. |
| `roles` | map speaker labels to clinical roles. `.srt` prefixes map directly; diarized ids use one LLM call. Events are attributed to the dominant role of their evidence turns. |
| `context` | pull patient set-up from a clinical-context MCP server; infer anything missing from the transcript. |
| `extract` | windowed workers plus a whole-transcript safety sweep, merged and deduped. |
| `reduce` | deterministic fold of events into `CaseState` snapshots, advancing through the profile's phase order and recording provenance. |
| `handoff` / `opnote` / `family` | build their prompts entirely from the active site profile. |
| `critic_check` -> `critic_revise` | flag fabricated facts only, verify quotes, at most one revision round. |

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
SBAR), `mmor_robotic` (robot phases). The profile used is recorded in the run's `casefile.json`
and reused by `pi stage`.

## Clinical context over MCP

The site profile covers *how a hospital talks*. What a hospital *knows* about the patient comes
from connected systems over the Model Context Protocol.

`pi/mcp_client.py` is a minimal MCP client (stdio, JSON-RPC 2.0, no SDK dependency). Servers are
declared in `.mcp.json` (or `PI_MCP_CONFIG`), using the standard `mcpServers` config shape:

```json
{ "mcpServers": {
    "clinical-context": { "command": "python3", "args": ["mcp_servers/clinical_context_server.py"] }
} }
```

When a server exposing `lookup_patient` is connected, `context` matches the case and pulls
scheduled procedure, indication, anaesthesia plan, home medications, allergies, and problem
list. Fields the server does not supply are inferred from the transcript. `context.sources`
records which per field; the web UI badges each `EHR` or `heard`. With `PI_MCP=off` or no
`.mcp.json`, the pipeline runs the transcript-only path unchanged.

`mcp_servers/clinical_context_server.py` is a reference server backed by three de-identified
sample records. A real deployment points `.mcp.json` at a hospital EHR / OR-board MCP server;
the tool names and shapes are the contract. `pi mcp` lists the connected servers.

## Web UI

`pi serve` starts a FastAPI backend and a single-page UI (`web/index.html`): transcript,
procedural timeline, and case state plus documents. A draggable playhead replays the
`CaseState`; hovering an event, a state chip, or a linked phrase in a document highlights the
chain across all three panels. The profile dropdown re-runs the case under another profile.

`python web/build.py` bakes the current runs into `web/standalone.html`, a single self-contained
file that runs without the server.

## Installation

```bash
python3 -m venv .venv          # Python 3.9+
./.venv/bin/pip install -e .
cp .env.example .env           # add GROQ_API_KEY, or configure Ollama
```

An LLM provider is required: the Groq API when `GROQ_API_KEY` is set (`qwen/qwen3.8-27b`), else
a local Ollama model. `ffmpeg` is optional, used to chunk audio over the hosted size limit.

## Usage

```bash
./.venv/bin/python -m pi.cli serve                                 # web UI + API on :8000
./.venv/bin/python -m pi.cli run data/synthetic/case01_lapchole.srt
./.venv/bin/python -m pi.cli run recording.m4a -c my_case          # audio/video -> Whisper -> pipeline
./.venv/bin/python -m pi.cli run case.srt --profile uk_or          # pick a site profile

./.venv/bin/python -m pi.cli profiles                              # list site profiles
./.venv/bin/python -m pi.cli mcp                                   # list connected MCP servers
./.venv/bin/python -m pi.cli graph                                 # print the graph as mermaid
./.venv/bin/python -m pi.cli show <case_id> events|state|provenance|handoff|log
./.venv/bin/python -m pi.cli stage events <case_id>                # re-run one node
./.venv/bin/python -m pi.cli evaluate <case_id>                    # score against ground truth
```

Run artifacts land in `runs/<case_id>/`: `casefile.json`, `turns.json`, `events.json`,
`state.jsonl`, `handoff.md`, `opnote.md`, `family.md`.

## Configuration

Set via environment or `.env`:

| variable | purpose |
|---|---|
| `PI_PROFILE` | site profile name or path (default `default_or`) |
| `PI_PROVIDER` | `groq` \| `ollama` \| `gemini` |
| `PI_MODEL`, `PI_CRITIC_MODEL` | model overrides |
| `GROQ_API_KEY` | Groq API key |
| `PI_STT` | `groq` \| `assemblyai` \| `deepgram` \| `local` |
| `PI_WHISPER_TASK` | `transcribe` \| `translate` |
| `PI_MCP_CONFIG`, `PI_MCP` | MCP config path; `PI_MCP=off` disables it |
| `PI_MIN_SPACING`, `PI_WINDOW_TURNS`, `PI_SWEEP_CHUNK` | throughput tuning for rate limits |

## Results

Model: `qwen/qwen3.8-27b` on Groq. `pi evaluate` scores extracted event types and reconstructed
state fields against hand-authored ground truth.

### Synthetic cases

| case | profile | turns | events | event-type score | state score | drafts accepted |
|---|---|---:|---:|:---:|:---:|:---:|
| `case01_lapchole` | `default_or` | 30 | 27 | 11 / 11 | 5 / 5 | 3 / 3 |
| `case02_tka_uneventful` | `default_or` | 18 | 16 | 7 / 7 | 4 / 4 | 3 / 3 |
| `case03_trauma_exlap` | `default_or` | 20 | 32 | 10 / 10 | 3 / 3 | 3 / 3 |
| `case04_cath_pci` | `cath_lab` | 16 | 23 | 6 / 6 | 2 / 2 | 3 / 3 |

**34 / 34** ground-truth event types found, **14 / 14** state fields correct, **12 / 12** drafts
passed the critic with no unresolved fabrication flags.

Behavioural checks these cases exist to verify:

- `case02` (routine knee): the critic must not invent complications. Severity comes out `stable`,
  complications `none documented`, family note plainly reassuring.
- `case03` (damage-control trauma): severity `unstable`; the deliberately-incorrect count is
  reported as intentional (4 retained laparotomy pads); running transfusion figures are
  reconciled to **8 PRBC / 6 FFP / 2 platelets / 1 cryo** rather than summed (a naive sum gives
  24 / 18 / 6 / 3).
- `case04` (percutaneous PCI): no "incision" language anywhere; handoff format is SBAR; family
  note is framed around a catheter procedure and closes with the cardiologist, not a surgeon.

### Profile swing

The same `case01_lapchole` transcript with `--profile uk_or`: handoff switches from I-PASS to
**SBAR**, terminal phase `handoff` to `handover`, "sponge count" to "swab count", "anesthesia"
to "anaesthesia", and the family closing line changes. 27 events, all three drafts accepted, no
code change.

### MCP clinical context

`case01_lapchole` with the reference `clinical-context` server connected: `context` resolved the
case to sample record `P-4471` and pulled **7 of 7 setup fields from the EHR**, including
**2 allergies (penicillin, shellfish)** and **3 home medications** that the transcript never
mentions. The handoff then carries an `Allergies:` line the transcript-only run cannot produce.
`case04_cath_pci` similarly pulls an iodinated-contrast allergy for a contrast procedure. With
`PI_MCP=off` the same runs fall back to 4 transcript-inferred fields.

### MM-OR (real data)

| take | source | turns | events |
|---|---|---:|---:|
| `007_TKA` | machine-translated transcript | 537 | 7 |
| `007_TKA` | raw German audio, Whisper `translate`, ffmpeg-chunked (58 min) | 446 | 10 |
| `006_PKA` | transcript | 631 | 11 |
| `003_TKA` | transcript | 502 | 8 |

All four complete end to end with three critic-accepted drafts. Extraction is sparse because
these transcripts are thin: patient descriptor, indication, and disposition are usually never
stated. On the raw `007_TKA` audio the family draft misframed the case as a training simulation,
which the translated chatter's mention of a Mako simulator step made plausible; the handoff and
op-note held up.

### Timing

~2-3 min per synthetic case, ~8-10 min for a 500-630-turn MM-OR case. The bottleneck is Groq's
free-tier cap of **8,000 tokens per minute** per model; long transcripts need
`PI_MIN_SPACING=19 PI_WINDOW_TURNS=90`. A 429 on one draft degrades to a placeholder rather than
crashing; `pi stage <name>` re-runs it.

## Repository layout

```
pi/
  graph.py            LangGraph pipeline
  agents/             ingest, roles, context, extract, reduce, projections, critic
  profile.py          SiteProfile + loader
  profiles/*.json     shipped site profiles
  llm.py              provider shim (groq / ollama / gemini) + throttle + retry
  stt.py              speech-to-text backends
  mcp_client.py       minimal MCP client
  server.py           FastAPI backend
  webexport.py        casefile -> web JSON + provenance links
mcp_servers/          reference clinical-context MCP server + sample records
web/                  single-page UI, build script, baked standalone
data/synthetic/       hand-authored cases + ground truth
scripts/              MM-OR download helper
```

## Data and PHI

`data/synthetic/` holds hand-authored cases with ground-truth JSON; this drives development and
`pi evaluate`. MM-OR transcripts and audio (`scripts/download_mm-or.sh`) are a real-world stress
test and are not committed; the dataset requires a form at <https://github.com/egeozsoy/MM-OR>.

Use synthetic, MM-OR, or properly de-identified research data only, never a real patient
recording obtained yourself. `PI_STT=groq` uploads audio to Groq; use `PI_STT=local` for
anything sensitive.

## Limitations

- Diarizing STT backends (AssemblyAI, Deepgram) are wired but only exercised on labelled `.srt`.
- The MCP reference server holds three records; real matching needs a real EHR server.
- The shipped profiles are starting points, not validated against any hospital's templates.
- Extraction quality is bounded by the free-tier model; a larger model raises event recall.
