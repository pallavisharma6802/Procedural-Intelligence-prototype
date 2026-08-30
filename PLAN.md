# Procedural Intelligence — design

## Product thesis
Not a transcription app. The core object is a **procedural state timeline** (event-sourced):
`recording/transcript → turns → ProceduralEvent[] → reduce → CaseState snapshots → projections`.
One reconstructed case state feeds handoff, documentation, and family communication, and every
value in every document traces back to the transcript line that produced it.

## Design principle: fixed pipeline, swappable site profile
The pipeline is **hospital-agnostic**. The agent graph, the `ProceduralEvent` vocabulary, and the
`CaseState` fields never change per deployment. Everything local — phase names, handoff format,
note headings, family-letter style, terminology — lives in one declarative **SiteProfile**
(`pi/profile.py`, `pi/profiles/*.json`). A new hospital = one JSON file, no code.

Shipped profiles: `default_or` (US OR, I-PASS), `uk_or` (NHS theatre, SBAR, British terms),
`cath_lab` (percutaneous — access site not incision, SBAR, cardiologist), `mmor_robotic`
(robot phases). Select with `PI_PROFILE` or `pi run --profile`.

## Pipeline — a LangGraph graph (`pi/graph.py`)

```
START → ingest → context → extract → reduce → handoff → opnote → family → critic_check
        │                   │        (fold,                                    │
   .srt/.vtt → captions     │        profile   flagged & round 0 ──────────────┤
   audio/video → Whisper    │        phases)          ▼                        │
        → same Turn[]       │                   critic_revise ─────────────────┘
                    windowed + safety-sweep           │ else
                    extraction (profile-aware)        ▼
                                                critic_finalize → END
```

- **ingest** — `.srt`/`.vtt` parsed deterministically, or audio/video → Whisper (`pi/stt.py`:
  Groq `whisper-large-v3` default, `faster-whisper` local). Both emit the same timed `Turn[]`.
  Records the active profile id onto the CaseFile.
- **context** (LLM) — patient descriptor / planned procedure / indication / anaesthesia.
- **extract** (LLM) — windowed workers + a chunked whole-transcript safety sweep; both prompts
  inject the profile's phase vocabulary and `event_focus`. Merge + fuzzy dedupe.
- **reduce** — plain fold, `state(n) = reduce(state(n-1), event(n))`, advancing through the
  profile's phase order; every field records provenance back to events → turns.
- **handoff / opnote / family** (LLM) — system prompt built entirely from the profile; run as
  sequential nodes (3 concurrent hosted calls trip Groq's 8k-tokens/min free-tier limit).
- **critic_check** (LLM) — one combined fact-check of all drafts (per-draft fallback if the
  request would be too large). Flags **only fabricated facts** — invented vitals/labs/doses/
  events — never style or standard-of-care advice; verifies each flagged quote is really in the
  draft; drops self-negated flags. `check → revise → check` is the graph's one real cycle,
  guarded to one round. A draft that failed to generate, or that the critic couldn't check, is
  marked NOT accepted.
- All hosted LLM calls go through a global spacing throttle + bounded 429/5xx retry, so a blown
  quota fails fast and one lost draft never crashes the run.

## Data model (`pi/schemas.py`)
- `Turn { id, start_s, end_s, speaker?, text, source }`  (source: `srt` | `whisper`)
- `ProceduralEvent { id, t_start_s, type, payload, evidence_turn_ids, confidence }`
  - types: `phase_transition, medication_given, incision, conversion, implant_placed, line_placed,
    drain_placed, device_step, blood_loss, hemodynamic_event, transfusion, count_status, specimen,
    complication, equipment_issue, personnel_change, disposition`
- `CaseState { as_of_s, phase, meds[], ebl_ml?, transfusion_totals{}, implants[], lines[], drains[],
   converted?, counts?, complications[], open_concerns[], disposition?, provenance }`
- `CaseContext { patient_descriptor, planned_procedure, indication, anesthesia_type }`
- `Draft { kind, text, unsupported_claims[], revised, accepted }`
- `SiteProfile { id, care_setting, phases, phase_synonyms, procedure_start_phase, event_focus,
   handoff{name,intro,sections[]}, opnote_sections[], family{...}, terminology{} }`

## Tech stack
- Python 3.9 in `.venv`. `pydantic` v2, `typer`, `httpx`, `srt`, `rich`, `langgraph`.
- LLM shim `pi/llm.py`: Groq (default, `qwen/qwen3.8-27b` — Groq hosts no medical model),
  Ollama local fallback, Gemini optional. `PI_MODEL` / `PI_CRITIC_MODEL` override.
- STT optional local backend: `pip install '.[local-stt]'` (faster-whisper).
- Artifacts: JSON/Markdown under `runs/<case_id>/` (`casefile.json` is the full blackboard).

## CLI (`pi/cli.py`)
```
pi run <file> [--profile P] [--case-id X] [--upto STAGE]   # transcript OR audio/video
pi profiles                                                # list site profiles
pi graph                                                   # mermaid of the LangGraph
pi stage <name> <case-id>                                  # re-run one node (reuses run's profile)
pi show <case-id> [turns|events|context|state|provenance|handoff|opnote|family|log]
pi evaluate <case-id>                                      # score vs data/synthetic/<id>_ground_truth.json
```

## Status (2026-08-28)
- 4 synthetic cases pass `pi evaluate` (case01/02/03 default_or, case04 cath_lab).
- 4 real MM-OR knee transcripts (382–631 turns) run clean end to end, all drafts critic-accepted.
- Whisper audio front-end verified on a synthetic clip.
- Profile swing verified: one transcript, `default_or` vs `uk_or` → I-PASS vs SBAR, US vs UK
  terminology and headings, `handoff` vs `handover` phase — no code change.

## Next
- Speaker diarization (pyannote) for transcribed audio — currently `speaker=None`.
- A real de-identified transcript per non-OR profile (cath lab, L&D, endoscopy).
- Profile authoring guide; validate shipped profiles against real hospital templates.

## No-PHI rule
Synthetic / MM-OR / properly de-identified research data only — transcripts and audio. Never a
real patient recording obtained ourselves. `PI_STT=groq` uploads audio to Groq; use `PI_STT=local`
for anything sensitive.
