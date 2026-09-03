"""turns -> ProceduralEvent[]. LLM, windowed over the transcript, windows run concurrently."""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter

from ..casefile import CaseFile
from ..llm import complete_json, info
from ..profile import SiteProfile, active_profile
from ..schemas import EventType, ProceduralEvent, Turn
from .base import Agent

WINDOW_TURNS = int(os.environ.get("PI_WINDOW_TURNS", "60"))
WINDOW_OVERLAP = int(os.environ.get("PI_WINDOW_OVERLAP", "10"))
SWEEP_CHUNK = int(os.environ.get("PI_SWEEP_CHUNK", "220"))
_MAX_PROMPT_CHARS = int(os.environ.get("PI_MAX_PROMPT_CHARS", "11000"))


def _types(profile: SiteProfile, consult: bool) -> str:
    phases = " | ".join(profile.phases)
    if consult:
        return f"""- phase_transition   payload: {{"phase": one of {phases}}}  (use the closest one)
- finding            payload: {{"description","category"?:"symptom|exam|history|impression"}}  one distinct symptom, one piece of relevant history, one examination finding, or the clinician's working diagnosis. One fact per event; do not restate the same fact in different words.
- medication_given   payload: {{"name","dose"?,"route"?}}   any drug taken, tried, or advised
- complication       payload: {{"description","resolved"?}}   a red-flag / safety concern raised
- disposition        payload: {{"destination": e.g. "follow-up", "referral", "prescription", "self-care", "home", "clinic", "A&E", "detail"?}}   the management plan and follow-up"""
    return f"""- phase_transition   payload: {{"phase": one of {phases}}}  (use the closest one)
- medication_given   payload: {{"name","dose"?,"route"?}}
- incision           payload: {{"site"?}}   (for a percutaneous procedure, the first vascular/needle access)
- conversion         payload: {{"from","to"}}          e.g. from "laparoscopic" to "open"
- implant_placed     payload: {{"device"}}             ONLY prostheses/hardware left in the patient (joint, mesh, graft, stent, pacemaker). NOT drains or IV lines.
- line_placed        payload: {{"type","site"?}}       IV / arterial / central line / sheath
- drain_placed       payload: {{"type","site"?}}       JP, Blake, chest tube, NG
- device_step        payload: {{"step"}}
- blood_loss         payload: {{"ebl_ml": number}}     latest running or final EBL estimate
- hemodynamic_event  payload: {{"description"}}         hypotension, tachycardia, arrhythmia, desat
- transfusion        payload: {{"product":"PRBC|FFP|platelets|cryo","units"?,"cumulative"?:bool}} (cumulative=true when the number is a running total "so far" / "total given", not units hung at that moment)
- count_status       payload: {{"status":"correct|incorrect|pending","intentional"?:bool,"detail"?}} (intentional=true when items are deliberately left in, e.g. damage-control packing)
- specimen           payload: {{"description"}}
- complication       payload: {{"description","resolved"?}}
- equipment_issue    payload: {{"description"}}
- personnel_change   payload: {{"description"}}
- disposition        payload: {{"destination": short unit name (e.g. PACU, ICU, CCU, ward, floor, home), "detail"?}}"""


def system_prompt(profile: SiteProfile) -> str:
    consult = "finding" in (profile.event_focus or []) or getattr(profile, "consultation", False)
    guidance = (
        """Guidance for a consultation / clinic transcript:
- Emit a `finding` for each distinct symptom the patient reports (onset, character, associated
  symptoms, red-flag negatives), each relevant piece of history (past conditions, medications,
  social history), each examination finding, and the clinician's working diagnosis (impression).
- `medication_given` covers any drug discussed - already taken, tried, or advised.
- `disposition` is the plan: follow-up interval, referral, safety-netting, prescription, or
  self-care advice.
- A `complication` here is a red-flag / safety concern raised during the consultation."""
        if consult else
        """Guidance:
- A `complication` is any unintended adverse event: bleeding beyond what is expected, organ/vessel
  injury, inability to obtain the critical view, hemodynamic instability needing intervention,
  unplanned conversion. When surgery is converted lap->open, emit BOTH a `conversion` event AND a
  `complication` describing why.
- A drain (Jackson-Pratt, Blake, chest tube) is `drain_placed`, never `implant_placed`.
- Emit a fresh `blood_loss` event each time an EBL number is stated."""
    )
    return f"""You are a clinical scribe extracting a structured timeline from a \
{profile.care_setting.replace("_", " ")} transcript window. Return ONLY events that matter for a \
handoff, a clinical note, or a patient-facing summary. Ignore small talk and logistics.

Event types (use exactly these):
{_types(profile, consult)}

{guidance}

Return JSON: {{"events":[{{"type","payload","evidence_turn_ids":["t0007"],"confidence":0-1}}]}}
Return {{"events":[]}} if the window has nothing clinically meaningful.
{_EXAMPLE if not consult else _CONSULT_EXAMPLE}"""


_EXAMPLE = """
EXAMPLE
window:
  t0004 [00:05:30] SURGEON: Let's do our time out...
  t0005 [00:05:52] ANESTHESIA: Cefazolin two grams IV went in about ten minutes ago.
  t0013 [00:24:00] SURGEON: I'm not comfortable continuing laparoscopically, converting to open.
  t0024 [00:52:00] SURGEON: Placing a Jackson-Pratt drain in the fossa.
output:
{"events":[
 {"type":"phase_transition","payload":{"phase":"timeout"},"evidence_turn_ids":["t0004"],"confidence":0.9},
 {"type":"medication_given","payload":{"name":"cefazolin","dose":"2 g","route":"IV"},"evidence_turn_ids":["t0005"],"confidence":0.95},
 {"type":"conversion","payload":{"from":"laparoscopic","to":"open"},"evidence_turn_ids":["t0013"],"confidence":0.9},
 {"type":"complication","payload":{"description":"unable to proceed laparoscopically, converted to open"},"evidence_turn_ids":["t0013"],"confidence":0.8},
 {"type":"drain_placed","payload":{"type":"Jackson-Pratt","site":"gallbladder fossa"},"evidence_turn_ids":["t0024"],"confidence":0.9}
]}"""

_CONSULT_EXAMPLE = """
EXAMPLE
window:
  t0006 [00:00:40] PATIENT: I've had a cough for about ten days now, bringing up green phlegm
  t0009 [00:01:10] PATIENT: bit short of breath going up the stairs, not at rest
  t0012 [00:01:50] CLINICIAN: any chest pain, any fevers? - no chest pain, felt hot a couple of nights
  t0017 [00:03:00] PATIENT: I've got asthma, I use the blue inhaler most days
  t0031 [00:07:00] CLINICIAN: sounds like a chest infection. I'll start you on amoxicillin 500 three times a day for five days, and see you back if you're not better in a week
output:
{"events":[
 {"type":"finding","payload":{"description":"10-day productive cough with green sputum","category":"symptom"},"evidence_turn_ids":["t0006"],"confidence":0.9},
 {"type":"finding","payload":{"description":"exertional breathlessness on stairs, not at rest","category":"symptom"},"evidence_turn_ids":["t0009"],"confidence":0.85},
 {"type":"finding","payload":{"description":"no chest pain; subjective fevers a couple of nights","category":"symptom"},"evidence_turn_ids":["t0012"],"confidence":0.8},
 {"type":"finding","payload":{"description":"asthma, uses salbutamol inhaler most days","category":"history"},"evidence_turn_ids":["t0017"],"confidence":0.9},
 {"type":"finding","payload":{"description":"lower respiratory tract / chest infection","category":"impression"},"evidence_turn_ids":["t0031"],"confidence":0.85},
 {"type":"medication_given","payload":{"name":"amoxicillin","dose":"500 mg TDS x5 days","route":"oral"},"evidence_turn_ids":["t0031"],"confidence":0.9},
 {"type":"disposition","payload":{"destination":"follow-up","detail":"review in 1 week if not improving"},"evidence_turn_ids":["t0031"],"confidence":0.9}
]}"""


def sweep_prompt(profile: SiteProfile) -> str:
    focus = ", ".join(profile.event_focus) or (
        "complication, conversion, blood_loss, hemodynamic_event, transfusion, count_status, "
        "specimen, disposition, medication_given"
    )
    return f"""You are a procedural safety auditor for a {profile.care_setting.replace("_", " ")} \
case. Read the FULL transcript and extract every safety-critical event, even if subtle. \
Focus on: {focus}.

Event types:
{_types(profile, consult=False)}

Be thorough - it is worse to miss a complication or a wrong count than to over-report.
Return JSON: {{"events":[{{"type","payload","evidence_turn_ids":["t0012"],"confidence":0-1}}]}}"""


def _coerce_seconds(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(v)
    except ValueError:
        return None


def _windows(turns: list[Turn]) -> list[list[Turn]]:
    out, i = [], 0
    while i < len(turns):
        out.append(turns[i : i + WINDOW_TURNS])
        i += WINDOW_TURNS - WINDOW_OVERLAP
    return out


def _fmt(turns: list[Turn]) -> str:
    def tag(t: Turn) -> str:
        who = t.role or t.speaker
        return f"{who}: " if who else ""

    return "\n".join(f"{t.id} [{t.clock}] {tag(t)}{t.text}" for t in turns)


def _attribute_roles(events, turns) -> None:
    by_id = {t.id: t for t in turns}
    for e in events:
        roles = [by_id[i].role for i in e.evidence_turn_ids if i in by_id and by_id[i].role]
        if roles:
            e.by_role = Counter(roles).most_common(1)[0][0]


class EventAgent(Agent):
    name = "events"
    requires = ("turns",)
    produces = "events"

    async def run(self, cf: CaseFile) -> CaseFile:
        profile = active_profile()
        sem = asyncio.Semaphore(int(os.environ.get("PI_CONCURRENCY", "1")))

        async def guarded(sys_prompt, turns, tag):
            async with sem:
                return await self._extract(sys_prompt, turns, tag)

        # Consultations are short and finding-dense: coherent non-overlapping passes over the
        # transcript avoid the paraphrase duplicates a windowed + sweep pass would produce.
        consult = "finding" in (profile.event_focus or []) or "consult" in profile.care_setting
        if consult:
            avg_len = max(1, len(_fmt(cf.turns)) // max(1, len(cf.turns)))
            step = max(40, _MAX_PROMPT_CHARS // avg_len)   # ~one prompt-sized chunk, no overlap
            chunks = [cf.turns[i : i + step] for i in range(0, len(cf.turns), step)] or [[]]
            raw = [e for batch in await asyncio.gather(
                *[guarded(system_prompt(profile), c, f"c{i}") for i, c in enumerate(chunks)]
            ) for e in batch]
            passes = f"{len(chunks)} consult pass(es)"
        else:
            wins = _windows(cf.turns)
            jobs = [guarded(system_prompt(profile), w, f"w{i}") for i, w in enumerate(wins)]
            sweep_chunks = [cf.turns[i : i + SWEEP_CHUNK] for i in range(0, len(cf.turns), SWEEP_CHUNK)] or [[]]
            jobs += [guarded(sweep_prompt(profile), c, f"sweep{i}") for i, c in enumerate(sweep_chunks)]
            raw = [e for batch in await asyncio.gather(*jobs) for e in batch]
            passes = f"{len(wins)} windows + {len(sweep_chunks)} sweep"

        cf.events = _dedupe(raw)
        _attribute_roles(cf.events, cf.turns)
        cf.log(self.name, f"{info()}: {passes} -> {len(raw)} raw -> {len(cf.events)} events (profile={profile.id})")
        return cf

    async def _extract(self, sys_prompt: str, window: list[Turn], tag: str) -> list[ProceduralEvent]:
        body = _fmt(window)
        if len(body) > _MAX_PROMPT_CHARS:  # keep the request under Groq's size cap
            body = body[:_MAX_PROMPT_CHARS] + "\n... [chunk truncated]"
        user = f"TRANSCRIPT:\n{body}"
        try:
            data = await asyncio.to_thread(complete_json, sys_prompt, user)
        except Exception as exc:  # noqa: BLE001 - one bad window shouldn't kill the run
            print(f"  [events] {tag} failed: {exc}")
            return []
        valid_ids = {t.id for t in window}
        events: list[ProceduralEvent] = []
        for j, item in enumerate(data.get("events", []) if isinstance(data, dict) else []):
            try:
                etype = EventType(item["type"])
            except (KeyError, ValueError):
                continue
            ev_ids = [i for i in item.get("evidence_turn_ids", []) if i in valid_ids]
            by_id = {t.id: t for t in window}
            if ev_ids:  # trust our own turn timing over the model's
                t0 = min(by_id[i].start_s for i in ev_ids)
            else:
                t0 = _coerce_seconds(item.get("t_start_s"))
            events.append(
                ProceduralEvent(
                    id=f"e_{tag}_{j}",
                    t_start_s=float(t0 if t0 is not None else window[0].start_s),
                    type=etype,
                    payload=item.get("payload", {}) or {},
                    evidence_turn_ids=ev_ids,
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return events


# types where two nearby mentions are the same real event even if worded differently
_FUZZY = {
    EventType.complication,
    EventType.hemodynamic_event,
    EventType.conversion,
    EventType.count_status,
    EventType.disposition,
    EventType.incision,
    EventType.equipment_issue,
    EventType.specimen,
}


def _slug(*vals) -> str:
    raw = " ".join(str(v) for v in vals if v).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in raw).split())


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_FINDING_STOP = {
    "the", "and", "for", "with", "has", "have", "been", "was", "were", "not", "any", "day",
    "days", "week", "weeks", "ago", "last", "since", "past", "also", "other", "some", "this",
    "that", "his", "her", "their", "patient", "reports", "symptom", "symptoms", "history",
    "described", "onset", "started", "start", "feeling", "felt", "about", "over", "than",
    "now", "still", "when", "which",
}


def _finding_key(desc: str) -> str:
    words = [w for w in _slug(desc).split() if len(w) >= 4 and w not in _FINDING_STOP]
    return "find:" + " ".join(sorted(set(words))[:4])


def _dedupe_key(e: ProceduralEvent) -> str:
    """A loose identity for an event, so near-duplicate phrasings collapse."""
    p = e.payload or {}
    t = e.type
    if t == EventType.finding:
        return _finding_key(p.get("description", ""))
    if t == EventType.medication_given:
        return "med:" + _slug(p.get("name"))
    if t in (EventType.line_placed, EventType.drain_placed):
        return f"{t.value}:" + (_slug(p.get("site")) or _slug(p.get("type")))[:20]
    if t == EventType.implant_placed:
        return "impl:" + _slug(p.get("device"))[:24]
    if t == EventType.transfusion:
        return "txf:" + _slug(p.get("product"))
    if t == EventType.phase_transition:
        return "phase:" + _slug(p.get("phase"))
    if t == EventType.blood_loss:
        return "ebl:" + str(p.get("ebl_ml"))  # distinct EBL numbers are distinct events
    if t in _FUZZY:
        return t.value  # collapse any nearby same-type narrative event
    return t.value + ":" + _slug(*p.values())


def _dedupe(events: list[ProceduralEvent]) -> list[ProceduralEvent]:
    """Windowed pass + sweep pass re-emit the same events. Collapse them."""
    events.sort(key=lambda e: (e.t_start_s, -e.confidence))
    kept: list[ProceduralEvent] = []
    for e in events:
        ekey = _dedupe_key(e)
        dup = None
        for k in kept:
            if k.type != e.type:
                continue
            if e.type == EventType.transfusion:
                window = 1e9  # one row per product; running totals collapse
            elif e.type in _FUZZY or e.type == EventType.medication_given:
                window = 240
            else:
                window = 45
            if abs(k.t_start_s - e.t_start_s) > window:
                continue
            if _dedupe_key(k) == ekey:
                dup = k
                break
        if dup:
            dup.evidence_turn_ids = sorted(set(dup.evidence_turn_ids) | set(e.evidence_turn_ids))
            dup.confidence = max(dup.confidence, e.confidence)
            if e.type == EventType.transfusion:
                du, eu = _num(dup.payload.get("units")), _num(e.payload.get("units"))
                if eu is not None and (du is None or eu > du):
                    dup.payload, dup.t_start_s = e.payload, e.t_start_s
            elif len(json.dumps(e.payload)) > len(json.dumps(dup.payload)):
                dup.payload = e.payload  # keep the more detailed description
        else:
            kept.append(e)
    kept.sort(key=lambda e: e.t_start_s)
    for n, e in enumerate(kept):
        e.id = f"e{n:03d}"
    return kept
