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


def _types(profile: SiteProfile) -> str:
    phases = " | ".join(profile.phases)
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
    return f"""You are a clinical scribe extracting a structured procedural timeline from a \
{profile.care_setting.replace("_", " ")} transcript window. Return ONLY events that matter for a \
handoff, a procedure note, or a family update. Ignore small talk, equipment banter and logistics.

Event types (use exactly these):
{_types(profile)}

Guidance:
- A `complication` is any unintended adverse event: bleeding beyond what is expected, organ/vessel
  injury, inability to obtain the critical view, hemodynamic instability needing intervention,
  unplanned conversion. When surgery is converted lap->open, emit BOTH a `conversion` event AND a
  `complication` describing why.
- A drain (Jackson-Pratt, Blake, chest tube) is `drain_placed`, never `implant_placed`.
- Emit a fresh `blood_loss` event each time an EBL number is stated.

Return JSON: {{"events":[{{"type","payload","evidence_turn_ids":["t0007"],"confidence":0-1}}]}}
Return {{"events":[]}} if the window has nothing clinically meaningful.

EXAMPLE
window:
  t0004 [00:05:30] SURGEON: Let's do our time out...
  t0005 [00:05:52] ANESTHESIA: Cefazolin two grams IV went in about ten minutes ago.
  t0013 [00:24:00] SURGEON: I'm not comfortable continuing laparoscopically, converting to open.
  t0024 [00:52:00] SURGEON: Placing a Jackson-Pratt drain in the fossa.
output:
{{"events":[
 {{"type":"phase_transition","payload":{{"phase":"timeout"}},"evidence_turn_ids":["t0004"],"confidence":0.9}},
 {{"type":"medication_given","payload":{{"name":"cefazolin","dose":"2 g","route":"IV"}},"evidence_turn_ids":["t0005"],"confidence":0.95}},
 {{"type":"conversion","payload":{{"from":"laparoscopic","to":"open"}},"evidence_turn_ids":["t0013"],"confidence":0.9}},
 {{"type":"complication","payload":{{"description":"unable to proceed laparoscopically, converted to open"}},"evidence_turn_ids":["t0013"],"confidence":0.8}},
 {{"type":"drain_placed","payload":{{"type":"Jackson-Pratt","site":"gallbladder fossa"}},"evidence_turn_ids":["t0024"],"confidence":0.9}}
]}}"""


def sweep_prompt(profile: SiteProfile) -> str:
    focus = ", ".join(profile.event_focus) or (
        "complication, conversion, blood_loss, hemodynamic_event, transfusion, count_status, "
        "specimen, disposition, medication_given"
    )
    return f"""You are a procedural safety auditor for a {profile.care_setting.replace("_", " ")} \
case. Read the FULL transcript and extract every safety-critical event, even if subtle. \
Focus on: {focus}.

Event types:
{_types(profile)}

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
        sys_win = system_prompt(profile)
        sys_sweep = sweep_prompt(profile)
        wins = _windows(cf.turns)
        sem = asyncio.Semaphore(int(os.environ.get("PI_CONCURRENCY", "1")))

        async def guarded(sys_prompt, turns, tag):
            async with sem:
                return await self._extract(sys_prompt, turns, tag)

        jobs = [guarded(sys_win, w, f"w{i}") for i, w in enumerate(wins)]
        # whole-transcript safety pass, chunked so the prompt stays bounded on long cases
        sweep_chunks = [cf.turns[i : i + SWEEP_CHUNK] for i in range(0, len(cf.turns), SWEEP_CHUNK)] or [[]]
        jobs += [guarded(sys_sweep, c, f"sweep{i}") for i, c in enumerate(sweep_chunks)]
        results = await asyncio.gather(*jobs)
        raw = [e for batch in results for e in batch]
        cf.events = _dedupe(raw)
        _attribute_roles(cf.events, cf.turns)
        cf.log(
            self.name,
            f"{info()}: {len(wins)} windows + {len(sweep_chunks)} sweep -> {len(raw)} raw -> "
            f"{len(cf.events)} events (profile={profile.id})",
        )
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


def _norm_payload(p: dict) -> str:
    def low(v):
        return v.lower().strip() if isinstance(v, str) else v

    return json.dumps({k: low(v) for k, v in sorted(p.items())})


def _dedupe(events: list[ProceduralEvent]) -> list[ProceduralEvent]:
    """Windowed pass + sweep pass re-emit the same events. Collapse them."""
    events.sort(key=lambda e: (e.t_start_s, -e.confidence))
    kept: list[ProceduralEvent] = []
    for e in events:
        same_payload = _norm_payload(e.payload)
        dup = None
        for k in kept:
            if k.type != e.type:
                continue
            window = 180 if e.type in _FUZZY else 30
            if abs(k.t_start_s - e.t_start_s) > window:
                continue
            if e.type in _FUZZY or _norm_payload(k.payload) == same_payload:
                dup = k
                break
        if dup:
            dup.evidence_turn_ids = sorted(set(dup.evidence_turn_ids) | set(e.evidence_turn_ids))
            dup.confidence = max(dup.confidence, e.confidence)
            if len(json.dumps(e.payload)) > len(json.dumps(dup.payload)):
                dup.payload = e.payload  # keep the more detailed description
        else:
            kept.append(e)
    kept.sort(key=lambda e: e.t_start_s)
    for n, e in enumerate(kept):
        e.id = f"e{n:03d}"
    return kept
