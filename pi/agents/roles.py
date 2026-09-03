"""Attribute each transcript line to a canonical clinical role.

- `.srt` with explicit "SURGEON:" style prefixes  -> normalised directly, no LLM
- diarized audio ("A"/"B"/"SPEAKER_01")           -> one LLM call over a sample per speaker
- plain audio, no speaker labels                  -> one LLM call labelling every line by content
"""

from __future__ import annotations

import asyncio
from collections import Counter

from ..casefile import CaseFile
from ..llm import complete_json, info
from ..profile import active_profile
from ..schemas import ROLES
from .base import Agent

_DIRECT = {
    "surgeon": "surgeon", "attending": "surgeon", "consultant": "surgeon",
    "operator": "surgeon", "cardiologist": "surgeon", "proceduralist": "surgeon",
    "assistant": "assistant", "registrar": "assistant", "resident": "assistant",
    "fellow": "assistant", "pa": "assistant", "first assist": "assistant",
    "anesthesia": "anesthesia", "anaesthesia": "anesthesia", "anesthesiologist": "anesthesia",
    "anaesthetist": "anesthesia", "crna": "anesthesia", "odp": "anesthesia",
    "circulator": "circulating_nurse", "circulating": "circulating_nurse",
    "circulating nurse": "circulating_nurse", "runner": "circulating_nurse",
    "nurse": "circulating_nurse", "theatre nurse": "circulating_nurse",
    "scrub": "scrub_nurse", "scrub nurse": "scrub_nurse", "scrub tech": "scrub_nurse",
    "tech": "scrub_nurse", "instrument nurse": "scrub_nurse",
    "perfusion": "perfusionist", "perfusionist": "perfusionist",
}

SYSTEM = f"""You are labelling who is speaking in an operating-room transcript. You are given a
few sample lines from each raw speaker id. Assign each speaker id to ONE role from:
{", ".join(ROLES)}.

Clues: the person running the time-out / calling the operative steps / asking for instruments is
usually the surgeon; the one reporting blood pressure, heart rate, drugs and airway is anesthesia;
the one reporting counts, fetching supplies and talking to the room is a circulating nurse; the
one passing instruments is a scrub nurse. If genuinely unclear, use "other".

Return JSON: {{"roles": {{"<speaker id>": "<role>", ...}}}}"""

_OR_ROLES = [r for r in ROLES if r not in ("clinician", "patient")]
_CONSULT_ROLES = ["clinician", "patient", "other"]

SYSTEM_PERLINE_OR = f"""You are attributing each line of an operating-room transcript to the person
most likely saying it. Roles: {", ".join(_OR_ROLES)}.

Clues: the person running the time-out / calling operative steps / asking for instruments is the
surgeon; whoever reports blood pressure, heart rate, drugs and the airway is anesthesia; whoever
reports instrument/sponge counts, fetches supplies and updates the room is a circulating nurse;
whoever passes instruments is a scrub nurse. Use "other" only when there is no signal.

You get numbered lines. Return JSON {{"roles": {{"0": "<role>", "1": "<role>", ...}}}} with an
entry for every line number."""

SYSTEM_PERLINE_CONSULT = f"""You are attributing each line of a clinical consultation transcript to
the speaker. Roles: {", ".join(_CONSULT_ROLES)}.

The clinician asks the questions, summarises, explains the diagnosis, and gives the plan and
safety-netting advice. The patient describes their symptoms, history and concerns and answers
questions. Use "other" only for a third party (e.g. a relative).

You get numbered lines. Return JSON {{"roles": {{"0": "<role>", "1": "<role>", ...}}}} with an
entry for every line number."""


class RolesAgent(Agent):
    name = "roles"
    requires = ("turns",)
    produces = "turns"

    async def run(self, cf: CaseFile) -> CaseFile:
        speakers = [t.speaker for t in cf.turns if t.speaker]
        if not speakers:
            if cf.turns:
                await self._infer_per_line(cf)
            else:
                cf.log(self.name, "no turns - skipped")
            return cf

        mapping: dict[str, str] = {}
        unresolved: list[str] = []
        for spk in dict.fromkeys(speakers):
            direct = _DIRECT.get(spk.strip().lower())
            if direct:
                mapping[spk] = direct
            else:
                unresolved.append(spk)

        if unresolved:
            mapping.update(await self._infer(cf, unresolved))

        for t in cf.turns:
            if t.speaker:
                t.role = mapping.get(t.speaker, t.role or "other")
        counts = Counter(t.role for t in cf.turns if t.role)
        cf.log(self.name, f"roles: {dict(counts)}")
        return cf

    async def _infer_per_line(self, cf: CaseFile) -> None:
        prof = active_profile()
        consult = "finding" in (prof.event_focus or []) or "consult" in prof.care_setting
        sys_prompt = SYSTEM_PERLINE_CONSULT if consult else SYSTEM_PERLINE_OR
        lines = "\n".join(f"{i}: {t.text}" for i, t in enumerate(cf.turns))
        user = f"CARE SETTING: {prof.care_setting}\n\nLINES:\n{lines}"
        try:
            data = await asyncio.to_thread(complete_json, sys_prompt, user)
            raw = data.get("roles", {}) if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            cf.log(self.name, f"per-line inference failed: {exc}")
            return
        for i, t in enumerate(cf.turns):
            r = raw.get(str(i))
            t.role = r if r in ROLES else (t.role or "other")
        counts = Counter(t.role for t in cf.turns if t.role)
        cf.log(self.name, f"{info()}: per-line roles {dict(counts)}")

    async def _infer(self, cf: CaseFile, speakers: list[str]) -> dict[str, str]:
        by_spk: dict[str, list[str]] = {s: [] for s in speakers}
        for t in cf.turns:
            if t.speaker in by_spk and len(by_spk[t.speaker]) < 4:
                by_spk[t.speaker].append(t.text)
        sample = "\n\n".join(
            f"speaker {s}:\n" + "\n".join(f"  - {x}" for x in lines) for s, lines in by_spk.items()
        )
        user = f"CARE SETTING: {active_profile().care_setting}\n\nSAMPLES:\n{sample}"
        try:
            data = await asyncio.to_thread(complete_json, SYSTEM, user)
            raw = data.get("roles", {}) if isinstance(data, dict) else {}
            return {s: (raw.get(s) if raw.get(s) in ROLES else "other") for s in speakers}
        except Exception as exc:  # noqa: BLE001
            print(f"  [roles] inference failed: {exc}")
            return {s: "other" for s in speakers}
