"""Map raw diarization speaker labels -> canonical clinical roles.

- `.srt` with explicit "SURGEON:" style prefixes  -> normalised directly, no LLM
- diarized audio ("A"/"B"/"SPEAKER_01")           -> one LLM call over a sample per speaker
- no speaker labels at all                        -> no-op
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


class RolesAgent(Agent):
    name = "roles"
    requires = ("turns",)
    produces = "turns"

    async def run(self, cf: CaseFile) -> CaseFile:
        speakers = [t.speaker for t in cf.turns if t.speaker]
        if not speakers:
            cf.log(self.name, "no speaker labels — skipped")
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
