"""Shared machinery for the three draft-generating agents (handoff / opnote / family).

Each agent's system prompt is *built from the active SiteProfile* — the pipeline code is the
same for every hospital; only the profile (handoff format, note headings, terminology, family
style) changes.
"""

from __future__ import annotations

import asyncio
import json

from ..casefile import CaseFile
from ..llm import complete, info
from ..profile import SiteProfile, active_profile
from ..schemas import CaseState, Draft, ProceduralEvent
from .base import Agent


def timeline_digest(events: list[ProceduralEvent], turns_by_id: dict) -> str:
    lines = []
    for e in events:
        ev_txt = "; ".join(turns_by_id[i].text for i in e.evidence_turn_ids if i in turns_by_id)
        if len(ev_txt) > 140:
            ev_txt = ev_txt[:137] + "..."
        lines.append(f"[{e.clock}] {e.type.value} {json.dumps(e.payload)}  <- {ev_txt}")
    return "\n".join(lines)


def state_digest(s: CaseState) -> str:
    d = s.model_dump(exclude={"provenance", "transfusions", "disposition_path"})
    return json.dumps(d, indent=2, default=str)


class ProjectionAgent(Agent):
    """Subclasses implement `build_system(profile)`. `run` and the critic both call it."""

    kind: str = ""
    requires = ("states", "events")
    produces = "drafts"
    temperature = 0.3

    def build_system(self, profile: SiteProfile) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def system_prompt(self) -> str:
        return self.build_system(active_profile())

    async def run(self, cf: CaseFile) -> CaseFile:
        final = cf.final_state()
        tbi = {t.id: t for t in cf.turns}
        ctx = cf.context.model_dump(exclude_none=True, exclude={"evidence_turn_ids"}) if cf.context else {}
        user = (
            f"CASE CONTEXT:\n{json.dumps(ctx, indent=2) or '{}'}\n\n"
            f"CASE STATE (final):\n{state_digest(final)}\n\n"
            f"EVENT TIMELINE (with transcript evidence):\n{timeline_digest(cf.events, tbi)}\n\n"
            "Write the document now. Use only facts present above. If something a section "
            "normally contains is unknown, write 'not documented'."
        )
        try:
            text = await asyncio.to_thread(
                complete, self.system_prompt(), user, temperature=self.temperature
            )
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the pipeline
            print(f"  [{self.name}] generation failed: {exc}")
            cf.drafts[self.kind] = Draft(
                kind=self.kind,
                text=f"[{self.kind} not generated — LLM call failed: {exc}]",
                unsupported_claims=["draft not generated"],
            )
            cf.log(self.name, f"generation FAILED: {exc}")
            return cf
        cf.drafts[self.kind] = Draft(kind=self.kind, text=text.strip())
        cf.log(self.name, f"{info()}: drafted {self.kind} ({len(text)} chars, profile={active_profile().id})")
        return cf
