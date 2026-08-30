"""turns -> CaseContext. One LLM call over the head + tail of the transcript."""

from __future__ import annotations

import asyncio

from ..casefile import CaseFile
from ..llm import complete_json, info
from ..schemas import CaseContext
from .base import Agent

SYSTEM = """From this OR transcript excerpt, extract the case set-up. Return JSON:
{"patient_descriptor": "e.g. 54-year-old man (age/sex only, NEVER a name)",
 "planned_procedure": "...",
 "indication": "preoperative diagnosis / reason for surgery",
 "anesthesia_type": "...",
 "evidence_turn_ids": ["t0001"]}
Use null for any field not stated. Do not guess."""


class ContextAgent(Agent):
    name = "context"
    requires = ("turns",)
    produces = "context"

    async def run(self, cf: CaseFile) -> CaseFile:
        head = cf.turns[:25]
        tail = cf.turns[-6:]
        excerpt = "\n".join(f"{t.id} [{t.clock}] {t.text}" for t in head + tail)
        try:
            data = await asyncio.to_thread(complete_json, SYSTEM, excerpt)
            clean = {k: v for k, v in (data or {}).items() if k in CaseContext.model_fields and v}
            cf.context = CaseContext(**clean)
        except Exception as exc:  # noqa: BLE001
            print(f"  [context] failed: {exc}")
            cf.context = CaseContext()
        cf.log(self.name, f"{info()}: {cf.context.model_dump(exclude_none=True)}")
        return cf
