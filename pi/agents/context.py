"""turns -> CaseContext.

If a clinical-context MCP server is connected, the patient set-up (procedure, indication,
anaesthesia plan, home meds, allergies, problem list) is pulled from it. Anything the MCP
server does not supply is inferred from the transcript by one LLM call. Each field records
its source in `context.sources`.
"""

from __future__ import annotations

import asyncio

from ..casefile import CaseFile
from ..llm import complete_json, info
from ..mcp_client import load_pool
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
        ctx = CaseContext()
        head = cf.turns[:25]
        tail = cf.turns[-6:]
        excerpt = "\n".join(f"{t.id} [{t.clock}] {t.text}" for t in head + tail)

        pool = load_pool()
        if pool.enabled:
            try:
                async with pool as p:
                    await self._from_mcp(p, ctx, cf.turns)
            except Exception as exc:  # noqa: BLE001
                print(f"  [context] mcp lookup failed: {exc}")

        missing = [f for f in ("patient_descriptor", "planned_procedure", "indication", "anesthesia_type")
                   if not getattr(ctx, f)]
        if missing:
            try:
                data = await asyncio.to_thread(complete_json, SYSTEM, excerpt) or {}
                for f in missing:
                    if data.get(f):
                        setattr(ctx, f, data[f])
                        ctx.sources[f] = "transcript"
                if data.get("evidence_turn_ids"):
                    ctx.evidence_turn_ids = data["evidence_turn_ids"]
            except Exception as exc:  # noqa: BLE001
                print(f"  [context] llm failed: {exc}")

        cf.context = ctx
        src = "+".join(sorted(set(ctx.sources.values()))) or "none"
        cf.log(self.name, f"{info()}: sources={src} {ctx.model_dump(exclude_none=True, exclude={'sources', 'evidence_turn_ids'})}")
        return cf

    async def _from_mcp(self, pool, ctx: CaseContext, turns) -> None:
        if not pool.has("lookup_patient"):
            return
        hint = " ".join(t.text for t in turns[:6])[:400]
        res = await pool.call("lookup_patient", {"query": hint})
        match = (res or {}).get("match")
        if not match:
            return
        pid = match["patient_id"]
        ctx.patient_descriptor = match.get("descriptor")
        ctx.indication = match.get("indication")
        ctx.sources["patient_descriptor"] = "ehr"
        ctx.sources["indication"] = "ehr"

        if pool.has("get_scheduled_procedure"):
            sp = await pool.call("get_scheduled_procedure", {"patient_id": pid}) or {}
            if sp.get("procedure"):
                ctx.planned_procedure = sp["procedure"]
                ctx.sources["planned_procedure"] = "ehr"
            if sp.get("anaesthesia_plan"):
                ctx.anesthesia_type = sp["anaesthesia_plan"]
                ctx.sources["anesthesia_type"] = "ehr"
        if pool.has("get_allergies"):
            ctx.allergies = (await pool.call("get_allergies", {"patient_id": pid}) or {}).get("allergies", [])
            if ctx.allergies:
                ctx.sources["allergies"] = "ehr"
        if pool.has("get_active_medications"):
            ctx.home_medications = (await pool.call("get_active_medications", {"patient_id": pid}) or {}).get("home_medications", [])
            if ctx.home_medications:
                ctx.sources["home_medications"] = "ehr"
        if pool.has("get_problem_list"):
            ctx.problem_list = (await pool.call("get_problem_list", {"patient_id": pid}) or {}).get("problem_list", [])
            if ctx.problem_list:
                ctx.sources["problem_list"] = "ehr"
