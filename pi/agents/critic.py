"""Fact-checks the drafts against the timeline. Flags fabricated facts only, then revises once.

One combined check call covers all drafts (keeps free-tier token use low); only the drafts
that get flagged are revised, then re-checked in a second combined call.
"""

from __future__ import annotations

import asyncio
import difflib

from ..casefile import CaseFile
from ..llm import CRITIC_MODEL, complete, complete_json, info
from ._projection import ProjectionAgent, state_digest, timeline_digest
from .base import Agent

SYSTEM = """You are a clinical fact-checker for auto-generated OR documents. You receive a SOURCE
(case context + reconstructed state + event timeline with transcript quotes) and one or more
DOCUMENTS. Flag ONLY fabrications: statements of clinical FACT that the source neither states nor
directly implies.

A FACT = a vital sign, lab value, medication / dose / route, a specific EBL number, an event that
occurred, a time, an operative finding, personnel present, patient history.

NEVER flag:
- Recommendations / next steps / monitoring plans ("watch for X", "notify surgery if Y",
  "obtain CBC per protocol") - that is standard-of-care guidance, not a claim about what happened.
- Required template / boilerplate lines ("The surgeon will speak with you directly",
  "Receiver read-back:", section headers).
- "not documented" / "none documented" / "not specified".
- Rewording, summarizing, or reasonable inference from facts that ARE in the source
  (e.g. "converted to open" implies an open incision was made; "acute cholecystitis" as the
  indication may appear as the pre-op diagnosis; patient sex inferred from a "he"/"she" pronoun
  in the transcript; illness-severity labels like "stable" / "watcher" / "unstable").

Flag a statement only if a clinician comparing it to the source would say "that specific fact is
invented" or "that contradicts the source".

OUTPUT RULES:
- Do NOT include your deliberation. Each "reason" is ONE short sentence, max 20 words.
- If you consider flagging something and then decide it is acceptable, do NOT include it at all.
- "quote" must be text copied verbatim from that document.

Return JSON:
{"results": {"<doc_name>": [ {"quote": "<verbatim from that document>", "reason": "<one sentence>"} ], ...}}
Every doc name you were given must appear as a key, with [] if it is clean."""


_NEGATION = (
    "will not flag",
    "not flag this",
    "reasonable inference",
    "not a fabrication",
    "is acceptable",
    "this is fine",
    "i will not",
    "actually,",
    "on second thought",
    "supported by the source",
    "is correct",
    "this is correct",
    "matches the source",
    "consistent with the source",
    "no fabrication",
)


def _self_negated(reason: str) -> bool:
    low = reason.lower()
    return any(p in low for p in _NEGATION)


def _not_generated(draft) -> bool:
    return draft.text.startswith("[") and "not generated" in draft.text[:60]


def _fmt_docs(drafts: dict) -> str:
    return "\n\n".join(f"===== DOCUMENT: {k} =====\n{d.text}" for k, d in drafts.items())


class CriticAgent(Agent):
    name = "critic"
    requires = ("drafts",)
    produces = "drafts"

    #: which projection agents to re-run on failure, injected by the orchestrator
    generators: dict[str, ProjectionAgent] = {}

    # ---- pieces the orchestrator graph drives one at a time ------------------
    def build_source(self, cf: CaseFile) -> str:
        tbi = {t.id: t for t in cf.turns}
        ctx = cf.context.model_dump(exclude_none=True, exclude={"evidence_turn_ids"}) if cf.context else {}
        timeline = timeline_digest(cf.events, tbi)
        if len(timeline) > 8000:
            timeline = timeline[:8000] + "\n... [timeline truncated]"
        return (
            f"CASE CONTEXT:\n{ctx}\n\n"
            f"CASE STATE:\n{state_digest(cf.final_state())}\n\n"
            f"TIMELINE:\n{timeline}"
        )

    def checkable(self, cf: CaseFile) -> dict:
        keep = {}
        for k, d in cf.drafts.items():
            if _not_generated(d):
                d.accepted = False
                d.unsupported_claims = ["draft not generated (LLM call failed) - rerun this stage"]
                cf.log(self.name, f"{k}: skipped - not generated")
            else:
                keep[k] = d
        return keep

    async def check(self, cf: CaseFile, source: str) -> dict[str, list[str]] | None:
        drafts = self.checkable(cf)
        return await self._check(source, drafts) if drafts else {}

    async def revise(self, cf: CaseFile, source: str, flagged: dict[str, list[str]]) -> None:
        to_revise = {k: v for k, v in flagged.items() if v and k in self.generators and k in cf.drafts}
        if not to_revise:
            return
        cf.log(self.name, "revising: " + ", ".join(f"{k}({len(v)})" for k, v in to_revise.items()))
        await asyncio.gather(*(self._revise(cf, k, source, to_revise[k]) for k in to_revise))
        for k in to_revise:
            cf.drafts[k].revised = True

    def finalize(self, cf: CaseFile, results: dict[str, list[str]] | None) -> None:
        drafts = self.checkable(cf)
        if results is None:
            for d in drafts.values():
                d.unsupported_claims = ["critic did not run - draft NOT fact-checked"]
                d.accepted = False
            cf.log(self.name, "check unavailable - drafts marked NOT fact-checked")
            return
        for k, draft in drafts.items():
            draft.unsupported_claims = results.get(k, [])
            draft.accepted = not draft.unsupported_claims
            cf.log(self.name, f"{k}: accepted={draft.accepted} flags={len(draft.unsupported_claims)}")

    async def run(self, cf: CaseFile) -> CaseFile:
        """Standalone path (used by `pi stage critic`): check -> revise once -> re-check."""
        source = self.build_source(cf)
        results = await self.check(cf, source)
        flagged = {k: v for k, v in (results or {}).items() if v}
        if flagged:
            await self.revise(cf, source, flagged)
            recheck = await self.check(cf, source)
            if recheck is not None:
                results = recheck
        self.finalize(cf, results)
        return cf

    async def _check(self, source: str, drafts: dict) -> dict[str, list[str]] | None:
        """Check the given drafts. One combined call when it fits, else one call per draft."""
        combined = f"SOURCE:\n{source}\n\n{_fmt_docs(drafts)}"
        if len(combined) <= 12000 and len(drafts) > 1:
            return await self._check_call(SYSTEM, combined, drafts)
        # too big for one request (or a single draft) - check each on its own, merge
        out: dict[str, list[str]] = {}
        any_ok = False
        for k, d in drafts.items():
            one = await self._check_call(SYSTEM, f"SOURCE:\n{source}\n\n{_fmt_docs({k: d})}", {k: d})
            if one is None:
                out.setdefault(k, [])
            else:
                any_ok = True
                out.update(one)
        return out if any_ok else None

    async def _check_call(self, system: str, user: str, drafts: dict) -> dict[str, list[str]] | None:
        try:
            data = await asyncio.to_thread(complete_json, system, user, model=CRITIC_MODEL)
        except Exception as exc:  # noqa: BLE001
            print(f"  [critic] check failed: {exc}")
            return None
        raw = (data or {}).get("results", data) if isinstance(data, dict) else {}
        out: dict[str, list[str]] = {}
        for kind, draft in drafts.items():
            items = raw.get(kind, []) if isinstance(raw, dict) else []
            issues = []
            for x in items if isinstance(items, list) else []:
                if isinstance(x, dict):
                    q, r = str(x.get("quote", "")).strip(), str(x.get("reason", "")).strip()
                elif str(x).strip():
                    q, r = str(x).strip(), ""
                else:
                    continue
                if q and not _appears_in(q, draft.text):
                    continue  # critic invented a quote that isn't in the doc - drop it
                if _self_negated(r):
                    continue  # critic talked itself out of this one
                r = r.split(". ")[0][:200]  # first sentence, in case it rambled
                issues.append(f"{q} - {r}".strip(" -") if q else r)
            out[kind] = [i for i in issues if i]
        return out

    async def _revise(self, cf: CaseFile, kind: str, source: str, issues: list[str]) -> None:
        gen = self.generators[kind]
        bullet = "\n".join(f"- {i}" for i in issues)
        user = (
            f"{source}\n\nYour previous draft stated facts NOT supported by the source:\n{bullet}\n\n"
            f"Rewrite the {kind} document. Remove or correct each unsupported fact - replace an "
            "unknowable specific with 'not documented'. Keep everything else, including standard "
            "monitoring guidance and required template lines. Output only the document."
        )
        text = await asyncio.to_thread(complete, gen.system_prompt(), user, temperature=0.2)
        cf.drafts[kind].text = text.strip()
        cf.log(f"{self.name}:revise", f"{info()}: rewrote {kind}")


def _appears_in(quote: str, doc: str, threshold: float = 0.82) -> bool:
    q = " ".join(quote.lower().split())
    d = " ".join(doc.lower().split())
    if q in d:
        return True
    # tolerate minor paraphrase / truncation by the critic
    window = len(q)
    for i in range(0, max(1, len(d) - window + 1), max(1, window // 4)):
        if difflib.SequenceMatcher(None, q, d[i : i + window]).ratio() >= threshold:
            return True
    return False
