from __future__ import annotations

from ..profile import SiteProfile
from ._projection import ProjectionAgent


class HandoffAgent(ProjectionAgent):
    name = "handoff"
    kind = "handoff"

    def build_system(self, profile: SiteProfile) -> str:
        h = profile.handoff
        sections = "\n".join(f"**{s.label}**: {s.guidance}" for s in h.sections)
        gloss = ("\n" + profile.glossary()) if profile.terminology else ""
        return (
            f"{h.intro}\n\n"
            f"Structure ({h.name}), each section on its own line:\n{sections}\n\n"
            "Include every fact the timeline supports - findings, medications, EBL, transfusions, "
            "lines, drains, implants, counts - and nothing it does not. If the case context lists "
            "allergies, state them. Never invent a vital sign, lab value or examination finding."
            f"{gloss}"
        )
