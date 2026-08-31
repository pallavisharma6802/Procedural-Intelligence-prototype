from __future__ import annotations

from ..profile import SiteProfile
from ._projection import ProjectionAgent

_DEFAULT_SECTIONS = [
    "Preoperative diagnosis", "Postoperative diagnosis", "Procedure", "Surgeon / anesthesia",
    "Anesthesia type", "Findings", "Implants", "Estimated blood loss", "Specimens",
    "Complications", "Counts", "Disposition",
]


class OpNoteAgent(ProjectionAgent):
    name = "opnote"
    kind = "opnote"

    def build_system(self, profile: SiteProfile) -> str:
        sections = profile.opnote_sections or _DEFAULT_SECTIONS
        headings = "\n".join(f"{s}:" for s in sections)
        gloss = ("\n" + profile.glossary()) if profile.terminology else ""
        return (
            "You draft a brief procedure / operative note from the reconstructed timeline. "
            "Use exactly these headings, each on its own line, in this order:\n\n"
            f"{headings}\n"
            "Brief narrative: 3-6 sentences, chronological, only from the timeline.\n\n"
            "Write 'not documented' for any heading the timeline does not support, and "
            "'none documented' under Complications if there were none. This is a DRAFT for "
            "clinician review - do not invent operative detail that is not in the timeline."
            f"{gloss}"
        )
