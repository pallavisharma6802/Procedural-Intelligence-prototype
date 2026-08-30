from __future__ import annotations

from ..profile import SiteProfile
from ._projection import ProjectionAgent


class FamilyAgent(ProjectionAgent):
    name = "family"
    kind = "family"
    temperature = 0.4

    def build_system(self, profile: SiteProfile) -> str:
        f = profile.family
        extra = ("\n- " + f.notes) if f.notes else ""
        gloss = ("\n- " + profile.glossary()) if profile.terminology else ""
        return (
            f"You write a short, warm status update for the patient's family, in {f.language}, "
            f"readable by a non-medical person at {f.reading_level}.\n\n"
            "Rules:\n"
            f"- {f.sentences} sentences. No jargon, no numbers unless truly needed, no drug names or doses.\n"
            "- Say what was done and that the team is taking good care of them.\n"
            "- If a complication occurred, say \"there was a difficulty during the procedure that the "
            "team managed\" without alarming detail.\n"
            "- If the case was serious (major bleeding, critical condition, intensive care on a "
            "breathing machine, a planned return to the operating room), be honest and clear that "
            "the patient is very sick and in critical condition and the team is doing everything "
            "possible — calm and non-graphic, do not downplay it. Never use raw phrases like "
            "\"incorrect count\"; if items were deliberately left in for a planned next operation, "
            "say the surgery is being done in stages.\n"
            "- If the case was routine and uncomplicated, say so plainly and reassuringly — do not "
            "invent difficulties or hedge.\n"
            "- Never speculate about prognosis. Never state anything not supported by the timeline."
            f"{extra}{gloss}\n"
            f"- End with exactly: \"{f.closing_line}\""
        )
