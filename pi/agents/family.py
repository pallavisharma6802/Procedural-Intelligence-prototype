from __future__ import annotations

from ..profile import SiteProfile
from ._projection import ProjectionAgent


class FamilyAgent(ProjectionAgent):
    name = "family"
    kind = "family"
    temperature = 0.3

    def build_system(self, profile: SiteProfile) -> str:
        f = profile.family
        extra = ("\n- " + f.notes) if f.notes else ""
        return (
            f"You write a brief status update for the patient's family, in {f.language}, "
            f"readable by a non-medical person at {f.reading_level}.\n\n"
            "This is a waiting-room update, not a summary of the operation. Do NOT narrate what "
            "happened during surgery. Do NOT mention: individual surgical steps, a switch from one "
            "approach to another, blood loss, drains, tubes, lines, instrument counts, medications, "
            "equipment, or which unit the patient moves to next.\n\n"
            "Include only:\n"
            f"- {f.sentences} sentences, plain language, no jargon, no numbers.\n"
            "- The operation that was done, named simply (e.g. \"surgery on the gallbladder\", "
            "\"a knee replacement\").\n"
            "- Where the patient is now, in general terms only: \"in the recovery area\" for a "
            "routine case; \"in the intensive care unit\" only if the case was genuinely critical.\n"
            "- If a complication occurred, say exactly: \"there was a difficulty during the "
            "procedure that the team managed\" - nothing more specific.\n"
            "- If the case was genuinely serious (major bleeding, critical condition, breathing "
            "machine, a planned return to the operating room), state clearly and calmly that the "
            "patient is very sick and in critical condition and the team is doing everything "
            "possible. Do not downplay it and do not make it graphic. If items were deliberately "
            "left in place for a planned next operation, say only that the surgery is being done "
            "in stages.\n"
            "- If the case was routine and uncomplicated, say so plainly and reassuringly. Do not "
            "invent difficulties.\n"
            "- Never speculate about prognosis or recovery time. Never state anything not "
            "supported by the case."
            f"{extra}\n"
            f"- End with exactly: \"{f.closing_line}\""
        )
