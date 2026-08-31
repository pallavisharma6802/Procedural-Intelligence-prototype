"""Site profile - everything that varies between hospitals and care settings.

The core pipeline is fixed: the same agent graph, the same `ProceduralEvent` vocabulary,
the same `CaseState` fields. A `SiteProfile` declares only the *local* conventions -
phase names, handoff format, note headings, family-letter style, terminology - so one
deployment differs from another by its profile, never by code.

Select one with `PI_PROFILE=<name-or-path>` (env) or `pi run --profile <name>`.
Built-ins live in `pi/profiles/*.json`; a path to any `.json` also works.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"

DEFAULT_PHASES = [
    "arrival",
    "setup",
    "timeout",
    "induction",
    "positioning_prep",
    "incision",
    "dissection",
    "key_procedure",
    "closure",
    "emergence",
    "handoff",
]


class HandoffSection(BaseModel):
    label: str
    guidance: str = ""


class HandoffFormat(BaseModel):
    name: str = "I-PASS"
    intro: str = "Write the receiving-team handoff. Be terse and factual; no invented vitals or labs."
    sections: list[HandoffSection] = Field(default_factory=list)


class FamilyStyle(BaseModel):
    language: str = "English"
    reading_level: str = "about a 7th-grade level"
    sentences: str = "4-6"
    closing_line: str = "The surgeon will come speak with you directly."
    notes: str = ""


class SiteProfile(BaseModel):
    id: str = "default_or"
    label: str = "Generic operating room (US)"
    care_setting: str = "operating_room"

    #: ordered phase vocabulary for this setting; the reducer only advances forward through it
    phases: list[str] = Field(default_factory=lambda: list(DEFAULT_PHASES))
    #: map loose/local phase words the model might emit onto a canonical phase
    phase_synonyms: dict[str, str] = Field(default_factory=dict)
    #: the phase an `incision` event bumps the case to (percutaneous settings: the access phase)
    procedure_start_phase: str = "incision"
    #: event types to emphasise in the safety sweep (names from schemas.EventType)
    event_focus: list[str] = Field(default_factory=list)

    handoff: HandoffFormat = Field(default_factory=HandoffFormat)
    opnote_sections: list[str] = Field(default_factory=list)
    family: FamilyStyle = Field(default_factory=FamilyStyle)

    #: canonical term -> the label this site uses (applied as a glossary in every draft prompt)
    terminology: dict[str, str] = Field(default_factory=dict)

    # ---- helpers ------------------------------------------------------
    def glossary(self) -> str:
        if not self.terminology:
            return ""
        pairs = "; ".join(f'say "{v}" not "{k}"' for k, v in self.terminology.items())
        return f"LOCAL TERMINOLOGY - {pairs}."

    def canonical_phase(self, name: str | None) -> str | None:
        if not name:
            return None
        return self.phase_synonyms.get(name, self.phase_synonyms.get(name.lower(), name))

    # ---- loading -----------------------------------------------------
    @classmethod
    def load(cls, name_or_path: Optional[str]) -> "SiteProfile":
        name_or_path = name_or_path or "default_or"
        p = Path(name_or_path)
        if p.suffix == ".json" and p.exists():
            return cls.model_validate_json(p.read_text())
        f = PROFILE_DIR / f"{name_or_path}.json"
        if f.exists():
            return cls.model_validate_json(f.read_text())
        if name_or_path == "default_or":
            return cls()  # usable even if the file is missing
        raise FileNotFoundError(
            f"unknown profile {name_or_path!r} - available: {', '.join(available())}"
        )


def available() -> list[str]:
    return sorted(f.stem for f in PROFILE_DIR.glob("*.json")) or ["default_or"]


_CACHE: dict[str, SiteProfile] = {}


def active_profile() -> SiteProfile:
    name = os.environ.get("PI_PROFILE", "default_or")
    if name not in _CACHE:
        _CACHE[name] = SiteProfile.load(name)
    return _CACHE[name]
