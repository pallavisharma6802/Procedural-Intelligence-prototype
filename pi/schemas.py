"""Core data model. The ProceduralEvent log is the source of truth; CaseState is derived."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _sec_to_clock(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class Turn(BaseModel):
    """One utterance / caption cue."""

    id: str
    start_s: float
    end_s: float
    speaker: Optional[str] = None
    text: str
    source: str = "srt"  # srt | whisper

    @property
    def clock(self) -> str:
        return _sec_to_clock(self.start_s)


class EventType(str, Enum):
    phase_transition = "phase_transition"
    medication_given = "medication_given"
    incision = "incision"
    implant_placed = "implant_placed"
    line_placed = "line_placed"
    drain_placed = "drain_placed"
    device_step = "device_step"
    conversion = "conversion"
    blood_loss = "blood_loss"
    hemodynamic_event = "hemodynamic_event"
    transfusion = "transfusion"
    count_status = "count_status"
    specimen = "specimen"
    complication = "complication"
    equipment_issue = "equipment_issue"
    personnel_change = "personnel_change"
    disposition = "disposition"


class ProceduralEvent(BaseModel):
    id: str
    t_start_s: float
    t_end_s: Optional[float] = None
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    extractor: str = "llm"  # llm | rule | manual

    @property
    def clock(self) -> str:
        return _sec_to_clock(self.t_start_s)


# Phase vocabulary is not fixed here — it belongs to the active SiteProfile
# (see pi/profile.py). `CaseState.phase` is just whatever string that profile uses.


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    route: Optional[str] = None
    t_s: float


class Transfusion(BaseModel):
    product: str  # PRBC | FFP | platelets | cryo
    units: Optional[float] = None
    t_s: float


class Complication(BaseModel):
    description: str
    t_s: float
    resolved: Optional[bool] = None


class CaseState(BaseModel):
    as_of_s: float
    phase: str = "arrival"
    meds: list[Medication] = Field(default_factory=list)
    ebl_ml: Optional[float] = None
    transfusions: list[Transfusion] = Field(default_factory=list)  # raw events (provenance)
    transfusion_totals: dict[str, float] = Field(default_factory=dict)  # product -> best total
    implants: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    drains: list[str] = Field(default_factory=list)
    converted: Optional[str] = None  # e.g. "laparoscopic -> open"
    counts: Optional[str] = None  # e.g. "correct x2" | "incorrect" | "pending"
    complications: list[Complication] = Field(default_factory=list)
    open_concerns: list[str] = Field(default_factory=list)
    disposition: Optional[str] = None
    disposition_path: list[str] = Field(default_factory=list)
    last_event_id: Optional[str] = None
    provenance: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def clock(self) -> str:
        return _sec_to_clock(self.as_of_s)


class CaseContext(BaseModel):
    """Set-up facts stated early in the case. No patient name — descriptor only."""

    patient_descriptor: Optional[str] = None  # "54-year-old man"
    planned_procedure: Optional[str] = None
    indication: Optional[str] = None  # preoperative diagnosis
    anesthesia_type: Optional[str] = None
    evidence_turn_ids: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    kind: str  # handoff | opnote | family
    text: str
    unsupported_claims: list[str] = Field(default_factory=list)
    revised: bool = False
    accepted: bool = False
