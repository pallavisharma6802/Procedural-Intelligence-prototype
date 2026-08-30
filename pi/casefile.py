"""The shared blackboard every agent reads and writes. Persists as JSON under runs/<case_id>/."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .schemas import CaseContext, CaseState, Draft, ProceduralEvent, Turn

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class LogEntry(BaseModel):
    t_wall: float
    agent: str
    message: str


class CaseFile(BaseModel):
    case_id: str
    source_path: Optional[str] = None
    profile_id: Optional[str] = None
    context: Optional[CaseContext] = None
    turns: list[Turn] = Field(default_factory=list)
    events: list[ProceduralEvent] = Field(default_factory=list)
    states: list[CaseState] = Field(default_factory=list)
    drafts: dict[str, Draft] = Field(default_factory=dict)
    run_log: list[LogEntry] = Field(default_factory=list)

    # ---- lifecycle -------------------------------------------------------
    @property
    def dir(self) -> Path:
        return RUNS_DIR / self.case_id

    @classmethod
    def load(cls, case_id: str) -> "CaseFile":
        p = RUNS_DIR / case_id / "casefile.json"
        if not p.exists():
            raise FileNotFoundError(f"no run for case_id={case_id!r} ({p})")
        return cls.model_validate_json(p.read_text())

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "casefile.json").write_text(self.model_dump_json(indent=2))
        # human-readable side artifacts
        for name, blob in {
            "turns.json": [t.model_dump() for t in self.turns],
            "events.json": [e.model_dump() for e in self.events],
            "state.jsonl": None,
        }.items():
            if blob is not None:
                (self.dir / name).write_text(json.dumps(blob, indent=2))
        (self.dir / "state.jsonl").write_text(
            "\n".join(s.model_dump_json() for s in self.states)
        )
        for kind, draft in self.drafts.items():
            (self.dir / f"{kind}.md").write_text(draft.text)

    def log(self, agent: str, message: str) -> None:
        self.run_log.append(LogEntry(t_wall=time.time(), agent=agent, message=message))

    # ---- convenience ---------------------------------------------------
    def turn_by_id(self, tid: str) -> Optional[Turn]:
        return next((t for t in self.turns if t.id == tid), None)

    def final_state(self) -> Optional[CaseState]:
        return self.states[-1] if self.states else None
