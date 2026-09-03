"""events -> CaseState snapshots. Deterministic fold. No LLM.

state_n = reduce(state_{n-1}, event_n), one snapshot appended per event, each field
carrying provenance back to the events that set it.

Phase handling is driven by the active SiteProfile: its `phases` list defines the order,
`phase_synonyms` normalises loose labels, `procedure_start_phase` is where an `incision`
event lands, and the last phase in the list is terminal (a `disposition` event jumps there).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..casefile import CaseFile
from ..profile import active_profile
from ..schemas import (
    CaseState,
    Complication,
    EventType,
    Medication,
    ProceduralEvent,
    Transfusion,
)
from .base import Agent


@dataclass
class _PhaseModel:
    order: list[str]
    synonyms: dict[str, str]
    start: str
    terminal: str

    def rank(self, phase: str) -> int:
        return self.order.index(phase) if phase in self.order else -1

    def norm(self, phase: str | None) -> str | None:
        if not phase:
            return None
        return self.synonyms.get(phase, self.synonyms.get(phase.lower(), phase))


class StateReducerAgent(Agent):
    name = "state"
    requires = ("events",)
    produces = "states"

    async def run(self, cf: CaseFile) -> CaseFile:
        prof = active_profile()
        pm = _PhaseModel(
            order=list(prof.phases),
            synonyms=dict(prof.phase_synonyms),
            start=prof.procedure_start_phase if prof.procedure_start_phase in prof.phases else prof.phases[0],
            terminal=prof.phases[-1],
        )
        state = CaseState(as_of_s=0.0, phase=pm.order[0])
        snapshots: list[CaseState] = [state]  # always at least the initial empty state
        for ev in sorted(cf.events, key=lambda e: e.t_start_s):
            state = _apply(state, ev, pm)
            snapshots.append(state)
        cf.states = snapshots
        cf.log(self.name, f"folded {len(cf.events)} events -> {len(snapshots)} snapshots (profile={prof.id})")
        return cf


def _num(v) -> float | None:
    """Best-effort number from an extractor payload value ('400', '~150 mL', 400).

    Qualitative words ('minimal', 'moderate', 'none') carry no number -> None.
    """
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
    return float(m.group()) if m else None


def _prov(state: CaseState, field: str, ev: ProceduralEvent) -> None:
    state.provenance.setdefault(field, [])
    if ev.id not in state.provenance[field]:
        state.provenance[field].append(ev.id)


def _advance(s: CaseState, target: str, pm: _PhaseModel, ev: ProceduralEvent) -> None:
    if pm.rank(target) >= 0 and pm.rank(target) >= pm.rank(s.phase):
        s.phase = target
        _prov(s, "phase", ev)


def _apply(prev: CaseState, ev: ProceduralEvent, pm: _PhaseModel) -> CaseState:
    s = prev.model_copy(deep=True)
    s.as_of_s = ev.t_start_s
    s.last_event_id = ev.id
    p = ev.payload

    if ev.type == EventType.phase_transition:
        _advance(s, pm.norm(p.get("phase")), pm, ev)
    elif ev.type == EventType.incision:
        _advance(s, pm.start, pm, ev)
    elif ev.type == EventType.medication_given:
        s.meds.append(Medication(name=p.get("name", "unknown"), dose=p.get("dose"), route=p.get("route"), t_s=ev.t_start_s))
        _prov(s, "meds", ev)
    elif ev.type == EventType.blood_loss and p.get("ebl_ml") is not None:
        ebl = _num(p["ebl_ml"])
        if ebl is not None:
            s.ebl_ml = ebl
            _prov(s, "ebl_ml", ev)
    elif ev.type == EventType.transfusion:
        prod = p.get("product", "PRBC")
        units = p.get("units")
        s.transfusions.append(Transfusion(product=prod, units=units, t_s=ev.t_start_s))
        if units is not None:
            try:
                u = float(units)
            except (TypeError, ValueError):
                u = None
            if u is not None:
                # Trauma transcripts state running totals ("6 so far", "8 total") that climb
                # monotonically; taking the max tracks the true total and ignores the smaller
                # discrete mentions that would otherwise be double-counted.
                s.transfusion_totals[prod] = max(s.transfusion_totals.get(prod, 0.0), u)
        _prov(s, "transfusions", ev)
    elif ev.type == EventType.implant_placed and p.get("device"):
        s.implants.append(p["device"])
        _prov(s, "implants", ev)
    elif ev.type == EventType.line_placed:
        s.lines.append(" ".join(x for x in (p.get("type"), p.get("site")) if x) or "line")
        _prov(s, "lines", ev)
    elif ev.type == EventType.drain_placed:
        s.drains.append(" ".join(x for x in (p.get("type"), p.get("site")) if x) or "drain")
        _prov(s, "drains", ev)
    elif ev.type == EventType.conversion:
        frm, to = p.get("from"), p.get("to")
        s.converted = f"{frm} -> {to}" if frm and to else (to or "converted")
        s.open_concerns.append(f"case converted {s.converted}")
        _prov(s, "converted", ev)
    elif ev.type == EventType.count_status and p.get("status"):
        s.counts = p["status"]
        if p.get("intentional"):
            detail = p.get("detail") or "items intentionally retained for damage control"
            s.counts = f"{p['status']} (intentional: {detail})"
            s.open_concerns.append(f"retained surgical items - intentional: {detail}")
        _prov(s, "counts", ev)
    elif ev.type == EventType.complication and p.get("description"):
        s.complications.append(Complication(description=p["description"], t_s=ev.t_start_s, resolved=p.get("resolved")))
        s.open_concerns.append(p["description"])
        _prov(s, "complications", ev)
    elif ev.type == EventType.hemodynamic_event and p.get("description"):
        s.open_concerns.append(p["description"])
        _prov(s, "open_concerns", ev)
    elif ev.type == EventType.disposition:
        dest = p.get("destination")
        if dest:
            label = dest + (f" ({p['detail']})" if p.get("detail") else "")
            if dest not in " ".join(s.disposition_path):
                s.disposition_path.append(dest)
            s.disposition = " then ".join(s.disposition_path) if len(s.disposition_path) > 1 else label
            _advance(s, pm.terminal, pm, ev)
            _prov(s, "disposition", ev)

    return s
