"""Turn a CaseFile into the JSON the web UI consumes.

Adds two things the raw casefile doesn't have:
  - `profile`: the resolved site profile (phases drive the timeline bands)
  - `links`:   per-draft spans that trace a sentence back to state fields -> events -> turns
"""

from __future__ import annotations

import re
from typing import Any

from .casefile import CaseFile
from .profile import SiteProfile

# state field -> (how to recognise it in draft text, human label)
_FACT_FIELDS = ["ebl_ml", "meds", "implants", "drains", "lines", "transfusion_totals",
                "counts", "converted", "disposition", "complications", "phase"]


def _sentences(text: str) -> list[tuple[int, int, str]]:
    out, i = [], 0
    for chunk in re.split(r"(?<=[.!?])\s+|\n+", text):
        chunk = chunk.strip()
        if not chunk:
            i += 1
            continue
        start = text.find(chunk, i)
        if start < 0:
            start = i
        out.append((start, start + len(chunk), chunk))
        i = start + len(chunk)
    return out


def _find_events(events: list, etype: str, match: str) -> list[str]:
    """event ids of type `etype` whose payload values contain `match` (lowercased)."""
    out = []
    for e in events:
        if e["type"] != etype:
            continue
        blob = " ".join(str(v) for v in (e.get("payload") or {}).values()).lower()
        if match in blob:
            out.append(e["id"])
    return out


def _needles(state: dict, events: list) -> list[tuple[str, list[str], str, list[str]]]:
    """(field, [lowercased strings to look for], label, specific_event_ids)."""
    n: list = []
    if state.get("ebl_ml") is not None:
        v = state["ebl_ml"]
        n.append(("ebl_ml", [f"{int(v)}", f"{v:g}"], "estimated blood loss",
                  _find_events(events, "blood_loss", str(int(v)))))
    for m in state.get("meds", []):
        if m.get("name"):
            nm = m["name"].lower()
            n.append(("meds", [nm], f"medication: {m['name']}", _find_events(events, "medication_given", nm)))
    for d in state.get("implants", []):
        key = str(d).lower().split("(")[0].strip()
        n.append(("implants", [str(d).lower(), key], f"implant: {d}",
                  _find_events(events, "implant_placed", key.split()[0] if key else str(d).lower())))
    for d in state.get("drains", []):
        w = str(d).lower().split()[0]
        n.append(("drains", [w], f"drain: {d}", _find_events(events, "drain_placed", w)))
    for prod in (state.get("transfusion_totals") or {}):
        n.append(("transfusion_totals", [prod.lower()], f"transfusion: {prod}",
                  _find_events(events, "transfusion", prod.lower())))
    if state.get("counts"):
        n.append(("counts", ["count"], f"counts: {state['counts']}",
                  [e["id"] for e in events if e["type"] == "count_status"]))
    if state.get("converted"):
        n.append(("converted", ["convert", "converted"], f"conversion: {state['converted']}",
                  [e["id"] for e in events if e["type"] == "conversion"]))
    if state.get("disposition"):
        dest = str(state["disposition"]).split()[0].lower()
        n.append(("disposition", [dest], f"disposition: {state['disposition']}",
                  [e["id"] for e in events if e["type"] == "disposition"]))
    for c in state.get("complications", []):
        words = re.findall(r"[a-z]{6,}", c.get("description", "").lower())[:2]
        if words:
            eids = _find_events(events, "complication", words[0])
            n.append(("complications", words, f"complication: {c['description'][:60]}", eids))
    return n


def _draft_links(text: str, state: dict, events: list, events_by_id: dict) -> list[dict[str, Any]]:
    prov = state.get("provenance", {})
    links: list[dict[str, Any]] = []
    for s0, s1, sent in _sentences(text):
        sl = sent.lower()
        for field, needles, label, spec in _needles(state, events):
            if any(nd and nd in sl for nd in needles):
                eids = spec or prov.get(field, [])
                tids = sorted({t for e in eids for t in events_by_id.get(e, {}).get("evidence_turn_ids", [])})
                if eids or tids:
                    links.append({"start": s0, "end": s1, "field": field, "label": label,
                                  "event_ids": eids, "turn_ids": tids})
                break
    return links


def export_case(cf: CaseFile) -> dict[str, Any]:
    prof = SiteProfile.load(cf.profile_id or "default_or")
    data = cf.model_dump(mode="json")
    events_by_id = {e["id"]: e for e in data["events"]}
    final = data["states"][-1] if data["states"] else {}

    links = {}
    for kind, draft in data.get("drafts", {}).items():
        if draft.get("text") and not draft["text"].startswith("["):
            links[kind] = _draft_links(draft["text"], final, data["events"], events_by_id)

    return {
        "case_id": cf.case_id,
        "source": (cf.source_path or "").split("/")[-1],
        "profile": {
            "id": prof.id, "label": prof.label, "care_setting": prof.care_setting,
            "phases": prof.phases, "handoff_name": prof.handoff.name,
        },
        "context": data.get("context"),
        "turns": data["turns"],
        "events": data["events"],
        "states": data["states"],
        "drafts": data.get("drafts", {}),
        "links": links,
        "run_log": data.get("run_log", []),
    }
