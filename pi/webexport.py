"""Turn a CaseFile into the JSON the web UI consumes.

Adds two things the raw casefile doesn't have:
  - `profile`: the resolved site profile (phases drive the timeline bands)
  - `links`:   per-draft spans that trace a sentence back to state fields -> events -> turns
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .casefile import CaseFile
from .profile import SiteProfile
from .stt import is_audio

# state field -> (how to recognise it in draft text, human label)
_FACT_FIELDS = ["ebl_ml", "meds", "implants", "drains", "lines", "transfusion_totals",
                "counts", "converted", "disposition", "complications", "phase"]


def _phrase_span(text: str, at: int, needle_len: int) -> tuple[int, int]:
    """Expand [at, at+needle_len) to nearby word boundaries for a readable highlight."""
    s = at
    while s > 0 and text[s - 1] not in " \n\t":
        s -= 1
    while s < at and text[s] in "([\"'*":
        s += 1
    e = at + needle_len
    _UNITS = {"ml", "l", "iv", "im", "g", "mg", "mcg", "cc", "units", "unit", "french", "fr",
              "correct", "incorrect", "mmhg", "bpm"}
    while e < len(text) and text[e] == " ":
        nxt = text[e + 1:].split(" ", 1)[0].strip(".,;:)")
        if nxt.lower() in _UNITS:
            e += 1 + len(text[e + 1:].split(" ", 1)[0])
        else:
            break
    while e > s and text[e - 1] in " .,;:)":
        e -= 1
    return s, e


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
    _STOP = {"requiring", "resolved", "management", "managed", "during", "initial", "selecting",
             "correct", "unable", "obtain", "attempt", "without", "further", "possible"}
    for c in state.get("complications", []):
        words = [w for w in re.findall(r"[a-z]{6,}", c.get("description", "").lower()) if w not in _STOP][:1]
        if words:
            eids = _find_events(events, "complication", words[0])
            n.append(("complications", words, f"complication: {c['description'][:60]}", eids))
    return n


def _draft_links(text: str, state: dict, events: list, events_by_id: dict) -> list[dict[str, Any]]:
    prov = state.get("provenance", {})
    low = text.lower()
    raw: list[dict[str, Any]] = []
    for field, needles, label, spec in _needles(state, events):
        eids = spec or prov.get(field, [])
        tids = sorted({t for e in eids for t in events_by_id.get(e, {}).get("evidence_turn_ids", [])})
        if not (eids or tids):
            continue
        for nd in needles:
            nd = (nd or "").strip()
            if len(nd) < 3:
                continue
            i = low.find(nd)
            while i != -1:
                if i < 3 or not low[i - 1].isalnum():  # word-ish start
                    s, e = _phrase_span(text, i, len(nd))
                    raw.append({"start": s, "end": e, "field": field, "label": label,
                                "event_ids": eids, "turn_ids": tids})
                i = low.find(nd, i + len(nd))
    # de-overlap: keep the first, drop anything that intersects it
    raw.sort(key=lambda l: (l["start"], -(l["end"] - l["start"])))
    out: list[dict[str, Any]] = []
    for lk in raw:
        if out and lk["start"] < out[-1]["end"]:
            continue
        out.append(lk)
    return out


def export_case(cf: CaseFile) -> dict[str, Any]:
    prof = SiteProfile.load(cf.profile_id or "default_or")
    data = cf.model_dump(mode="json")
    events_by_id = {e["id"]: e for e in data["events"]}
    final = data["states"][-1] if data["states"] else {}

    links = {}
    for kind, draft in data.get("drafts", {}).items():
        if draft.get("text") and not draft["text"].startswith("["):
            links[kind] = _draft_links(draft["text"], final, data["events"], events_by_id)

    src = cf.source_path or ""
    has_audio = bool(src and is_audio(src) and Path(src).exists())
    return {
        "case_id": cf.case_id,
        "source": src.split("/")[-1],
        "has_audio": has_audio,
        "audio_bytes": Path(src).stat().st_size if has_audio else 0,
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
