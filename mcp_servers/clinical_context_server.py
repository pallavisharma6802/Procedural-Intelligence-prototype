#!/usr/bin/env python3
"""Reference clinical-context MCP server (stdio, JSON-RPC 2.0).

Serves de-identified sample records from sample_records.json so the pipeline's context
step can be exercised end to end. A real deployment points at a hospital EHR / OR-board
MCP server instead; the tool names and shapes are the contract.

Tools:
  lookup_patient(query)              -> best-matching record header
  get_scheduled_procedure(patient_id)
  get_active_medications(patient_id)
  get_allergies(patient_id)
  get_problem_list(patient_id)
"""

import json
import sys
from pathlib import Path

RECORDS = json.loads((Path(__file__).parent / "sample_records.json").read_text())

TOOLS = [
    {
        "name": "lookup_patient",
        "description": "Find a patient record by a free-text hint (age, procedure, condition).",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_scheduled_procedure",
        "description": "Booked procedure, surgeon, and anaesthesia plan for a patient.",
        "inputSchema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_active_medications",
        "description": "Home medication list for a patient.",
        "inputSchema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_allergies",
        "description": "Recorded allergies for a patient.",
        "inputSchema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    },
    {
        "name": "get_problem_list",
        "description": "Active problem list for a patient.",
        "inputSchema": {
            "type": "object",
            "properties": {"patient_id": {"type": "string"}},
            "required": ["patient_id"],
        },
    },
]


def _record(patient_id):
    return next((r for r in RECORDS if r["patient_id"] == patient_id), None)


_STOP = set("the a an and or of for to in on with is are was were be patient dr this that "
            "please can you we have has will now here there ok year old man woman going "
            "coming possible into out".split())


def _terms(text):
    return {w for w in "".join(c if c.isalnum() or c == " " else " " for c in text.lower()).split()
            if len(w) > 2 and w not in _STOP}


def _lookup(query):
    q = _terms(query)
    scored = []
    for r in RECORDS:
        blob = _terms(" ".join([
            r["descriptor"], r["indication"], r["scheduled_procedure"]["procedure"],
            " ".join(r["problem_list"]), " ".join(r.get("aliases", [])),
        ]))
        scored.append((len(q & blob), r))
    scored.sort(key=lambda x: -x[0])
    best_hits, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if best_hits < 2 or best_hits == runner_up:
        return {"match": None}
    return {
        "match": {
            "patient_id": best["patient_id"],
            "descriptor": best["descriptor"],
            "indication": best["indication"],
        },
        "confidence": round(min(1.0, best_hits / 5), 2),
    }


def call_tool(name, args):
    if name == "lookup_patient":
        return _lookup(args.get("query", ""))
    rec = _record(args.get("patient_id", ""))
    if rec is None:
        return {"error": "unknown patient_id"}
    if name == "get_scheduled_procedure":
        return rec["scheduled_procedure"]
    if name == "get_active_medications":
        return {"home_medications": rec["home_medications"]}
    if name == "get_allergies":
        return {"allergies": rec["allergies"]}
    if name == "get_problem_list":
        return {"problem_list": rec["problem_list"]}
    return {"error": f"unknown tool {name}"}


def handle(msg):
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": mid,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "clinical-context", "version": "0.1.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        out = call_tool(params.get("name"), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0", "id": mid,
            "result": {"content": [{"type": "text", "text": json.dumps(out)}]},
        }
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
