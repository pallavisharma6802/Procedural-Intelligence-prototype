"""FastAPI backend for the web UI. `pi serve` runs it.

  GET  /                       -> the single-page app
  GET  /api/cases              -> list runs
  GET  /api/cases/{id}         -> exported case JSON (webexport)
  POST /api/cases              -> upload a transcript/recording, run the pipeline
  GET  /api/cases/{id}/status  -> {stage, done, error} while a run is in flight
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .casefile import RUNS_DIR, CaseFile
from .graph import run_pipeline
from .profile import SiteProfile, available
from .webexport import export_case

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
UPLOAD_DIR = RUNS_DIR / "_uploads"

app = FastAPI(title="Procedural Intelligence")


def _status_path(case_id: str) -> Path:
    return RUNS_DIR / case_id / "_status.json"


def _write_status(case_id: str, **kw) -> None:
    p = _status_path(case_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev.update(kw)
    p.write_text(json.dumps(prev))


def _run_in_thread(case_id: str, source: Path, profile: str) -> None:
    import os

    os.environ["PI_PROFILE"] = profile

    async def _go():
        cf = CaseFile(case_id=case_id, source_path=str(source))
        stages = ["ingest", "roles", "context", "extract", "reduce",
                  "handoff", "opnote", "family", "critic_check", "critic_finalize"]
        _write_status(case_id, stages=stages, done_stages=[], done=False, error=None)
        # run_pipeline streams node names to stdout; re-derive progress from the run_log
        try:
            await run_pipeline(cf, verbose=False)
            _write_status(case_id, done=True, done_stages=stages)
        except Exception as exc:  # noqa: BLE001
            _write_status(case_id, done=True, error=str(exc))

    threading.Thread(target=lambda: asyncio.run(_go()), daemon=True).start()


# cases shown first, in this order, when present
_FEATURED = ["case01_lapchole", "case04_cath_pci", "case01_uk", "case03_trauma_exlap",
             "mmor_007_TKA", "mmor_007_audio", "case02_tka_uneventful"]


@app.get("/api/cases")
def list_cases():
    out = []
    for d in sorted(RUNS_DIR.glob("*/casefile.json")):
        if d.parent.name.startswith("_"):
            continue
        try:
            cf = json.loads(d.read_text())
        except Exception:  # noqa: BLE001
            continue
        cid = cf.get("case_id", d.parent.name)
        n_ev = len(cf.get("events", []))
        if cid not in _FEATURED and (n_ev < 3 or cid in {"or_clip"}):
            continue
        out.append({
            "case_id": cid,
            "profile": cf.get("profile_id"),
            "source": (cf.get("source_path") or "").split("/")[-1],
            "n_events": n_ev,
            "n_turns": len(cf.get("turns", [])),
            "featured": cid in _FEATURED,
        })
    rank = {c: i for i, c in enumerate(_FEATURED)}
    return sorted(out, key=lambda x: (rank.get(x["case_id"], 99), -x["n_events"]))


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    try:
        return JSONResponse(export_case(CaseFile.load(case_id)))
    except FileNotFoundError:
        raise HTTPException(404, f"no run {case_id!r}")


@app.get("/api/cases/{case_id}/status")
def get_status(case_id: str):
    p = _status_path(case_id)
    base = json.loads(p.read_text()) if p.exists() else {"done": False, "error": None}
    log_p = RUNS_DIR / case_id / "casefile.json"
    if log_p.exists():
        try:
            agents = {e["agent"] for e in json.loads(log_p.read_text()).get("run_log", [])}
            base["logged_agents"] = sorted(agents)
        except Exception:  # noqa: BLE001
            pass
    return base


@app.post("/api/cases")
async def create_case(file: UploadFile, profile: str = Form("default_or"), case_id: str = Form(None)):
    if profile not in available():
        raise HTTPException(400, f"unknown profile; have {available()}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cid = case_id or Path(file.filename or "case").stem
    cid = "".join(c if c.isalnum() or c in "-_" else "_" for c in cid)[:60] or "case"
    dest = UPLOAD_DIR / f"{cid}{Path(file.filename or '').suffix or '.srt'}"
    dest.write_bytes(await file.read())
    _run_in_thread(cid, dest, profile)
    return {"case_id": cid}


@app.get("/api/profiles")
def profiles():
    return [SiteProfile.load(n).model_dump() for n in available()]


@app.get("/", response_class=HTMLResponse)
def index():
    html = (WEB_DIR / "index.html").read_text()
    return html.replace("__BUNDLE__", "{}")  # live mode pulls cases from the API


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"Procedural Intelligence — http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
