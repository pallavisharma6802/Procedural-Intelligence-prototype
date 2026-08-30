from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

_envfile = Path(__file__).resolve().parent.parent / ".env"
if _envfile.exists():
    for _line in _envfile.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import typer
from rich import print as rprint
from rich.table import Table

from .casefile import CaseFile
from .graph import mermaid, run_pipeline, run_stage
from .llm import info as llm_info
from .profile import SiteProfile, available

app = typer.Typer(add_completion=False, help="Procedural Intelligence pipeline")


@app.command()
def run(
    source: Path = typer.Argument(..., help="transcript (.srt/.vtt) OR audio/video file"),
    case_id: str = typer.Option(None, "--case-id", "-c"),
    profile: str = typer.Option(None, "--profile", "-p", help="site profile name or .json path"),
    upto: str = typer.Option("critic", help="transcript|understand|state|projections|critic"),
):
    """Run the full LangGraph pipeline on a transcript or a recording."""
    if profile:
        os.environ["PI_PROFILE"] = profile
    prof = SiteProfile.load(os.environ.get("PI_PROFILE"))
    cid = case_id or source.stem
    cf = CaseFile(case_id=cid, source_path=str(source.resolve()))
    rprint(f"[bold]case:[/bold] {cid}   [bold]llm:[/bold] {llm_info()}   [bold]profile:[/bold] {prof.id} ({prof.label})")
    cf = asyncio.run(run_pipeline(cf, upto=upto))
    rprint(f"\n[green]done[/green] -> {cf.dir}")


@app.command()
def graph():
    """Print the pipeline graph as mermaid."""
    print(mermaid())


@app.command()
def profiles():
    """List available site profiles."""
    for name in available():
        p = SiteProfile.load(name)
        rprint(f"  [bold]{p.id}[/bold]  — {p.label}  [dim]({p.care_setting}, {p.handoff.name} handoff)[/dim]")


@app.command()
def stage(name: str, case_id: str, profile: str = typer.Option(None, "--profile", "-p")):
    """Re-run a single agent (or 'understand'/'projections') on an existing run."""
    cf = CaseFile.load(case_id)
    os.environ["PI_PROFILE"] = profile or cf.profile_id or os.environ.get("PI_PROFILE", "default_or")
    asyncio.run(run_stage(cf, name))
    rprint(f"[green]{name} re-run[/green] (profile={os.environ['PI_PROFILE']}) -> {cf.dir}")


@app.command()
def show(case_id: str, what: str = typer.Argument("state")):
    """Inspect an artifact: turns|events|state|handoff|opnote|family|log"""
    cf = CaseFile.load(case_id)
    if what == "context":
        c = cf.context.model_dump(exclude_none=True, exclude={"evidence_turn_ids"}) if cf.context else {}
        rprint(c or "[dim]no case context extracted (transcript never states the setup)[/dim]")
    elif what == "turns":
        for t in cf.turns:
            rprint(f"[dim]{t.clock}[/dim] {('['+t.speaker+'] ') if t.speaker else ''}{t.text}")
    elif what == "events":
        tbl = Table("time", "type", "payload", "conf", "evidence")
        for e in cf.events:
            tbl.add_row(e.clock, e.type.value, json.dumps(e.payload), f"{e.confidence:.2f}",
                        ",".join(e.evidence_turn_ids))
        rprint(tbl)
    elif what == "state":
        s = cf.final_state()
        if not s:
            raise typer.Exit("no state yet")
        rprint(f"[dim]profile: {cf.profile_id or 'default_or'}[/dim]")
        rprint(json.loads(s.model_dump_json()))
    elif what == "provenance":
        s = cf.final_state()
        ev_by_id = {e.id: e for e in cf.events}
        tbi = {t.id: t for t in cf.turns}
        for field, ev_ids in (s.provenance if s else {}).items():
            rprint(f"[bold cyan]{field}[/bold cyan]")
            for eid in ev_ids:
                ev = ev_by_id.get(eid)
                if not ev:
                    continue
                quotes = " | ".join(tbi[i].text for i in ev.evidence_turn_ids if i in tbi)
                rprint(f"  [dim]{ev.clock}[/dim] {ev.type.value} {json.dumps(ev.payload)}")
                rprint(f"    [dim]“{quotes}”[/dim]")
    elif what in ("handoff", "opnote", "family"):
        d = cf.drafts.get(what)
        if not d:
            raise typer.Exit(f"no {what} draft")
        rprint(f"[bold]{what}[/bold]  accepted={d.accepted} revised={d.revised}")
        if d.unsupported_claims:
            rprint("[red]flagged:[/red]")
            for c in d.unsupported_claims:
                print(f"  - {c}")
        print("\n" + d.text)
    elif what == "log":
        for e in cf.run_log:
            rprint(f"[dim]{e.agent:16s}[/dim] {e.message}")
    else:
        raise typer.Exit(f"unknown artifact {what!r}")


@app.command()
def evaluate(case_id: str):
    """Compare drafts/state against data/synthetic/<case_id>_ground_truth.json if present."""
    cf = CaseFile.load(case_id)
    gt_path = Path(__file__).resolve().parent.parent / "data" / "synthetic" / f"{case_id}_ground_truth.json"
    if not gt_path.exists():
        raise typer.Exit(f"no ground truth at {gt_path}")
    gt = json.loads(gt_path.read_text())
    s = cf.final_state()
    got_types = sorted({e.type.value for e in cf.events})
    want_types = sorted(gt.get("expected_event_types", []))
    rprint("[bold]event type coverage[/bold]")
    rprint(f"  expected: {want_types}")
    rprint(f"  got:      {got_types}")
    rprint(f"  missing:  {sorted(set(want_types) - set(got_types))}")
    rprint("[bold]state checks[/bold]")
    ok = 0
    checks = gt.get("expected_state", {})
    for k, v in checks.items():
        got = getattr(s, k, None)
        hit = (str(v).lower() in str(got).lower()) if isinstance(v, str) else (
            got is not None and abs(float(got) - float(v)) < 1e-6
        )
        ok += hit
        rprint(f"  [{'green' if hit else 'red'}]{'✓' if hit else '✗'}[/] state.{k}: want={v!r}  got={got!r}")
    covered = len(set(want_types) & set(got_types))
    rprint(
        f"\n[bold]score[/bold]  event-types {covered}/{len(want_types)}   "
        f"state {ok}/{len(checks)}"
    )


if __name__ == "__main__":
    app()
