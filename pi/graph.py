"""LangGraph orchestration of the agent pipeline.

    ingest → roles → context → extract → reduce → handoff → opnote → family → critic_check
                                                                                │
                                                      ┌── flagged & round 0 ────┤
                                                      ▼                         │
                                                critic_revise ──────────────────┘
                                                      │ (else)
                                                      ▼
                                                critic_finalize → END

The pipeline is mostly linear (one writer per super-step lets us pass the CaseFile
through a single channel). The one real branch is the critic's check→revise→re-check
loop, expressed as a conditional edge.
"""

from __future__ import annotations

import warnings
from typing import Optional

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.graph import END, START, StateGraph

from typing_extensions import TypedDict

from .agents import (
    ContextAgent,
    CriticAgent,
    EventAgent,
    FamilyAgent,
    HandoffAgent,
    OpNoteAgent,
    RolesAgent,
    StateReducerAgent,
    TranscriptAgent,
)
from .casefile import CaseFile
from .profile import active_profile

MAX_REVISE_ROUNDS = 1


class PIState(TypedDict, total=False):
    cf: CaseFile
    critic_source: Optional[str]
    critic_results: Optional[dict]
    revise_rounds: int


#: node name -> the `--upto` label that should stop after it
STOP_LABELS = {
    "ingest": "transcript",
    "extract": "understand",
    "reduce": "state",
    "family": "projections",
    "critic_finalize": "critic",
}
ORDER = ["transcript", "understand", "state", "projections", "critic"]


def _agents():
    projections = {"handoff": HandoffAgent(), "opnote": OpNoteAgent(), "family": FamilyAgent()}
    critic = CriticAgent()
    critic.generators = projections
    return {
        "transcript": TranscriptAgent(),
        "roles": RolesAgent(),
        "context": ContextAgent(),
        "events": EventAgent(),
        "state": StateReducerAgent(),
        "critic": critic,
        **projections,
    }


def build_graph():
    a = _agents()

    def _node(agent_key: str):
        agent = a[agent_key]

        async def node(state: PIState) -> PIState:
            cf = state["cf"]
            if agent_key == "transcript":
                cf.profile_id = active_profile().id
            await agent.run(cf)
            cf.save()
            return {"cf": cf}

        node.__name__ = f"node_{agent_key}"
        return node

    g = StateGraph(PIState)

    g.add_node("ingest", _node("transcript"))
    g.add_node("roles", _node("roles"))
    g.add_node("context", _node("context"))
    g.add_node("extract", _node("events"))
    g.add_node("reduce", _node("state"))
    g.add_node("handoff", _node("handoff"))
    g.add_node("opnote", _node("opnote"))
    g.add_node("family", _node("family"))

    async def critic_check(state: PIState) -> PIState:
        cf = state["cf"]
        src = state.get("critic_source") or a["critic"].build_source(cf)
        results = await a["critic"].check(cf, src)
        cf.save()
        return {"cf": cf, "critic_source": src, "critic_results": results}

    async def critic_revise(state: PIState) -> PIState:
        cf = state["cf"]
        flagged = {k: v for k, v in (state.get("critic_results") or {}).items() if v}
        await a["critic"].revise(cf, state["critic_source"], flagged)
        cf.save()
        return {"cf": cf, "revise_rounds": state.get("revise_rounds", 0) + 1}

    async def critic_finalize(state: PIState) -> PIState:
        cf = state["cf"]
        a["critic"].finalize(cf, state.get("critic_results"))
        cf.save()
        return {"cf": cf}

    g.add_node("critic_check", critic_check)
    g.add_node("critic_revise", critic_revise)
    g.add_node("critic_finalize", critic_finalize)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "roles")
    g.add_edge("roles", "context")
    g.add_edge("context", "extract")
    g.add_edge("extract", "reduce")
    g.add_edge("reduce", "handoff")
    g.add_edge("handoff", "opnote")
    g.add_edge("opnote", "family")
    g.add_edge("family", "critic_check")

    def route_after_check(state: PIState) -> str:
        results = state.get("critic_results") or {}
        has_flags = any(results.values()) if isinstance(results, dict) else False
        if has_flags and state.get("revise_rounds", 0) < MAX_REVISE_ROUNDS:
            return "critic_revise"
        return "critic_finalize"

    g.add_conditional_edges("critic_check", route_after_check, ["critic_revise", "critic_finalize"])
    g.add_edge("critic_revise", "critic_check")
    g.add_edge("critic_finalize", END)

    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def run_pipeline(cf: CaseFile, *, upto: str = "critic", verbose: bool = True) -> CaseFile:
    """Stream the graph, stopping once the node mapped to `upto` has run."""
    if upto not in ORDER:
        raise ValueError(f"upto must be one of {ORDER}")
    stop_nodes = {n for n, label in STOP_LABELS.items() if ORDER.index(label) >= ORDER.index(upto)}
    state: PIState = {"cf": cf, "revise_rounds": 0, "critic_source": None, "critic_results": None}
    async for update in _graph().astream(state, stream_mode="updates"):
        for node, payload in update.items():
            if payload and "cf" in payload:
                cf = payload["cf"]
            if verbose:
                print(f"  ✓ {node}")
            if node in stop_nodes and STOP_LABELS[node] == upto:
                return cf
    return cf


async def run_stage(cf: CaseFile, stage: str) -> CaseFile:
    """Re-run one agent outside the graph (for `pi stage <name>`)."""
    a = _agents()
    alias = {"understand": ("context", "events"), "projections": ("handoff", "opnote", "family")}
    names = alias.get(stage, (stage,))
    for n in names:
        await a[n].run(cf)
    cf.save()
    return cf


def mermaid() -> str:
    return _graph().get_graph().draw_mermaid()
