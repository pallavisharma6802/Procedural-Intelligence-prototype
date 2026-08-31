"""Back-compat shim. The pipeline is now a LangGraph graph - see pi/graph.py."""

from .graph import mermaid, run_pipeline, run_stage

__all__ = ["run_pipeline", "run_stage", "mermaid"]
