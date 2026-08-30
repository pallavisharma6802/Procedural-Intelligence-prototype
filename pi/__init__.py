"""Procedural Intelligence: reconstruct a procedural state timeline from OR audio/transcript."""

import os
import warnings

# Third-party import-time noise (langgraph/langchain pending-deprecations, urllib3/LibreSSL).
# This is a CLI; keep stderr clean. Set PI_WARNINGS=1 to see them.
if not os.environ.get("PI_WARNINGS"):
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

__version__ = "0.1.0"
