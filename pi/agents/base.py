"""Uniform agent interface. Deterministic and LLM agents share it."""

from __future__ import annotations

import abc

from ..casefile import CaseFile


class Agent(abc.ABC):
    name: str = "agent"
    #: casefile attributes this agent needs populated before it can run
    requires: tuple[str, ...] = ()
    #: casefile attribute this agent produces
    produces: str = ""

    def ready(self, cf: CaseFile) -> bool:
        return all(getattr(cf, r) for r in self.requires)

    @abc.abstractmethod
    async def run(self, cf: CaseFile) -> CaseFile:
        ...
