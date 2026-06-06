"""Plan + scratchpad: the agent''s working memory.

Two slots, both priority 2, both writable. The plan is the public, committed
trajectory; the scratchpad is private scratch space. The agent emits them as
well-known tags inside its reply; the runner strips them and stores the rest
back into the next-turn slots.

Wire format (the agent''s reply must contain):

    <<<PLAN>>>
    1) ...
    2) ...
    <<<END>>>

    <<<SCRATCHPAD>>>
    ...private notes...
    <<<END>>>

    <the actual user-facing response>

If a tag is missing, the slot is left unchanged (it persists from last turn).
"""
from __future__ import annotations
import re
from dataclasses import dataclass


_PLAN = re.compile(r"<<<PLAN>>>\n(.*?)\n<<<END>>>", re.DOTALL)
_SCRATCH = re.compile(r"<<<SCRATCHPAD>>>\n(.*?)\n<<<END>>>", re.DOTALL)


@dataclass
class AgentReply:
    plan: str | None
    scratchpad: str | None
    body: str

    @classmethod
    def parse(cls, text: str) -> "AgentReply":
        plan_m = _PLAN.search(text)
        scratch_m = _SCRATCH.search(text)
        body = _PLAN.sub("", text)
        body = _SCRATCH.sub("", body).strip()
        return cls(
            plan=plan_m.group(1).strip() if plan_m else None,
            scratchpad=scratch_m.group(1).strip() if scratch_m else None,
            body=body,
        )
