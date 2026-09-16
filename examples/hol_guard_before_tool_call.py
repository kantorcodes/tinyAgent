"""Use HOL Guard at TinyAgent's before_tool_call boundary."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from typing import Any

from tinyagent import AgentTool, AgentToolResult, TextContent, ToolCallContent, ToolLoopControl
from tinyagent.agent_types import JsonObject

_GUARD_TIMEOUT_SECONDS = 10
_ALLOWED = {"allow", "benign"}


def _guard_decision(command: str) -> tuple[bool, str]:
    executable = shutil.which("hol-guard")
    if not executable:
        return False, "hol-guard is not installed"

    try:
        completed = subprocess.run(
            [executable, "command", "test", command, "--json"],
            capture_output=True,
            text=True,
            timeout=_GUARD_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            return False, "Guard evaluation failed"
        payload: Any = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return False, "Guard evaluation failed"

    if not isinstance(payload, dict):
        return False, "Guard returned malformed output"
    decision = str(payload.get("decision", "")).lower()
    return decision in _ALLOWED, f"decision was {decision or 'unknown'}"


async def hol_guard_before_tool_call(
    tool_call: ToolCallContent,
    tool: AgentTool | None,
    args: JsonObject,
) -> ToolLoopControl | None:
    """Block command-bearing calls unless HOL Guard explicitly allows them."""
    del tool_call, tool
    if "command" not in args:
        return None

    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        allowed, reason = False, "missing command"
    else:
        allowed, reason = await asyncio.to_thread(_guard_decision, command)
    if allowed:
        return None

    message = f"Blocked by HOL Guard: {reason}"
    return ToolLoopControl(
        result=AgentToolResult(
            content=[TextContent(text=message)],
            details={"policy": "hol-guard"},
        ),
        is_error=True,
    )
