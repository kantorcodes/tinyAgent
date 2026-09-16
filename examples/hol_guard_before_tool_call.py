"""Use HOL Guard at TinyAgent's before_tool_call boundary."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Mapping

from tinyagent import (
    AgentTool,
    AgentToolResult,
    BeforeToolCallFn,
    TextContent,
    ToolCallContent,
    ToolLoopControl,
)
from tinyagent.agent_types import JsonObject

_GUARD_TIMEOUT_SECONDS = 10


def _guard_decision(command: str) -> tuple[bool, str]:
    """Return true only for HOL Guard's explicit benign allow result."""
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
        payload: object = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return False, "Guard evaluation failed"

    if not isinstance(payload, dict):
        return False, "Guard returned malformed output"
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        return False, "Guard returned malformed output"

    explicitly_benign = classification.get("explicitly_benign") is True
    minimum_action = payload.get("minimum_action")
    allowed = explicitly_benign and minimum_action == "allow"
    reason = (
        f"minimum_action={minimum_action or 'unknown'}, "
        f"explicitly_benign={explicitly_benign}"
    )
    return allowed, reason


def make_hol_guard_before_tool_call(command_fields: Mapping[str, str]) -> BeforeToolCallFn:
    """Build a fail-closed Guard hook for selected command-bearing tools."""

    async def before_tool_call(
        tool_call: ToolCallContent,
        tool: AgentTool | None,
        args: JsonObject,
    ) -> ToolLoopControl | None:
        del tool
        tool_name = tool_call.name
        if tool_name is None:
            return None
        command_field = command_fields.get(tool_name)
        if command_field is None:
            return None

        command = args.get(command_field)
        if not isinstance(command, str) or not command.strip():
            return _blocked("missing command input")

        allowed, reason = await asyncio.to_thread(_guard_decision, command)
        if allowed:
            return None
        return _blocked(reason)

    return before_tool_call


def _blocked(reason: str) -> ToolLoopControl:
    return ToolLoopControl(
        result=AgentToolResult(
            content=[TextContent(text=f"Blocked by HOL Guard: {reason}")],
            details={"policy": "hol-guard"},
        ),
        is_error=True,
    )
