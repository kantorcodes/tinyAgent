"""Use HOL Guard's command classifier at TinyAgent's before_tool_call boundary.

This explicitly benign command filter runs before mapped tools execute. The
command-test CLI does not evaluate final runtime policy or issue approvals.
"""

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

def _reject_duplicate_guard_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Ambiguous Guard JSON must never let a later allow overwrite a stop."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in Guard JSON")
        result[key] = value
    return result


def _guard_decision(command: str) -> tuple[bool, str]:
    """Return true only for HOL Guard's explicit benign allow result."""
    executable = shutil.which("hol-guard")
    if not executable:
        return False, "hol-guard is not installed"

    try:
        completed = subprocess.run(
            [executable, "command", "test", "--json", "--", command],
            capture_output=True,
            text=True,
            timeout=_GUARD_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            return False, "Guard evaluation failed"
        payload: object = json.loads(completed.stdout, object_pairs_hook=_reject_duplicate_guard_keys)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, RecursionError):
        return False, "Guard evaluation failed"

    if not isinstance(payload, dict):
        return False, "Guard returned malformed output"
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        return False, "Guard returned malformed output"

    # An explicit Guard stop cannot be overridden by an allow-shaped classifier.
    decision = payload.get("decision")
    # A malformed continuation field cannot authorize a protected tool.
    if ("continue" in payload and not isinstance(payload["continue"], bool)):
        return False, "Guard returned malformed continuation"
    if payload.get("continue") is False or (
        "decision" in payload
        and (not isinstance(decision, str) or decision.strip().lower() != "allow")
    ):
        return False, "Guard explicitly stopped tool continuation"

    # Reject contradictory hook-level results if an extended Guard response
    # includes them. The command-test classifier normally has no hook envelope.
    hook_output = payload.get("hookSpecificOutput")
    if "hookSpecificOutput" in payload:
        if not isinstance(hook_output, dict) or (
            hook_output.get("permissionDecision") != "allow"
            or hook_output.get("hookEventName") != "PreToolUse"
        ):
            return False, "Guard hook response contradicts a benign classification"

    # A concrete stop/review action cannot be weakened by the command classifier.
    # Command inspection may report "not_run" when no runtime policy was evaluated.
    policy_action = payload.get("policy_action")
    if "policy_action" in payload and (
        not isinstance(policy_action, str)
        or policy_action.strip().lower() not in {"allow", "not_run"}
    ):
        return False, "Guard explicitly reported a non-allow policy action"

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
        """Block mapped tool calls unless HOL Guard explicitly allows the command."""
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
    """Return TinyAgent's pre-tool control result for a Guard denial."""
    return ToolLoopControl(
        result=AgentToolResult(
            content=[TextContent(text=f"Blocked by HOL Guard: {reason}")],
            details={"policy": "hol-guard"},
        ),
        is_error=True,
    )
