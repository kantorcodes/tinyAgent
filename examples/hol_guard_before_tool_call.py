"""Use HOL Guard's command classifier at TinyAgent's before_tool_call boundary.

This explicitly benign command filter runs before mapped tools execute. The
command-test CLI does not evaluate final runtime policy or issue approvals.
Protected local tools with a `cwd` argument are inspected from that exact
absolute, existing directory; remote execution roots are not supported.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
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


def _guard_decision(command: str, cwd: str | None = None) -> tuple[bool, str]:
    """Return true only for HOL Guard's explicit benign allow result."""
    # Command inspection is path-sensitive. Never inspect a different local
    # directory when the protected tool has supplied its execution directory.
    if cwd is not None and (
        not isinstance(cwd, str) or not os.path.isabs(cwd) or not os.path.isdir(cwd)
    ):
        return False, "Guard cannot inspect the tool working directory"
    executable = shutil.which("hol-guard")
    if not executable:
        return False, "hol-guard is not installed"
    # PATH may include relative directories. Resolve the binary BEFORE changing
    # cwd, or subprocess could launch a different executable from the tool dir.
    # Bind the executable before changing cwd, then refuse binaries from the
    # untrusted checkout. An absolute PATH entry can still point into a repo.
    executable_path = os.path.abspath(executable)
    executable = os.path.realpath(executable_path)
    workspace = os.path.realpath(cwd or os.getcwd())
    project_root = workspace
    current = workspace
    while True:
        # Use the outermost Git root; a nested .git marker may be spoofed.
        if os.path.lexists(os.path.join(current, ".git")):
            project_root = current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    try:
        if any(
            os.path.commonpath((project_root, candidate)) == project_root
            for candidate in (executable_path, executable)
        ):
            return False, "Guard executable is inside the untrusted worktree"
    except ValueError:
        return False, "Guard executable location could not be verified"

    try:
        completed = subprocess.run(
            [executable, "command", "test", "--json", "--", command],
            capture_output=True,
            text=True,
            timeout=_GUARD_TIMEOUT_SECONDS,
            check=False,
            cwd=cwd,
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
        # An absent or malformed name cannot establish that this is an
        # unrelated tool. Deny instead of silently skipping the Guard gate.
        tool_name = getattr(tool_call, "name", None)
        if not isinstance(tool_name, str) or not tool_name.strip():
            return _blocked("missing tool identity")
        command_field = command_fields.get(tool_name)
        if command_field is None:
            return None

        # Malformed arguments for a protected tool must not escape the hook
        # before it can return TinyAgent's blocking control result.
        if not isinstance(args, Mapping):
            return _blocked("invalid tool arguments")

        # Preserve the complete JSON tool input across asynchronous inspection.
        # A caller must not replace the command or other execution arguments
        # after Guard classified the original request.
        try:
            guarded_args = copy.deepcopy(dict(args))
        except Exception:
            return _blocked("invalid tool arguments")

        command = guarded_args.get(command_field)
        if not isinstance(command, str) or not command.strip():
            return _blocked("missing command input")

        # For local tools with an explicit cwd, evaluate paths from that exact
        # directory. A relative, missing, or nonlocal path must fail closed.
        cwd = guarded_args.get("cwd")
        if cwd is not None and (
            not isinstance(cwd, str) or not os.path.isabs(cwd) or not os.path.isdir(cwd)
        ):
            return _blocked("invalid local working directory")

        try:
            if cwd is None:
                allowed, reason = await asyncio.to_thread(_guard_decision, command)
            else:
                allowed, reason = await asyncio.to_thread(_guard_decision, command, cwd)
        except Exception:
            # Threadpool and environment failures outside subprocess.run must
            # never escape this optional security boundary as a hook exception.
            return _blocked("HOL Guard evaluation failed")
        if allowed:
            try:
                if copy.deepcopy(dict(args)) != guarded_args or getattr(tool_call, "name", None) != tool_name:
                    return _blocked("tool inputs changed during Guard inspection")
            except Exception:
                return _blocked("tool inputs changed during Guard inspection")
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
