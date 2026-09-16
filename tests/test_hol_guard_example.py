from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples import hol_guard_before_tool_call as example
from tinyagent import AgentTool, ToolCallContent


def _tool_call() -> ToolCallContent:
    return ToolCallContent(id="tc_1", name="shell", arguments={"command": "git status"})


async def test_explicit_allow_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(example, "_guard_decision", lambda command: (True, "decision was benign"))
    result = await example.hol_guard_before_tool_call(_tool_call(), None, {"command": "git status"})
    assert result is None


async def test_review_blocks_before_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(example, "_guard_decision", lambda command: (False, "decision was review"))
    result = await example.hol_guard_before_tool_call(_tool_call(), None, {"command": "rm -rf build"})
    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}


def test_guard_cli_failure_is_not_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(example.shutil, "which", lambda name: "/usr/bin/hol-guard")
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    allowed, _ = example._guard_decision("git status")
    assert allowed is False


async def test_non_command_tool_is_unchanged() -> None:
    tool = AgentTool(name="search", description="Search")
    result = await example.hol_guard_before_tool_call(_tool_call(), tool, {"query": "status"})
    assert result is None
