from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples import hol_guard_before_tool_call as example
from tinyagent import ToolCallContent


def _guard_output(*, explicitly_benign: bool, minimum_action: str) -> str:
    """Build the minimal Guard JSON payload used by the regression tests."""
    benign = "true" if explicitly_benign else "false"
    return (
        '{"classification":{"explicitly_benign":'
        f"{benign}}},\"minimum_action\":\"{minimum_action}\"}}"
    )


def _patch_guard_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    returncode: int = 0,
) -> None:
    """Patch the Guard CLI lookup and subprocess result for one test."""
    monkeypatch.setattr(example.shutil, "which", lambda name: "/usr/bin/hol-guard")
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout),
    )


def test_guard_requires_explicit_benign_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow execution only for the explicit benign plus allow combination."""
    _patch_guard_run(
        monkeypatch,
        stdout=_guard_output(explicitly_benign=True, minimum_action="allow"),
    )
    allowed, _ = example._guard_decision("git status")
    assert allowed is True


@pytest.mark.parametrize(
    ("explicitly_benign", "minimum_action"),
    [(False, "allow"), (True, "review"), (False, "review")],
)
def test_guard_blocks_non_allow_results(
    monkeypatch: pytest.MonkeyPatch,
    explicitly_benign: bool,
    minimum_action: str,
) -> None:
    """Block every Guard result that is not an explicit benign allow."""
    _patch_guard_run(
        monkeypatch,
        stdout=_guard_output(
            explicitly_benign=explicitly_benign,
            minimum_action=minimum_action,
        ),
    )
    allowed, _ = example._guard_decision("rm -rf build")
    assert allowed is False


def test_guard_cli_failure_is_not_allow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when the Guard CLI itself returns an error."""
    _patch_guard_run(monkeypatch, stdout="", returncode=1)
    allowed, _ = example._guard_decision("git status")
    assert allowed is False


async def test_mapped_command_is_blocked_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return TinyAgent's blocking control result for a mapped denied command."""
    monkeypatch.setattr(
        example,
        "_guard_decision",
        lambda command: (False, "minimum_action=review, explicitly_benign=False"),
    )
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(id="tc_1", name="shell", arguments={"command": "rm -rf build"})

    result = await hook(call, None, call.arguments)

    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}


async def test_unmapped_tool_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave tool calls outside the configured Guard mapping untouched."""
    def unexpected_guard_call(command: str) -> tuple[bool, str]:
        """Fail if the Guard evaluator is called for an unmapped tool."""
        raise AssertionError(f"Guard should not run for unmapped tool: {command}")

    monkeypatch.setattr(example, "_guard_decision", unexpected_guard_call)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(id="tc_1", name="search", arguments={"query": "status"})

    result = await hook(call, None, call.arguments)

    assert result is None
