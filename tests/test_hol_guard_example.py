from __future__ import annotations

import subprocess
from pathlib import Path
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


@pytest.mark.parametrize(
    "stop_signal",
    [{"continue": False}, {"decision": "deny"}, {"decision": "review"}, {"decision": None}, {"decision": 0}, {"decision": []}],
)
def test_explicit_stop_cannot_be_overridden_by_benign_classification(
    monkeypatch: pytest.MonkeyPatch,
    stop_signal: dict[str, object],
) -> None:
    import json

    payload = {
        "minimum_action": "allow",
        "classification": {"explicitly_benign": True},
        **stop_signal,
    }
    _patch_guard_run(monkeypatch, stdout=json.dumps(payload))
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert "explicitly stopped" in reason


@pytest.mark.parametrize("bad_continue", [None, "false", 0, 1, [], {}])
def test_malformed_continue_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    bad_continue: object,
) -> None:
    import json

    _patch_guard_run(
        monkeypatch,
        stdout=json.dumps({
            "continue": bad_continue,
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
        }),
    )
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert "malformed continuation" in reason


@pytest.mark.parametrize(
    "policy_action",
    ["block", "deny", "review", "ask", "warn", "sandbox-required", None, 1],
)
def test_conflicting_policy_action_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    policy_action: object,
) -> None:
    """Do not run a tool when a concrete policy stop contradicts benign inspection."""
    import json

    _patch_guard_run(
        monkeypatch,
        stdout=json.dumps({
            "policy_action": policy_action,
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
        }),
    )
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert "non-allow policy action" in reason


@pytest.mark.parametrize(
    "hook_output",
    [
        None,
        {"permissionDecision": "deny", "hookEventName": "PreToolUse"},
        {"permissionDecision": "ask", "hookEventName": "PreToolUse"},
        {"permissionDecision": "allow", "hookEventName": "PostToolUse"},
        {"permissionDecision": "allow"},
        "malformed",
    ],
)
def test_hook_level_contradiction_blocks_benign_command(
    monkeypatch: pytest.MonkeyPatch,
    hook_output: object,
) -> None:
    """An extended Guard response must not authorize against an explicit stop."""
    import json

    _patch_guard_run(
        monkeypatch,
        stdout=json.dumps({
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
            "hookSpecificOutput": hook_output,
        }),
    )
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert "contradicts" in reason


def test_matching_explicit_hook_allow_preserves_benign_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correctly bound explicit hook allow does not disable benign classification."""
    import json

    _patch_guard_run(
        monkeypatch,
        stdout=json.dumps({
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
            "hookSpecificOutput": {
                "permissionDecision": "allow",
                "hookEventName": "PreToolUse",
            },
        }),
    )
    allowed, _ = example._guard_decision("git status")
    assert allowed is True


def test_inspection_only_policy_state_preserves_benign_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit inspection-only state is not a separate block decision."""
    import json

    _patch_guard_run(
        monkeypatch,
        stdout=json.dumps({
            "policy_action": "not_run",
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
        }),
    )
    allowed, _ = example._guard_decision("git status")
    assert allowed is True


@pytest.mark.parametrize(
    "stdout",
    [
        '{"classification":{"explicitly_benign":true},"minimum_action":"review","minimum_action":"allow"}',
        '{"classification":{"explicitly_benign":false,"explicitly_benign":true},"minimum_action":"allow"}',
        '{"classification":{"explicitly_benign":true},"continue":false,"continue":true,"minimum_action":"allow"}',
    ],
)
def test_duplicate_guard_json_fields_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    """A later JSON key must not overwrite an earlier Guard review or deny."""
    _patch_guard_run(monkeypatch, stdout=stdout)
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert reason == "Guard evaluation failed"


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


@pytest.mark.parametrize("arguments", [{}, {"command": ""}, {"command": "   "}, {"command": 123}])
async def test_malformed_mapped_command_is_blocked_without_guard_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
) -> None:
    """Fail closed before Guard evaluation when mapped command input is unusable."""
    def unexpected_guard_call(command: str) -> tuple[bool, str]:
        raise AssertionError(f"Guard should not run for malformed command input: {command}")

    monkeypatch.setattr(example, "_guard_decision", unexpected_guard_call)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(id="tc_1", name="shell", arguments=arguments)

    result = await hook(call, None, call.arguments)

    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}


@pytest.mark.parametrize("name", [None, "", "  ", 42, []])
async def test_missing_tool_identity_is_blocked_without_guard_call(
    monkeypatch: pytest.MonkeyPatch, name: object
) -> None:
    def unexpected_guard_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Guard must not be invoked for an invalid tool identity")

    monkeypatch.setattr(example, "_guard_decision", unexpected_guard_call)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = SimpleNamespace(name=name, arguments={"command": "git status"})

    result = await hook(call, None, call.arguments)
    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}
    assert "missing tool identity" in result.result.content[0].text


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


@pytest.mark.parametrize(
    "failure",
    [
        OSError("Guard executable is not runnable"),
        subprocess.TimeoutExpired(cmd=["hol-guard"], timeout=10),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
    ],
)
def test_guard_subprocess_exceptions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    """A launcher error, timeout or undecodable output must never authorize a tool."""
    monkeypatch.setattr(example.shutil, "which", lambda name: "/usr/bin/hol-guard")

    def failed_run(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(example.subprocess, "run", failed_run)
    allowed, reason = example._guard_decision("git status")
    assert allowed is False
    assert reason == "Guard evaluation failed"


def test_excessively_nested_guard_json_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid but overdeep Guard JSON must return a denial, not raise from the hook."""
    deeply_nested_json = "[" * 20000 + "]" * 20000
    _patch_guard_run(monkeypatch, stdout=deeply_nested_json)

    allowed, reason = example._guard_decision("git status")

    assert allowed is False
    assert reason == "Guard evaluation failed"


async def test_unexpected_evaluator_failure_returns_a_blocked_tool_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected threaded Guard failure must not escape the policy hook."""
    def broken_guard(_command: str) -> tuple[bool, str]:
        raise RuntimeError("threaded Guard evaluation failed")

    monkeypatch.setattr(example, "_guard_decision", broken_guard)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(id="tc_2", name="shell", arguments={"command": "git status"})
    result = await hook(call, None, call.arguments)

    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}
    assert result.result.content[0].text == "Blocked by HOL Guard: HOL Guard evaluation failed"

@pytest.mark.parametrize("arguments", [None, [], "unexpected arguments"])
async def test_non_mapping_protected_tool_arguments_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
    arguments: object,
) -> None:
    """The security hook must return a block even for invalid argument shapes."""
    def unexpected_guard_call(command: str) -> tuple[bool, str]:
        raise AssertionError(f"Guard should not run for invalid arguments: {command}")

    monkeypatch.setattr(example, "_guard_decision", unexpected_guard_call)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(id="tc_invalid", name="shell", arguments={"command": "git status"})
    result = await hook(call, None, arguments)  # type: ignore[arg-type]

    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}
    assert result.result.content[0].text == "Blocked by HOL Guard: invalid tool arguments"


def test_guard_inspects_explicit_local_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Path-sensitive Guard checks must use the protected tool's cwd."""
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        seen["cwd"] = kwargs.get("cwd")
        seen["argv"] = argv
        return SimpleNamespace(
            returncode=0,
            stdout=_guard_output(explicitly_benign=True, minimum_action="allow"),
        )

    monkeypatch.setattr(example.shutil, "which", lambda name: "/usr/bin/hol-guard")
    monkeypatch.setattr(example.subprocess, "run", fake_run)
    allowed, _ = example._guard_decision("cat config.toml", cwd=str(tmp_path))
    assert allowed is True
    assert seen["cwd"] == str(tmp_path)
    assert seen["argv"][-1] == "cat config.toml"


@pytest.mark.parametrize("bad_cwd", ["", ".", "../workspace", 0, False])
def test_guard_rejects_invalid_working_directory_without_launching(
    monkeypatch: pytest.MonkeyPatch, bad_cwd: object
) -> None:
    """Missing absolute local path cannot become an unsafe default directory."""
    monkeypatch.setattr(
        example.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("Guard must not inspect a different cwd"),
    )
    allowed, reason = example._guard_decision("cat config.toml", cwd=bad_cwd)  # type: ignore[arg-type]
    assert allowed is False
    assert "working directory" in reason


async def test_mapped_local_cwd_is_forwarded_to_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_guard(command: str, cwd: str | None = None) -> tuple[bool, str]:
        seen.update({"command": command, "cwd": cwd})
        return True, "guard_allow"

    monkeypatch.setattr(example, "_guard_decision", fake_guard)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(
        id="tc_cwd", name="shell",
        arguments={"command": "cat ./config.toml", "cwd": str(tmp_path)},
    )
    result = await hook(call, None, call.arguments)
    assert result is None
    assert seen == {"command": "cat ./config.toml", "cwd": str(tmp_path)}


@pytest.mark.parametrize("cwd", ["", ".", "../other", 45])
async def test_mapped_tool_with_invalid_cwd_blocks_before_inspection(
    monkeypatch: pytest.MonkeyPatch, cwd: object
) -> None:
    monkeypatch.setattr(
        example, "_guard_decision",
        lambda *args, **kwargs: pytest.fail("Invalid cwd must not call Guard"),
    )
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})
    call = ToolCallContent(
        id="tc_bad_cwd", name="shell",
        arguments={"command": "cat config.toml", "cwd": cwd},
    )
    result = await hook(call, None, call.arguments)
    assert result is not None and result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}


def test_guard_binary_is_fixed_before_switching_tool_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not allow a relative PATH shim under the hook's launch checkout."""
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / ".git").mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / ".git").mkdir()
    monkeypatch.chdir(launcher)
    monkeypatch.setattr(example.shutil, "which", lambda name: "./fake-bin/hol-guard")
    monkeypatch.setattr(
        example.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("launch-checkout shim must not execute"),
    )
    allowed, reason = example._guard_decision("cat config.toml", cwd=str(target))
    assert allowed is False
    assert reason == "Guard executable is inside the untrusted worktree"


@pytest.mark.parametrize("symlinked", [False, True])
def test_guard_rejects_launcher_checkout_binary_for_other_tool_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, symlinked: bool
) -> None:
    """An absolute executable in the launcher repo is untrusted across cwd changes."""
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / ".git").mkdir()
    executable = launcher / "bin" / "hol-guard"
    executable.parent.mkdir()
    if symlinked:
        external = tmp_path / "external" / "hol-guard"
        external.parent.mkdir()
        external.write_text("test binary", encoding="utf-8")
        try:
            executable.symlink_to(external)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation is unavailable")
    else:
        executable.write_text("test binary", encoding="utf-8")
    tool_cwd = tmp_path / "other-repo"
    tool_cwd.mkdir()
    (tool_cwd / ".git").mkdir()
    monkeypatch.chdir(launcher)
    monkeypatch.setattr(example.shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(
        example.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("untrusted executable must not run"),
    )
    allowed, reason = example._guard_decision("pwd", cwd=str(tool_cwd))
    assert allowed is False
    assert reason == "Guard executable is inside the untrusted worktree"


def test_guard_allows_external_installed_binary_for_other_tool_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trusted external Guard remains usable when Bash chooses another repository."""
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    (launcher / ".git").mkdir()
    tool_cwd = tmp_path / "other-repo"
    tool_cwd.mkdir()
    (tool_cwd / ".git").mkdir()
    binary = tmp_path / "trusted-bin" / "hol-guard"
    binary.parent.mkdir()
    binary.write_text("test binary", encoding="utf-8")
    monkeypatch.chdir(launcher)
    monkeypatch.setattr(example.shutil, "which", lambda name: str(binary))
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        seen["argv"] = argv
        seen["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(
            returncode=0,
            stdout=_guard_output(explicitly_benign=True, minimum_action="allow"),
        )

    monkeypatch.setattr(example.subprocess, "run", fake_run)
    allowed, _ = example._guard_decision("pwd", cwd=str(tool_cwd))
    assert allowed is True
    assert seen["argv"][0] == str(binary)
    assert seen["cwd"] == str(tool_cwd)

@pytest.mark.parametrize("spoof_nested_git", [False, True])
def test_checkout_guard_binary_cannot_impersonate_installed_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spoof_nested_git: bool
) -> None:
    """A project-owned binary must not supply an allow, even with a nested .git."""
    repo = tmp_path / "repo"
    nested = repo / "src"
    bin_dir = repo / "bin"
    nested.mkdir(parents=True)
    bin_dir.mkdir()
    (repo / ".git").mkdir()
    if spoof_nested_git:
        (nested / ".git").mkdir()
    project_executable = bin_dir / "hol-guard"
    project_executable.write_text("#!/bin/sh\\nexit 0\\n")

    monkeypatch.setattr(example.shutil, "which", lambda _: str(project_executable))
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("project-owned Guard must never launch"),
    )
    allowed, reason = example._guard_decision("git status", cwd=str(nested))
    assert allowed is False
    assert reason == "Guard executable is inside the untrusted worktree"


def test_guard_symlink_inside_worktree_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A project-local PATH symlink cannot launder a supposedly trusted target."""
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / ".git").mkdir()
    trusted = tmp_path / "outside" / "hol-guard"
    trusted.parent.mkdir()
    trusted.write_text("#!/bin/sh\\nexit 0\\n")
    local_bin = repo / "bin"
    local_bin.mkdir()
    candidate = local_bin / "hol-guard"
    try:
        candidate.symlink_to(trusted)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported")

    monkeypatch.setattr(example.shutil, "which", lambda _: str(candidate))
    monkeypatch.setattr(
        example.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("worktree-controlled symlink must not launch"),
    )
    allowed, reason = example._guard_decision("git status", cwd=str(repo))
    assert allowed is False
    assert reason == "Guard executable is inside the untrusted worktree"


@pytest.mark.parametrize("mutation", ["command", "nested_arguments"])
async def test_async_guard_rejects_argument_mutation(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """A tool cannot execute different inputs from those classified by Guard."""
    call = ToolCallContent(
        id="tc_input_mutation",
        name="shell",
        arguments={"command": "git status", "options": {"mode": "safe"}},
    )

    def fake_guard(command: str) -> tuple[bool, str]:
        assert command == "git status"
        if mutation == "command":
            call.arguments["command"] = "rm -rf build"
        else:
            call.arguments["options"]["mode"] = "unsafe"
        return True, "guard_allow"

    monkeypatch.setattr(example, "_guard_decision", fake_guard)
    hook = example.make_hol_guard_before_tool_call({"shell": "command"})

    result = await hook(call, None, call.arguments)

    assert result is not None
    assert result.is_error is True
    assert result.result is not None
    assert result.result.details == {"policy": "hol-guard"}
    assert "tool inputs changed during Guard inspection" in result.result.content[0].text
