from __future__ import annotations

import asyncio
from types import SimpleNamespace

from taskpaw.agent import QueryEngine
from taskpaw.agent.loop_guard import (
    LoopGuard,
    TurnObservation,
    action_signature,
    build_observation,
    error_category,
    is_transient_error,
)
from taskpaw.config import LoopConfig, load_config
from taskpaw.tools import ToolRegistry, get_builtin_tools


def make_guard(**overrides) -> LoopGuard:
    options = {
        "max_turns": 20,
        "grace_turns": 0,
        "max_extensions": 0,
        "hard_cap_turns": 60,
    }
    options.update(overrides)
    return LoopGuard(LoopConfig(**options), context_window=200_000)


def read_turn(path: str = "a.txt", *, error: str | None = None, usage: int = 0) -> TurnObservation:
    if error is None:
        result = {"name": "read_file", "is_error": False, "content": "file content"}
    else:
        result = {"name": "read_file", "is_error": True, "content": error}
    return TurnObservation(
        actions=[{"name": "read_file", "input": {"path": path}}],
        results=[result],
        usage_tokens=usage,
    )


def write_turn(path: str, *, usage: int = 0) -> TurnObservation:
    return TurnObservation(
        actions=[{"name": "write_file", "input": {"path": path, "content": "x"}}],
        results=[{"name": "write_file", "is_error": False, "content": "wrote"}],
        usage_tokens=usage,
    )


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def test_signature_normalises_equivalent_paths():
    assert action_signature("read_file", {"path": "./a.txt"}) == action_signature(
        "read_file", {"path": "a.txt"}
    )
    assert action_signature("read_file", {"path": "a.txt"}) != action_signature(
        "read_file", {"path": "b.txt"}
    )


def test_error_classification():
    assert error_category("No such file: a.txt") == "not_found"
    assert error_category("Permission denied") == "permission"
    assert is_transient_error("Request timed out after 120s")
    assert not is_transient_error("No such file: a.txt")


def test_build_observation_detects_file_mutation():
    calls = [
        {
            "id": "1",
            "function": {"name": "write_file", "arguments": '{"path":"a.txt","content":"x"}'},
        }
    ]
    results = [SimpleNamespace(tool_use_id="1", is_error=False, content="wrote")]
    observation = build_observation(calls, results)
    assert observation.file_changed
    assert not observation.command_succeeded
    assert observation.actions[0]["input"]["path"] == "a.txt"


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------


def test_identical_actions_escalate_from_warning_to_wrapup():
    guard = make_guard()
    actions = [guard.observe(turn + 1, read_turn()).action for turn in range(5)]
    assert actions[:2] == ["continue", "continue"]
    assert actions[2] == "warn"
    assert actions[4] == "wrapup"
    assert guard.stop_reason == "repeated_action"


def test_different_arguments_are_not_repetition():
    guard = make_guard()
    for turn in range(6):
        decision = guard.observe(turn + 1, read_turn(f"file{turn}.txt"))
        assert decision.action == "continue"


def test_alternating_actions_are_caught_by_the_window():
    guard = make_guard(repeat_warn=3, repeat_abort=5, repeat_window=6)
    actions = []
    for turn in range(8):
        name = "read_file" if turn % 2 == 0 else "grep"
        action = {"name": name, "input": {"path": "a.txt"}}
        result = {"name": name, "is_error": False, "content": "x"}
        decision = guard.observe(turn + 1, TurnObservation(actions=[action], results=[result]))
        actions.append(decision.action)
    assert "warn" in actions


def test_transient_errors_are_exempt_from_repetition():
    guard = make_guard()
    for turn in range(8):
        decision = guard.observe(turn + 1, read_turn(error="Request timed out after 120s"))
        assert decision.action == "continue"


def test_repeated_permanent_errors_escalate():
    guard = make_guard()
    actions = [
        guard.observe(turn + 1, read_turn(error="No such file: a.txt")).action for turn in range(5)
    ]
    assert actions[2] == "warn"
    assert actions[4] == "wrapup"
    assert guard.stop_reason == "repeated_error"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_token_budget_warns_then_wraps_up():
    guard = LoopGuard(
        LoopConfig(max_run_tokens=1000, budget_warn=0.8, budget_abort=1.0), context_window=0
    )
    assert guard.observe(1, read_turn(path="a", usage=500)).action == "continue"
    warning = guard.observe(2, read_turn(path="b", usage=350))
    assert warning.action == "warn"
    assert warning.reason == "budget_warning"
    stopping = guard.observe(3, read_turn(path="c", usage=200))
    assert stopping.action == "wrapup"
    assert stopping.reason == "budget_exhausted"


def test_missing_usage_falls_back_to_estimate():
    guard = LoopGuard(LoopConfig(max_run_tokens=100), context_window=0)
    observation = read_turn(path="a")
    observation.estimated_tokens = 120
    decision = guard.observe(1, observation)
    assert decision.action == "wrapup"
    assert decision.reason == "budget_exhausted"


# ---------------------------------------------------------------------------
# Turn budget
# ---------------------------------------------------------------------------


def test_turn_cap_wraps_up_without_progress():
    guard = make_guard(max_turns=2)
    assert guard.observe(1, read_turn("a.txt")).action == "continue"
    decision = guard.observe(2, read_turn("b.txt"))
    assert decision.action == "wrapup"
    assert decision.reason == "turn_cap"


def test_turn_cap_extends_when_progress_is_visible():
    guard = make_guard(max_turns=2, extension_turns=5, max_extensions=1)
    assert guard.observe(1, write_turn("a.txt")).action == "continue"
    decision = guard.observe(2, write_turn("b.txt"))
    assert decision.action == "extend"
    assert decision.detail["max_turns"] == 7
    assert guard.max_turns == 7


def test_extension_is_blocked_when_budget_is_nearly_spent():
    guard = LoopGuard(
        LoopConfig(
            max_turns=2,
            grace_turns=0,
            extension_turns=5,
            max_extensions=1,
            max_run_tokens=1000,
        ),
        context_window=0,
    )
    assert guard.observe(1, write_turn("a.txt", usage=400)).action == "continue"
    assert guard.observe(2, write_turn("b.txt", usage=400)).action == "warn"
    decision = guard.observe(3, write_turn("c.txt"))
    assert decision.action == "wrapup"
    assert decision.reason == "turn_cap"


def test_hard_cap_wins_over_progress():
    guard = make_guard(max_turns=2, hard_cap_turns=3, extension_turns=5, max_extensions=2)
    assert guard.observe(1, write_turn("a.txt")).action == "continue"
    assert guard.observe(2, write_turn("b.txt")).action == "extend"
    decision = guard.observe(3, write_turn("c.txt"))
    assert decision.action == "wrapup"
    assert decision.reason == "hard_cap"


def test_stagnation_wraps_up_without_new_information():
    guard = make_guard(stagnation_warn=2, stagnation_abort=4)
    observation = TurnObservation(actions=[], results=[])
    actions = [guard.observe(turn + 1, observation).action for turn in range(4)]
    assert actions[1] == "warn"
    assert actions[3] == "wrapup"
    assert guard.stop_reason == "stagnation"


# ---------------------------------------------------------------------------
# Integration: the loop must wrap up instead of truncating silently
# ---------------------------------------------------------------------------


class LoopingClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 200_000

    def __init__(self) -> None:
        self.calls = 0
        self.tools_per_call: list[list[dict]] = []
        self.system_prompts: list[str] = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        self.tools_per_call.append(list(tools))
        self.system_prompts.append(system_prompt)
        if not tools:
            yield {"type": "text_delta", "text": "final summary"}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        yield {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "id": f"call_{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path":"missing.txt"}'},
            },
        }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class AnswerOnlyClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 10_000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "hello"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_query_result_reports_end_turn_for_normal_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    engine = QueryEngine(
        llm_client=AnswerOnlyClient(),
        tool_registry=ToolRegistry(),
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(engine.ask_complete_async("hi"))

    assert result.text == "hello"
    assert result.stop_reason == "end_turn"


def test_agent_wraps_up_instead_of_truncating_silently(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.llm.api_key = "test-key"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    client = LoopingClient()
    engine = QueryEngine(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run() -> list[dict]:
        return [event async for event in engine.ask("read the missing file")]

    events = asyncio.run(run())
    done = next(event for event in events if event.get("type") == "done")

    assert any(event.get("type") == "loop_warning" for event in events)
    assert any(event.get("type") == "loop_wrapup" for event in events)
    assert done["stop_reason"] == "repeated_error"
    # The wrap-up turn must be sent without tools.
    assert client.tools_per_call[-1] == []
    assert "final summary" in done["messages"][-2].content or "final summary" in str(
        done["messages"]
    )
