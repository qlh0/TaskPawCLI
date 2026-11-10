"""loop_guard.py — run-level guard rails for the agent loop.

The guard answers one question at every turn boundary: *should this run keep
going?* It watches four signals:

* repeated actions — the model keeps issuing the same tool call
* repeated errors  — the same tool keeps failing the same way
* stagnation       — several turns with no new information and no side effect
* budgets          — tokens burned and turns used

When a limit trips, the guard asks the caller to *wrap up* (one last model turn
with tools disabled) instead of truncating the answer silently. It also owns the
turn budget, so a run that keeps making progress can be extended instead of
being cut off at a fixed turn count.

The guard is a pure-ish state machine: it never touches the network or the
filesystem, which keeps it cheap to unit test.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Literal

from taskpaw.config import LoopConfig

LoopAction = Literal["continue", "warn", "extend", "wrapup", "abort"]

_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "revert_turn"})
_COMMAND_TOOLS = frozenset({"bash", "execute_command"})
_PATH_KEYS = frozenset(
    {"path", "file", "filename", "filepath", "file_path", "dir", "directory", "cwd", "target"}
)

# Errors that usually clear up on their own. Retrying them is normal behaviour,
# so they are exempt from the repetition counters.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "429",
    "rate limit",
    "too many requests",
    "could not connect",
    "connection reset",
    "connection refused",
    "502",
    "503",
    "504",
    "overloaded",
    "temporarily",
    "try again",
)

_ERROR_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("not_found", ("no such file", "not found", "cannot find", "找不到")),
    ("permission", ("permission denied", "access denied", "拒绝访问", "permissionerror")),
    ("syntax", ("syntaxerror", "syntax error", "invalid syntax")),
    ("timeout", ("timeout", "timed out")),
    ("rate_limit", ("rate limit", "too many requests", "429")),
    ("connection", ("connection", "could not connect", "network")),
)

_TOOL_HINTS = {
    "read_file": "use glob to locate the file name first, then read the exact path",
    "write_file": "re-read the file and use edit_file for a targeted change",
    "edit_file": "re-read the current content before editing again",
    "bash": "read the full error output and change the command or working directory",
    "execute_command": "read the full error output and change the command",
    "grep": "broaden the pattern or search a different directory",
    "glob": "broaden the pattern, or list the directory to inspect the layout",
}

_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class TurnObservation:
    """Everything the guard needs to know about one completed turn."""

    actions: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    usage_tokens: int = 0
    estimated_tokens: int = 0
    file_changed: bool = False
    command_succeeded: bool = False
    plan_step_completed: bool = False


@dataclass(slots=True)
class GuardDecision:
    action: LoopAction
    reason: str = ""
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def build_observation(
    tool_calls: list[dict[str, Any]],
    results: list[Any],
    *,
    usage_tokens: int = 0,
    estimated_tokens: int = 0,
) -> TurnObservation:
    """Convert raw tool calls/results into the shape the guard understands."""

    actions = [
        {
            "name": str(call.get("function", {}).get("name") or "unknown"),
            "input": _parse_arguments(call),
        }
        for call in tool_calls
    ]
    observed_results = [
        {
            "name": _tool_name_by_id(tool_calls, str(getattr(result, "tool_use_id", "") or "")),
            "is_error": bool(getattr(result, "is_error", False)),
            "content": str(getattr(result, "content", "") or ""),
        }
        for result in results
    ]

    file_changed = any(
        action["name"] in _MUTATING_TOOLS
        and not any(
            result["name"] == action["name"] and result["is_error"] for result in observed_results
        )
        for action in actions
    )
    command_succeeded = any(
        result["name"] in _COMMAND_TOOLS
        and not result["is_error"]
        and bool(result["content"].strip())
        for result in observed_results
    )
    return TurnObservation(
        actions=actions,
        results=observed_results,
        usage_tokens=max(0, usage_tokens),
        estimated_tokens=max(0, estimated_tokens),
        file_changed=file_changed,
        command_succeeded=command_succeeded,
    )


def action_signature(name: str, payload: dict[str, Any]) -> str:
    """Stable fingerprint for one tool call (normalised arguments included)."""

    canonical = json.dumps(
        _normalize_payload(payload),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha1(f"{name}:{canonical}".encode()).hexdigest()  # noqa: S324 - not security
    return digest[:16]


def error_category(content: str) -> str:
    lowered = content.lower()
    for category, markers in _ERROR_CATEGORIES:
        if any(marker in lowered for marker in markers):
            return category
    return "error"


def is_transient_error(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


class LoopGuard:
    """Tracks one run and decides whether it may continue."""

    def __init__(self, config: LoopConfig, *, context_window: int = 0) -> None:
        self.config = config
        self.context_window = context_window

        self.max_turns = max(1, config.max_turns)
        self.hard_cap_turns = max(self.max_turns, config.hard_cap_turns)
        self.turn = 0
        self.total_tokens = 0
        self.extensions_used = 0
        self.grace_left = max(0, config.grace_turns)

        self.wrapup_active = False
        self.wrapup_message = ""
        self.stop_reason = ""

        self._window: deque[str] = deque(maxlen=max(1, config.repeat_window))
        self._window_counts: Counter[str] = Counter()
        self._seen_signatures: set[str] = set()
        self._streak_signature = ""
        self._streak = 0
        self._last_action: dict[str, Any] = {}

        self._error_window: deque[str] = deque(maxlen=max(1, config.repeat_window))
        self._error_window_counts: Counter[str] = Counter()
        self._error_streak_signature = ""
        self._error_streak = 0
        self._last_error_signature = ""
        self._last_error_text = ""
        self._transient_retries = 0

        self._stagnation = 0
        self._progress_since_extension = False
        self._transient_only = False
        self._warned: set[str] = set()

        self._budget_limit = self._resolve_budget_limit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, turn: int, observation: TurnObservation) -> GuardDecision:
        """Record one finished turn and return the next move."""

        self.turn = max(turn, self.turn)
        spent = observation.usage_tokens or observation.estimated_tokens
        if spent:
            self.total_tokens += spent

        transient_only = self._is_transient_turn(observation)
        self._transient_only = transient_only
        new_signature = self._record_actions(observation, count=not transient_only)
        self._record_errors(observation)
        progressed = (
            observation.file_changed
            or observation.command_succeeded
            or observation.plan_step_completed
            or new_signature
        )
        if self._transient_only:
            # A retry against a flaky network is neither progress nor stagnation.
            pass
        elif progressed:
            self._stagnation = 0
            self._progress_since_extension = True
        else:
            self._stagnation += 1

        if self.turn >= self.hard_cap_turns:
            return self._wrapup(
                "hard_cap",
                f"Hard turn cap ({self.hard_cap_turns}) reached. Stop calling tools and produce "
                "your final answer now: what is done, what is missing, and why.",
                {"turn": self.turn, "hard_cap": self.hard_cap_turns},
            )

        for check in (
            self._budget_decision,
            self._error_decision,
            self._repeat_decision,
            self._stagnation_decision,
            self._turn_decision,
        ):
            decision = check()
            if decision is not None:
                return decision
        return GuardDecision("continue")

    def budget_ratio(self) -> float:
        if self._budget_limit <= 0:
            return 0.0
        return self.total_tokens / self._budget_limit

    @staticmethod
    def _is_transient_turn(observation: TurnObservation) -> bool:
        if not observation.results:
            return False
        return all(
            result.get("is_error") and is_transient_error(str(result.get("content") or ""))
            for result in observation.results
        )

    # ------------------------------------------------------------------
    # Signal recording
    # ------------------------------------------------------------------

    def _record_actions(self, observation: TurnObservation, *, count: bool = True) -> bool:
        if not count:
            # Identical calls that only failed because of a transient error are
            # normal retries; they must not feed the repetition counters.
            return False
        new_signature = False
        for action in observation.actions:
            name = str(action.get("name") or "unknown")
            payload = action.get("input") or {}
            signature = action_signature(name, payload if isinstance(payload, dict) else {})
            if signature not in self._seen_signatures:
                self._seen_signatures.add(signature)
                new_signature = True
            _push(self._window, self._window_counts, signature)

            if signature == self._streak_signature:
                self._streak += 1
            else:
                self._streak_signature = signature
                self._streak = 1
            self._last_action = {"name": name, "input": payload}
        return new_signature

    def _record_errors(self, observation: TurnObservation) -> None:
        for result in observation.results:
            if not result.get("is_error"):
                self._error_streak = 0
                self._error_streak_signature = ""
                continue
            content = str(result.get("content") or "")
            name = str(result.get("name") or "unknown")
            if is_transient_error(content):
                # Normal retry behaviour: do not count towards repetition.
                self._transient_retries += 1
                self._error_streak = 0
                self._error_streak_signature = ""
                continue

            signature = _error_signature(name, content)
            _push(self._error_window, self._error_window_counts, signature)
            if signature == self._error_streak_signature:
                self._error_streak += 1
            else:
                self._error_streak_signature = signature
                self._error_streak = 1
            self._last_error_signature = signature
            self._last_error_text = content

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def _budget_decision(self) -> GuardDecision | None:
        limit = self._budget_limit
        if limit <= 0:
            return None
        ratio = self.total_tokens / limit
        detail = {"tokens": self.total_tokens, "limit": limit, "ratio": round(ratio, 3)}
        if ratio >= self.config.budget_abort:
            return self._wrapup(
                "budget_exhausted",
                f"Token budget exhausted ({self.total_tokens} of ~{limit} tokens). Stop calling "
                "tools and produce your final answer now, using the information you already have.",
                detail,
            )
        if ratio >= self.config.budget_warn and "budget" not in self._warned:
            self._warned.add("budget")
            return self._warn(
                "budget_warning",
                f"About {int(ratio * 100)}% of the token budget is used ({self.total_tokens} of "
                f"~{limit}). Prefer finishing the task over starting new explorations.",
                detail,
            )
        return None

    def _repeat_decision(self) -> GuardDecision | None:
        window_hit = self._window_counts.most_common(1)
        window_count = window_hit[0][1] if window_hit else 0
        count = max(self._streak, window_count)
        if count < self.config.repeat_warn:
            return None

        name = str(self._last_action.get("name") or "unknown")
        preview = _preview(self._last_action.get("input") or {})
        signature = self._streak_signature or (window_hit[0][0] if window_hit else "")
        detail = {"tool": name, "count": count, "input": preview}
        if count >= self.config.repeat_abort:
            return self._wrapup(
                "repeated_action",
                f"The same tool call was repeated {count} times in a row ({name} {preview}). "
                "Repeating it will not change the result. Stop calling tools and produce your "
                "final answer now: what you completed, what is blocked, and the next step.",
                detail,
            )

        key = f"action:{signature}"
        if key in self._warned:
            return None
        self._warned.add(key)
        hint = _TOOL_HINTS.get(name, "choose a different tool or a different input")
        return self._warn(
            "repeated_action",
            f"You have called {name} {count} times with the same input ({preview}). Repeating it "
            f"will not change the result. Instead: {hint}. If you are blocked, explain what is "
            "missing rather than retrying.",
            detail,
        )

    def _error_decision(self) -> GuardDecision | None:
        window_hit = self._error_window_counts.most_common(1)
        window_count = window_hit[0][1] if window_hit else 0
        count = max(self._error_streak, window_count)
        if count < self.config.repeat_warn:
            return None

        signature = self._error_streak_signature or (window_hit[0][0] if window_hit else "")
        text = _shorten(self._last_error_text, 220)
        detail = {"count": count, "error": text}
        if count >= self.config.repeat_abort:
            return self._wrapup(
                "repeated_error",
                f"The same error occurred {count} times in a row: {text}. Do not retry it. Stop "
                "calling tools and produce your final answer now, including the error and what a "
                "human would need to fix it.",
                detail,
            )

        key = f"error:{signature}"
        if key in self._warned:
            return None
        self._warned.add(key)
        return self._warn(
            "repeated_error",
            f"The same failure happened {count} times in a row: {text}. Diagnose the root cause "
            "before retrying, or use a different tool. Do not send the same input again.",
            detail,
        )

    def _stagnation_decision(self) -> GuardDecision | None:
        count = self._stagnation
        if count >= self.config.stagnation_abort:
            return self._wrapup(
                "stagnation",
                f"{count} consecutive turns produced no new information or side effect. Stop "
                "calling tools and produce your final answer now: what you verified, what is "
                "still unknown, and the most likely next step.",
                {"stagnation": count},
            )
        if count >= self.config.stagnation_warn and f"stagnation:{count}" not in self._warned:
            self._warned.add(f"stagnation:{count}")
            return self._warn(
                "stagnation_warning",
                f"The last {count} turns produced no new information. Re-read the goal, state "
                "one concrete next step, and avoid repeating the same lookups.",
                {"stagnation": count},
            )
        return None

    def _turn_decision(self) -> GuardDecision | None:
        if self.turn < self.max_turns:
            return None
        if self._can_extend():
            self.extensions_used += 1
            self.max_turns += max(1, self.config.extension_turns)
            self._progress_since_extension = False
            return GuardDecision(
                "extend",
                "extended",
                "",
                {"max_turns": self.max_turns, "extensions_used": self.extensions_used},
            )
        if self.grace_left > 0:
            self.grace_left -= 1
            key = f"turn:{self.turn}"
            if key not in self._warned:
                self._warned.add(key)
                return self._warn(
                    "turn_cap",
                    f"You have used {self.turn} of {self.max_turns} turns. Start wrapping up: "
                    "finish the current step, then summarise the result instead of starting new "
                    "work.",
                    {"turn": self.turn, "max_turns": self.max_turns},
                )
            return None
        return self._wrapup(
            "turn_cap",
            f"Turn budget reached ({self.turn} turns). Stop calling tools and produce your final "
            "answer now: what is done, what is missing, and how the user can continue.",
            {"turn": self.turn, "max_turns": self.max_turns},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _can_extend(self) -> bool:
        if self.extensions_used >= max(0, self.config.max_extensions):
            return False
        if not self._progress_since_extension:
            return False
        limit = self._budget_limit
        return not (limit > 0 and self.total_tokens / limit >= 0.7)

    def _resolve_budget_limit(self) -> int:
        if self.config.max_run_tokens > 0:
            return self.config.max_run_tokens
        if self.context_window > 0:
            return int(self.context_window * max(1.0, self.config.budget_multiplier))
        return 0

    def _warn(
        self,
        reason: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> GuardDecision:
        return GuardDecision("warn", reason, message, detail or {})

    def _wrapup(
        self,
        reason: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> GuardDecision:
        self.wrapup_active = True
        self.wrapup_message = message
        if not self.stop_reason:
            self.stop_reason = reason
        return GuardDecision("wrapup", reason, message, detail or {})


def _push(window: deque[str], counts: Counter[str], signature: str) -> None:
    if window.maxlen is not None and len(window) == window.maxlen and window:
        evicted = window[0]
        counts[evicted] -= 1
        if counts[evicted] <= 0:
            del counts[evicted]
    window.append(signature)
    counts[signature] += 1


def _error_signature(name: str, content: str) -> str:
    normalized = _WHITESPACE.sub(" ", _DIGITS.sub("#", content.lower())).strip()[:240]
    digest = hashlib.sha1(f"{name}:{normalized}".encode()).hexdigest()  # noqa: S324 - not security
    return f"{name}:{error_category(content)}:{digest[:12]}"


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        name = str(key)
        if isinstance(value, str) and name.lower() in _PATH_KEYS:
            normalized[name] = _normalize_path(value)
        else:
            normalized[name] = value
    return normalized


def _normalize_path(value: str) -> str:
    stripped = value.strip().strip('"').strip("'")
    if not stripped:
        return ""
    return os.path.normcase(os.path.normpath(stripped))


def _parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": str(raw)[:200]}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _preview(payload: Any, limit: int = 120) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    return _shorten(text, limit)


def _shorten(text: str, limit: int) -> str:
    text = _WHITESPACE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."
