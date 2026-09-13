from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable


class CommandName(StrEnum):
    FOLLOW_ME = "FOLLOW_ME"
    STOP = "STOP"
    CANCEL = "CANCEL"
    FREEZE = "FREEZE"


PRIORITY_COMMANDS = frozenset({CommandName.STOP, CommandName.CANCEL, CommandName.FREEZE})


@dataclass(frozen=True)
class ParsedCommand:
    name: CommandName
    explicit: bool
    normalized_text: str

    @property
    def priority(self) -> bool:
        return self.name in PRIORITY_COMMANDS


@dataclass(frozen=True)
class CommandResult:
    command: CommandName
    accepted: bool
    message: str


def normalize_command(text: str) -> str:
    value = text.casefold().replace("_", " ")
    value = re.sub(r"[^a-z0-9\s'-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,!?:;-'\"")
    return value


class CommandRegistry:
    def __init__(self, aliases: dict[str, Iterable[str]] | None = None) -> None:
        configured = aliases or {
            "follow_me": ("follow me", "start following me", "follow", "come with me"),
            "stop": ("stop", "stop moving", "halt"),
            "cancel": ("cancel", "cancel that", "never mind"),
            "freeze": ("freeze", "freeze now", "hold position"),
        }
        self._aliases: dict[str, CommandName] = {}
        for raw_name, values in configured.items():
            name = self.validate_action_name(raw_name)
            self._aliases[normalize_command(name.value)] = name
            for alias in values:
                normalized = normalize_command(alias)
                if not normalized:
                    raise ValueError(f"empty alias for {name.value}")
                previous = self._aliases.get(normalized)
                if previous is not None and previous != name:
                    raise ValueError(f"command alias {alias!r} is ambiguous")
                self._aliases[normalized] = name

    @staticmethod
    def validate_action_name(value: object) -> CommandName:
        if not isinstance(value, str):
            raise ValueError("requested action must be a string")
        normalized = normalize_command(value).replace(" ", "_").upper()
        try:
            return CommandName(normalized)
        except ValueError as exc:
            raise ValueError(f"unapproved action: {value!r}") from exc

    def parse_explicit(self, transcript: str) -> ParsedCommand | None:
        match = re.match(r"^\s*command\s*[:,-]?\s*(.+?)\s*$", transcript, flags=re.IGNORECASE)
        if not match:
            return None
        normalized = normalize_command(match.group(1))
        name = self._aliases.get(normalized)
        if name is None:
            return None
        return ParsedCommand(name=name, explicit=True, normalized_text=normalized)

    @staticmethod
    def uses_explicit_syntax(transcript: str) -> bool:
        return re.match(r"^\s*command\b", transcript, flags=re.IGNORECASE) is not None

    def match_alias(self, text: str) -> ParsedCommand | None:
        normalized = normalize_command(text)
        name = self._aliases.get(normalized)
        return None if name is None else ParsedCommand(name, False, normalized)

    def execute(
        self,
        command: CommandName,
        available: Iterable[str],
        handler: Callable[[str], tuple[bool, str]],
    ) -> CommandResult:
        allowed = {self.validate_action_name(item) for item in available}
        if command not in allowed:
            return CommandResult(command, False, f"{command.value} is not available on this robot.")
        try:
            succeeded, message = handler(command.value)
        except Exception as exc:
            return CommandResult(command, False, f"{command.value} failed: {exc}")
        if not succeeded:
            return CommandResult(command, False, message or f"{command.value} failed.")
        return CommandResult(command, True, message or _default_ack(command))


def _default_ack(command: CommandName) -> str:
    return {
        CommandName.FOLLOW_ME: "Okay, following you.",
        CommandName.STOP: "Stopped.",
        CommandName.CANCEL: "Cancelled.",
        CommandName.FREEZE: "Holding position.",
    }[command]
