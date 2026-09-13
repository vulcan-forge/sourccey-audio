from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from .commands import CommandName, CommandRegistry, PRIORITY_COMMANDS


@dataclass(frozen=True)
class WakeDecision:
    accepted: bool
    text: str
    woke: bool = False
    reason: str = ""


class WakeSession:
    def __init__(
        self,
        enabled: bool,
        phrases: Iterable[str],
        engaged_seconds: float,
        emergency_bypass: bool,
        registry: CommandRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.phrases = tuple(sorted((phrase.casefold().strip() for phrase in phrases), key=len, reverse=True))
        self.engaged_seconds = engaged_seconds
        self.emergency_bypass = emergency_bypass
        self.registry = registry
        self._clock = clock
        self._engaged_until = 0.0

    @property
    def engaged(self) -> bool:
        return not self.enabled or self._clock() <= self._engaged_until

    def evaluate(self, transcript: str) -> WakeDecision:
        text = transcript.strip()
        explicit = self.registry.parse_explicit(text)
        if (
            self.enabled
            and self.emergency_bypass
            and explicit is not None
            and explicit.name in PRIORITY_COMMANDS
        ):
            self._engage()
            return WakeDecision(True, text, reason="emergency_bypass")

        phrase, remainder = self._strip_wake_phrase(text)
        if phrase:
            self._engage()
            return WakeDecision(bool(remainder), remainder, woke=True, reason="wake_phrase")
        if self.engaged:
            self._engage()
            return WakeDecision(True, text, reason="engaged_window")
        return WakeDecision(False, text, reason="wake_required")

    def _engage(self) -> None:
        self._engaged_until = self._clock() + self.engaged_seconds

    def _strip_wake_phrase(self, text: str) -> tuple[str | None, str]:
        normalized = text.casefold()
        for phrase in self.phrases:
            match = re.match(rf"^\s*{re.escape(phrase)}\b[\s,.:;!?-]*(.*)$", normalized)
            if match:
                offset = match.start(1)
                return phrase, text[offset:].strip()
        return None, text

