from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


class ConversationHistory:
    def __init__(
        self,
        max_turns: int,
        max_characters: int,
        inactivity_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_turns = max_turns
        self.max_characters = max_characters
        self.inactivity_seconds = inactivity_seconds
        self._clock = clock
        self._turns: deque[Turn] = deque()
        self._last_activity: float | None = None

    def add(self, user: str, assistant: str) -> None:
        self._expire_if_idle()
        self._turns.append(Turn(user.strip(), assistant.strip()))
        self._last_activity = self._clock()
        while len(self._turns) > self.max_turns or self.character_count > self.max_characters:
            self._turns.popleft()

    def messages(self) -> list[dict[str, str]]:
        self._expire_if_idle()
        result: list[dict[str, str]] = []
        for turn in self._turns:
            result.extend(
                (
                    {"role": "user", "content": turn.user},
                    {"role": "assistant", "content": turn.assistant},
                )
            )
        return result

    @property
    def character_count(self) -> int:
        return sum(len(turn.user) + len(turn.assistant) for turn in self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._last_activity = None

    def _expire_if_idle(self) -> None:
        if self._last_activity is None:
            return
        if self._clock() - self._last_activity > self.inactivity_seconds:
            self.clear()

