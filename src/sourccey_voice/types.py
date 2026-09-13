from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class ConversationReply:
    text: str
    requested_action: str | None = None


@dataclass(frozen=True)
class SpeechAudio:
    pcm16: bytes
    sample_rate: int


class SpeechRecognizer(Protocol):
    def start(self) -> None: ...
    def push_audio(self, samples: "Sequence[float]", sample_rate: int) -> None: ...
    def partial_transcript(self) -> str: ...
    def finalize(self) -> str: ...
    def stop(self) -> None: ...


class ConversationEngine(Protocol):
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        robot_state: dict[str, object],
        available_actions: Sequence[str],
    ) -> ConversationReply: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> SpeechAudio: ...


class Speaker(Protocol):
    @property
    def is_playing(self) -> bool: ...
    def play(self, audio: SpeechAudio) -> None: ...
    def interrupt(self) -> None: ...


class RobotCommandAdapter(Protocol):
    def available_actions(self) -> set[str]: ...
    def execute(self, action: str) -> tuple[bool, str]: ...
    def state(self) -> dict[str, object]: ...

