from __future__ import annotations

from collections.abc import Sequence

from sourccey_voice.types import ConversationReply, SpeechAudio


class FakeRecognizer:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.started = 0
        self.pushed: list[float] = []

    def start(self) -> None:
        self.started += 1

    def push_audio(self, samples: Sequence[float], sample_rate: int) -> None:
        assert sample_rate == 16000
        self.pushed.extend(samples)

    def partial_transcript(self) -> str:
        return self.transcript[:5]

    def finalize(self) -> str:
        return self.transcript

    def stop(self) -> None:
        return None


class FakeProbability:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def probability(self, samples, sample_rate: int) -> float:
        del samples, sample_rate
        return next(self.values)

    def reset(self) -> None:
        return None


class FakeConversation:
    def __init__(self, reply: ConversationReply | None = None, error: Exception | None = None) -> None:
        self.reply = reply or ConversationReply("Hello.")
        self.error = error
        self.calls = 0

    def generate(self, messages, robot_state, available_actions):
        del messages, robot_state, available_actions
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


class FakeTts:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts: list[str] = []

    def synthesize(self, text: str) -> SpeechAudio:
        self.texts.append(text)
        if self.error:
            raise self.error
        return SpeechAudio(b"\x00\x00" * 160, 16000)


class FakeSpeaker:
    def __init__(self, playing: bool = False) -> None:
        self.playing = playing
        self.played: list[SpeechAudio] = []
        self.interruptions = 0

    @property
    def is_playing(self) -> bool:
        return self.playing

    def play(self, audio: SpeechAudio) -> None:
        self.played.append(audio)
        self.playing = True

    def interrupt(self) -> None:
        self.interruptions += 1
        self.playing = False


class FakeRobot:
    def __init__(self, available=None, success: bool = True, error: Exception | None = None) -> None:
        self.available = set(available or [])
        self.success = success
        self.error = error
        self.executed: list[str] = []

    def available_actions(self) -> set[str]:
        return set(self.available)

    def execute(self, action: str) -> tuple[bool, str]:
        self.executed.append(action)
        if self.error:
            raise self.error
        return self.success, "done" if self.success else "hardware refused"

    def state(self) -> dict[str, object]:
        return {"mode": "test"}

