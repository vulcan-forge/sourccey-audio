from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from .config import VadConfig


class VadEvent(StrEnum):
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUING = "speech_continuing"
    SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True)
class VadUpdate:
    event: VadEvent
    samples: np.ndarray


class SpeechProbability(Protocol):
    def probability(self, samples: np.ndarray, sample_rate: int) -> float: ...
    def reset(self) -> None: ...


class EnergyProbability:
    """Dependency-free diagnostic VAD. Production config uses Silero."""

    def __init__(self, reference_rms: float = 0.035) -> None:
        self.reference_rms = reference_rms

    def probability(self, samples: np.ndarray, sample_rate: int) -> float:
        del sample_rate
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
        return min(1.0, rms / self.reference_rms)

    def reset(self) -> None:
        return None


class SileroProbability:
    def __init__(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise RuntimeError("Silero VAD is unavailable; install sourccey-voice[stt]") from exc
        self._torch = torch
        self._model = load_silero_vad()
        self._pending = np.empty(0, dtype=np.float32)
        self._last_probability = 0.0

    def probability(self, samples: np.ndarray, sample_rate: int) -> float:
        expected = 512 if sample_rate == 16000 else 256
        self._pending = np.concatenate((self._pending, samples.astype(np.float32, copy=False)))
        probabilities: list[float] = []
        while self._pending.size >= expected:
            window, self._pending = self._pending[:expected], self._pending[expected:]
            tensor = self._torch.from_numpy(window)
            probabilities.append(float(self._model(tensor, sample_rate).item()))
        # Robot audio arrives in 20 ms (320-sample) chunks, while Silero needs
        # 512 samples. Keep the most recent score for the intervening chunk;
        # returning zero there made VAD speech candidates reset every 20 ms.
        if probabilities:
            self._last_probability = max(probabilities)
        return self._last_probability

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._last_probability = 0.0
        self._model.reset_states()


class VoiceActivityDetector:
    def __init__(self, config: VadConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._candidate: list[np.ndarray] = []
        self._candidate_samples = 0
        self._speaking = False
        self._silence: list[np.ndarray] = []
        self._silence_samples = 0

    def push(self, samples: np.ndarray, probability: float) -> list[VadUpdate]:
        frame = samples.astype(np.float32, copy=False)
        speech = probability >= self.config.threshold
        if not self._speaking:
            return self._push_idle(frame, speech)

        if speech:
            updates = self._flush_silence_as_continuing()
            updates.append(VadUpdate(VadEvent.SPEECH_CONTINUING, frame))
            return updates

        self._silence.append(frame)
        self._silence_samples += frame.size
        if self._duration_ms(self._silence_samples) < self.config.silence_timeout_ms:
            return []

        post_samples = self._samples_for_ms(self.config.post_roll_ms)
        tail = _take_prefix(self._silence, post_samples)
        self._speaking = False
        self._silence.clear()
        self._silence_samples = 0
        self._candidate.clear()
        self._candidate_samples = 0
        self._remember(frame)
        return [VadUpdate(VadEvent.SPEECH_ENDED, tail)]

    def reset(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._candidate.clear()
        self._candidate_samples = 0
        self._speaking = False
        self._silence.clear()
        self._silence_samples = 0

    def _push_idle(self, frame: np.ndarray, speech: bool) -> list[VadUpdate]:
        if speech:
            self._candidate.append(frame)
            self._candidate_samples += frame.size
            if self._duration_ms(self._candidate_samples) >= self.config.min_speech_ms:
                start = _join((*self._pre_roll, *self._candidate))
                self._pre_roll.clear()
                self._pre_roll_samples = 0
                self._candidate.clear()
                self._candidate_samples = 0
                self._speaking = True
                return [VadUpdate(VadEvent.SPEECH_STARTED, start)]
            return []

        for candidate_frame in self._candidate:
            self._remember(candidate_frame)
        self._candidate.clear()
        self._candidate_samples = 0
        self._remember(frame)
        return []

    def _flush_silence_as_continuing(self) -> list[VadUpdate]:
        if not self._silence:
            return []
        samples = _join(self._silence)
        self._silence.clear()
        self._silence_samples = 0
        return [VadUpdate(VadEvent.SPEECH_CONTINUING, samples)]

    def _remember(self, frame: np.ndarray) -> None:
        limit = self._samples_for_ms(self.config.pre_roll_ms)
        if limit <= 0:
            return
        self._pre_roll.append(frame)
        self._pre_roll_samples += frame.size
        while self._pre_roll and self._pre_roll_samples > limit:
            removed = self._pre_roll.popleft()
            self._pre_roll_samples -= removed.size

    def _samples_for_ms(self, duration_ms: int) -> int:
        return int(self.sample_rate * duration_ms / 1000)

    def _duration_ms(self, sample_count: int) -> float:
        return sample_count * 1000 / self.sample_rate


def _join(frames: object) -> np.ndarray:
    values = list(frames)  # type: ignore[arg-type]
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def _take_prefix(frames: list[np.ndarray], sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.empty(0, dtype=np.float32)
    return _join(frames)[:sample_count]
