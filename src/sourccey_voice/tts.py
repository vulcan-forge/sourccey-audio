from __future__ import annotations

from typing import Any

import numpy as np

from .types import SpeechAudio


class KokoroTextToSpeech:
    def __init__(self, language_code: str, voice: str, speed: float = 1.0) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("Kokoro is unavailable; install sourccey-voice[tts]") from exc
        self._pipeline: Any = KPipeline(lang_code=language_code)
        self._voice = voice
        self._speed = speed

    def synthesize(self, text: str) -> SpeechAudio:
        chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        samples = np.clip(np.concatenate(chunks), -1.0, 1.0)
        return SpeechAudio((samples * 32767.0).astype("<i2").tobytes(), 24000)


class SilentTextToSpeech:
    def synthesize(self, text: str) -> SpeechAudio:
        del text
        return SpeechAudio(b"", 16000)


def resample_pcm16(audio: SpeechAudio, target_rate: int) -> SpeechAudio:
    if audio.sample_rate == target_rate or not audio.pcm16:
        return SpeechAudio(audio.pcm16, target_rate)
    source = np.frombuffer(audio.pcm16, dtype="<i2").astype(np.float32)
    target_count = max(1, round(source.size * target_rate / audio.sample_rate))
    source_positions = np.arange(source.size, dtype=np.float64)
    target_positions = np.linspace(0, max(0, source.size - 1), target_count)
    result = np.interp(target_positions, source_positions, source)
    return SpeechAudio(np.clip(result, -32768, 32767).astype("<i2").tobytes(), target_rate)

