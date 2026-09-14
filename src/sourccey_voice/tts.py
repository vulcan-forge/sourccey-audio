from __future__ import annotations

from typing import Any

import numpy as np

from .types import SpeechAudio


class KokoroTextToSpeech:
    def __init__(
        self,
        language_code: str,
        voice: str,
        speed: float = 1.0,
        robotic_processing: bool = False,
    ) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("Kokoro is unavailable; install sourccey-voice[tts]") from exc
        self._pipeline: Any = KPipeline(lang_code=language_code)
        self._voice = voice
        self._speed = speed
        self._robotic_processing = robotic_processing

    def synthesize(self, text: str) -> SpeechAudio:
        chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in self._pipeline(
            text, voice=self._voice, speed=self._speed
        ):
            chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        samples = np.clip(np.concatenate(chunks), -1.0, 1.0)
        if self._robotic_processing:
            samples = _apply_robotic_treatment(samples, sample_rate=24000)
        return SpeechAudio((samples * 32767.0).astype("<i2").tobytes(), 24000)


def _apply_robotic_treatment(samples: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """Add a restrained digital character while preserving intelligibility."""
    if samples.size == 0:
        return samples
    positions = np.arange(samples.size, dtype=np.float32) / sample_rate
    carrier = np.sin(2.0 * np.pi * 105.0 * positions)
    delay_samples = min(78, samples.size)
    delayed = np.concatenate((np.zeros(delay_samples, dtype=np.float32), samples[:-delay_samples]))
    treated = 0.72 * samples + 0.18 * delayed + 0.10 * samples * carrier
    return np.clip(0.78 * samples + 0.22 * treated, -1.0, 1.0)


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
