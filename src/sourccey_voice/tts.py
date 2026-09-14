from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .types import SpeechAudio


class ExternalTextToSpeech:
    """Load an optional desktop TTS runtime without coupling this module to it."""

    def __init__(self, runtime_path: str, factory: str, preset_path: str = "") -> None:
        location = runtime_path or os.environ.get("SOURCCEY_VOICE_RUNTIME_PATH", "")
        if not location:
            raise RuntimeError(
                "No desktop TTS runtime is configured. Set tts.runtime_path or "
                "SOURCCEY_VOICE_RUNTIME_PATH."
            )
        directory = Path(location).expanduser().resolve()
        if not directory.is_dir():
            raise RuntimeError(f"Desktop TTS runtime directory was not found: {directory}")
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        try:
            module_name, callable_name = factory.split(":", 1)
            create_tts = getattr(importlib.import_module(module_name), callable_name)
        except (ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError(f"Could not load desktop TTS factory {factory!r}") from exc
        selected_preset = preset_path or os.environ.get("SOURCCEY_VOICE_PRESET", "")
        self._delegate: Any = create_tts(preset_path=selected_preset)

    def synthesize(self, text: str) -> SpeechAudio:
        return self._delegate.synthesize(text)


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
