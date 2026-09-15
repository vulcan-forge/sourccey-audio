"""Replay robot recordings through the listening path without actuating anything."""

from __future__ import annotations

import json
import time
import wave
from dataclasses import asdict
from pathlib import Path

from .adapters import NullRobotAdapter
from .commands import CommandRegistry
from .config import VoiceConfig
from .conversation import ConversationHistory
from .runtime import VoiceRuntime
from .stt import MoonshineRecognizer
from .tts import SilentTextToSpeech
from .types import ConversationReply
from .vad import SileroProbability, VoiceActivityDetector
from .wake import WakeSession


class _NoPlayback:
    is_playing = False

    def play(self, audio):
        pass

    def interrupt(self):
        pass


class _NoGeneration:
    def generate(self, messages, robot_state, available_actions):
        return ConversationReply("Replay only.")


def replay_wav(path: Path, config: VoiceConfig) -> dict:
    # Validate WAV before loading either model. Never guess PCM format/rate.
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (
            1, 2, config.audio.sample_rate, "NONE"
        ):
            raise ValueError(f"replay requires mono PCM16 WAV at {config.audio.sample_rate} Hz")
        frame_count = source.getnframes()
    if config.audio.chunk_ms <= 0 or config.audio.sample_rate * config.audio.chunk_ms < 1000:
        raise ValueError("replay requires a positive audio chunk duration")
    recognizer = MoonshineRecognizer(config.stt.model_path, config.stt.architecture,
                                     config.stt.partial_interval_seconds)
    try:
        probability = SileroProbability()
        audio_time = [0.0]
        registry = CommandRegistry(config.commands)
        wake = WakeSession(
            config.wake.enabled, config.wake.phrases, config.wake.engaged_seconds,
            config.wake.emergency_bypass, registry, clock=lambda: audio_time[0],
            aliases=config.wake.aliases, contextual_aliases=config.wake.contextual_aliases,
            fuzzy_threshold=config.wake.fuzzy_threshold,
            continuation_seconds=config.wake.continuation_seconds,
            accept_so_prefix=config.wake.accept_so_prefix,
        )
        runtime = VoiceRuntime(
            registry=registry, robot=NullRobotAdapter(), conversation=_NoGeneration(),
            tts=SilentTextToSpeech(), speaker=_NoPlayback(),
            history=ConversationHistory(1, 1000, 60), wake=wake,
            output_sample_rate=config.audio.sample_rate, recognizer=recognizer,
            probability=probability, vad=VoiceActivityDetector(config.vad, config.audio.sample_rate),
            allow_tool_requests=False, latency_metrics=False,
        )
        chunk = config.audio.sample_rate * config.audio.chunk_ms // 1000
        results = []

        def feed(pcm: bytes) -> None:
            audio_time[0] += len(pcm) / 2 / config.audio.sample_rate
            results.extend(asdict(result) for result in runtime.push_pcm16(pcm, config.audio.sample_rate))

        started = time.perf_counter()
        with wave.open(str(path), "rb") as source:
            while pcm := source.readframes(chunk):
                feed(pcm)
        # Make the explicit file-end boundary visible to the segmenter.
        for _ in range(config.vad.silence_timeout_ms // config.audio.chunk_ms + 2):
            feed(b"\x00\x00" * chunk)
        report = {
            "file": str(path), "audio_seconds": frame_count / config.audio.sample_rate,
            "processing_seconds": round(time.perf_counter() - started, 3),
            "eof_silence_added": True, "results": results,
        }
        print(json.dumps(report, indent=2))
        return report
    finally:
        recognizer.close()
