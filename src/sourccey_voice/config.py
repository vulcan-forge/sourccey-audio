from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 20
    input_device: str = ""
    output_device: str = ""
    volume: float = 0.85
    aec_enabled: bool = True
    aec_required: bool = True
    aec_stream_delay_ms: int = 40
    noise_suppression: bool = True
    automatic_gain_control: bool = True


@dataclass(frozen=True)
class NetworkConfig:
    host: str = "0.0.0.0"
    port: int = 8766
    robot_url: str = "ws://127.0.0.1:8766"
    auth_token: str = "change-me"
    max_message_bytes: int = 131072
    stale_after_ms: int = 2000
    reconnect_seconds: float = 2.0


@dataclass(frozen=True)
class VadConfig:
    backend: str = "silero"
    threshold: float = 0.55
    min_speech_ms: int = 160
    silence_timeout_ms: int = 650
    pre_roll_ms: int = 240
    post_roll_ms: int = 120


@dataclass(frozen=True)
class SttConfig:
    backend: str = "moonshine"
    language: str = "en"
    model: str = "UsefulSensors/moonshine-streaming-medium"
    model_path: str = ""
    architecture: str = "medium"
    device: str = "auto"
    partial_interval_seconds: float = 0.35


@dataclass(frozen=True)
class LlmConfig:
    backend: str = "llama_cpp"
    model: str = "Qwen/Qwen3-4B-GGUF"
    model_file: str = "Qwen3-4B-Q4_K_M.gguf"
    model_path: str = ""
    device: str = "auto"
    context_tokens: int = 4096
    max_response_tokens: int = 128
    temperature: float = 0.35
    allow_tool_requests: bool = True
    structured_response: bool = True


@dataclass(frozen=True)
class TtsConfig:
    backend: str = "kokoro"
    language_code: str = "a"
    voice: str = "af_heart"
    device: str = "auto"
    speed: float = 1.08
    robotic_processing: bool = False
    pitch_semitones: float = 0.0


@dataclass(frozen=True)
class WakeConfig:
    enabled: bool = False
    phrases: tuple[str, ...] = ("sourccey", "hey sourccey")
    engaged_seconds: float = 20.0
    emergency_bypass: bool = True


@dataclass(frozen=True)
class ConversationConfig:
    max_turns: int = 6
    max_characters: int = 6000
    inactivity_seconds: float = 120.0
    personality: str = "You are Sourccey, a concise robot built by Vulcan Robotics."


@dataclass(frozen=True)
class RobotConfig:
    backend: str = "none"
    remote_ip: str = "127.0.0.1"
    fresh_state_timeout_seconds: float = 1.0


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    debug_partials: bool = False
    latency_metrics: bool = True


@dataclass(frozen=True)
class VoiceConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    commands: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_path: Path | None = field(default=None, compare=False)

    def validate(self) -> None:
        if self.audio.sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("audio.sample_rate must be 8000, 16000, 32000, or 48000")
        if self.audio.channels != 1:
            raise ValueError("only mono audio is currently supported")
        if self.audio.chunk_ms <= 0 or 1000 % self.audio.chunk_ms:
            raise ValueError("audio.chunk_ms must divide evenly into one second")
        if not 0.0 <= self.audio.volume <= 1.0:
            raise ValueError("audio.volume must be between 0 and 1")
        if not 0.0 < self.vad.threshold < 1.0:
            raise ValueError("vad.threshold must be between 0 and 1")
        if self.vad.min_speech_ms <= 0 or self.vad.silence_timeout_ms <= 0:
            raise ValueError("VAD durations must be positive")
        if self.vad.pre_roll_ms < 0 or self.vad.post_roll_ms < 0:
            raise ValueError("VAD roll durations cannot be negative")
        if self.conversation.max_turns < 1 or self.conversation.max_characters < 128:
            raise ValueError("conversation limits are too small")
        if not 1 <= self.network.port <= 65535:
            raise ValueError("network.port is invalid")
        if self.network.stale_after_ms <= 0 or self.network.max_message_bytes < 1024:
            raise ValueError("network safety limits are invalid")
        if not self.network.auth_token or self.network.auth_token == "change-me":
            raise ValueError("network.auth_token must be changed from the default")


_SECTION_TYPES = {
    "audio": AudioConfig,
    "network": NetworkConfig,
    "vad": VadConfig,
    "stt": SttConfig,
    "llm": LlmConfig,
    "tts": TtsConfig,
    "wake": WakeConfig,
    "conversation": ConversationConfig,
    "robot": RobotConfig,
    "logging": LoggingConfig,
}


def default_config_path() -> Path:
    packaged = Path(__file__).resolve().with_name("data") / "default.toml"
    repository = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    return packaged if packaged.is_file() else repository


def load_config(path: str | Path | None = None, *, validate: bool = True) -> VoiceConfig:
    selected = Path(path or os.environ.get("SOURCCEY_VOICE_CONFIG") or default_config_path())
    with selected.open("rb") as stream:
        raw = tomllib.load(stream)
    return config_from_mapping(raw, source_path=selected, validate=validate)


def config_from_mapping(
    raw: dict[str, Any], *, source_path: Path | None = None, validate: bool = True
) -> VoiceConfig:
    data = copy.deepcopy(raw)
    unknown = set(data) - {*_SECTION_TYPES, "commands"}
    if unknown:
        raise ValueError(f"unknown configuration sections: {', '.join(sorted(unknown))}")

    sections: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"configuration section {name!r} must be a table")
        allowed = set(section_type.__dataclass_fields__)
        extras = set(values) - allowed
        if extras:
            raise ValueError(f"unknown {name} settings: {', '.join(sorted(extras))}")
        if name == "wake" and "phrases" in values:
            values["phrases"] = tuple(values["phrases"])
        sections[name] = section_type(**values)

    commands_raw = data.get("commands", {})
    if not isinstance(commands_raw, dict):
        raise ValueError("commands must be a TOML table")
    commands = {str(key): tuple(map(str, aliases)) for key, aliases in commands_raw.items()}
    config = VoiceConfig(**sections, commands=commands, source_path=source_path)
    if validate:
        config.validate()
    return config
