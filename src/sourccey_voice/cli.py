from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .adapters import DeveloperRobotAdapter
from .commands import CommandRegistry
from .config import VoiceConfig, load_config
from .conversation import ConversationHistory
from .factory import build_runtime
from .llm import DeveloperConversationEngine
from .models import download_model
from .network import VoiceHostServer, WebSocketSpeaker
from .robot_audio import RobotAudioAgent
from .runtime import VoiceRuntime
from .tts import SilentTextToSpeech
from .types import SpeechAudio
from .wake import WakeSession


class ConsoleSpeaker:
    @property
    def is_playing(self) -> bool:
        return False

    def play(self, audio: SpeechAudio) -> None:
        del audio

    def interrupt(self) -> None:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sourccey-voice", description="Local voice runtime for Sourccey")
    parser.add_argument("--config", type=Path, help="TOML configuration path")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("host", help="run the client/host intelligence and audio server")
    sub.add_parser("robot-audio", help="run microphone, AEC, and speaker transport on Sourccey")
    sub.add_parser("config-check", help="validate configuration without loading models")
    replay = sub.add_parser("replay", help="check a recorded WAV through STT and wake gating, without playback or robot actions")
    replay.add_argument("wav", type=Path, help="mono PCM16 WAV at the configured audio sample rate")
    developer = sub.add_parser("dev", help="test routing without models or physical hardware")
    developer.add_argument("--once", help="process one transcript and exit")
    models = sub.add_parser("models", help="download a model into the external cache")
    models.add_argument("action", choices=("download",))
    models.add_argument("kind", choices=("stt", "llm", "tts"))
    return parser


def _load(args: argparse.Namespace, *, validate: bool) -> VoiceConfig:
    return load_config(args.config, validate=validate)


def _configure_logging(config: VoiceConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _developer_runtime(config: VoiceConfig) -> VoiceRuntime:
    registry = CommandRegistry(config.commands)
    return VoiceRuntime(
        registry=registry,
        robot=DeveloperRobotAdapter(),
        conversation=DeveloperConversationEngine(),
        tts=SilentTextToSpeech(),
        speaker=ConsoleSpeaker(),
        history=ConversationHistory(
            config.conversation.max_turns,
            config.conversation.max_characters,
            config.conversation.inactivity_seconds,
        ),
        wake=WakeSession(False, config.wake.phrases, config.wake.engaged_seconds, True, registry),
        output_sample_rate=config.audio.sample_rate,
    )


def _run_dev(config: VoiceConfig, once: str | None) -> int:
    runtime = _developer_runtime(config)

    def process(text: str) -> None:
        result = runtime.handle_transcript(text)
        if result.kind == "command":
            print(result.response)
        else:
            print(f"Sourccey: {result.response}")

    if once is not None:
        process(once)
        return 0
    print("Sourccey Voice developer mode. Type /quit to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text in {"/quit", "/exit"}:
            return 0
        if text:
            process(text)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "config-check":
            config = _load(args, validate=True)
            print(f"Configuration is valid: {config.source_path}")
            return 0
        if args.command == "dev":
            config = _load(args, validate=False)
            _configure_logging(config)
            return _run_dev(config, args.once)
        if args.command == "models":
            config = _load(args, validate=False)
            path = download_model(args.kind, config)
            setting = "stt.model_path" if args.kind == "stt" else "llm.model_path"
            print(f"Downloaded {args.kind} model to: {path}")
            if args.kind != "tts":
                print(f"Set {setting} to that path in your config.")
            return 0
        if args.command == "replay":
            from .replay import replay_wav

            config = _load(args, validate=False)
            _configure_logging(config)
            replay_wav(args.wav, config)
            return 0

        config = _load(args, validate=True)
        _configure_logging(config)
        if args.command == "robot-audio":
            asyncio.run(RobotAudioAgent(config.audio, config.network).run())
            return 0
        if args.command == "host":
            speaker = WebSocketSpeaker()
            runtime = build_runtime(config, speaker)
            asyncio.run(VoiceHostServer(config.network, runtime, speaker).run())
            return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"sourccey-voice: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
