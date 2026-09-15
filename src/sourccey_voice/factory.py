from __future__ import annotations

from .adapters import NullRobotAdapter, SourcceyClientRobotAdapter
from .commands import CommandRegistry
from .config import VoiceConfig
from .conversation import ConversationHistory
from .llm import LlamaCppConversationEngine
from .runtime import VoiceRuntime
from .stt import MoonshineRecognizer
from .tts import ExternalTextToSpeech
from .vad import SileroProbability, VoiceActivityDetector
from .wake import WakeSession


def build_runtime(config: VoiceConfig, speaker: object) -> VoiceRuntime:
    registry = CommandRegistry(config.commands)
    if config.robot.backend == "none":
        robot = NullRobotAdapter()
    elif config.robot.backend == "sourccey_client":
        robot = SourcceyClientRobotAdapter(
            config.robot.remote_ip, config.robot.fresh_state_timeout_seconds
        )
    else:
        raise ValueError(f"unsupported robot backend: {config.robot.backend}")

    if config.stt.backend != "moonshine":
        raise ValueError(f"unsupported STT backend: {config.stt.backend}")
    recognizer = MoonshineRecognizer(
        config.stt.model_path,
        config.stt.architecture,
        config.stt.partial_interval_seconds,
    )
    if config.vad.backend != "silero":
        raise ValueError(f"unsupported VAD backend: {config.vad.backend}")
    probability = SileroProbability()
    vad = VoiceActivityDetector(config.vad, config.audio.sample_rate)

    if config.llm.backend != "llama_cpp":
        raise ValueError(f"unsupported LLM backend: {config.llm.backend}")
    conversation = LlamaCppConversationEngine(
        config.llm.model_path,
        config.conversation.personality,
        context_tokens=config.llm.context_tokens,
        max_response_tokens=config.llm.max_response_tokens,
        temperature=config.llm.temperature,
        device=config.llm.device,
        structured_response=config.llm.structured_response,
    )
    if config.tts.backend == "external":
        tts = ExternalTextToSpeech(
            config.tts.runtime_path,
            config.tts.factory,
            config.tts.preset_path,
        )
    else:
        raise ValueError(f"unsupported TTS backend: {config.tts.backend}")

    history = ConversationHistory(
        config.conversation.max_turns,
        config.conversation.max_characters,
        config.conversation.inactivity_seconds,
    )
    wake = WakeSession(
        config.wake.enabled,
        config.wake.phrases,
        config.wake.engaged_seconds,
        config.wake.emergency_bypass,
        registry,
        aliases=config.wake.aliases,
        contextual_aliases=config.wake.contextual_aliases,
        fuzzy_threshold=config.wake.fuzzy_threshold,
        continuation_seconds=config.wake.continuation_seconds,
    )
    return VoiceRuntime(
        registry=registry,
        robot=robot,
        conversation=conversation,
        tts=tts,
        speaker=speaker,  # type: ignore[arg-type]
        history=history,
        wake=wake,
        output_sample_rate=config.audio.sample_rate,
        recognizer=recognizer,
        probability=probability,
        vad=vad,
        debug_partials=config.logging.debug_partials,
        allow_tool_requests=config.llm.allow_tool_requests,
        latency_metrics=config.logging.latency_metrics,
    )
