from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from .commands import CommandName, CommandRegistry, CommandResult
from .conversation import ConversationHistory
from .types import ConversationEngine, RobotCommandAdapter, Speaker, SpeechRecognizer, TextToSpeech
from .tts import resample_pcm16
from .vad import SpeechProbability, VadEvent, VoiceActivityDetector
from .wake import WakeSession

logger = logging.getLogger(__name__)

_SOURCCEY_TRANSCRIPT_ALIASES = re.compile(r"\b(?:sourcing|sourcey|sourcy|sorsi)\b", re.IGNORECASE)


def normalize_transcript(text: str) -> str:
    """Normalize the recurring STT spellings of Sourccey's name before routing."""
    return _SOURCCEY_TRANSCRIPT_ALIASES.sub("Sourccey", text)


@dataclass(frozen=True)
class InteractionResult:
    kind: str
    transcript: str
    response: str = ""
    command: str | None = None
    accepted: bool | None = None
    reason: str = ""


class VoiceRuntime:
    def __init__(
        self,
        *,
        registry: CommandRegistry,
        robot: RobotCommandAdapter,
        conversation: ConversationEngine,
        tts: TextToSpeech,
        speaker: Speaker,
        history: ConversationHistory,
        wake: WakeSession,
        output_sample_rate: int,
        recognizer: SpeechRecognizer | None = None,
        probability: SpeechProbability | None = None,
        vad: VoiceActivityDetector | None = None,
        debug_partials: bool = False,
        allow_tool_requests: bool = True,
        latency_metrics: bool = True,
    ) -> None:
        self.registry = registry
        self.robot = robot
        self.conversation = conversation
        self.tts = tts
        self.speaker = speaker
        self.history = history
        self.wake = wake
        self.output_sample_rate = output_sample_rate
        self.recognizer = recognizer
        self.probability = probability
        self.vad = vad
        self.debug_partials = debug_partials
        self.allow_tool_requests = allow_tool_requests
        self.latency_metrics = latency_metrics
        self._lock = threading.RLock()
        self._last_audio_diagnostic_at = 0.0

    def push_pcm16(self, pcm16: bytes, sample_rate: int) -> list[InteractionResult]:
        if self.recognizer is None or self.probability is None or self.vad is None:
            raise RuntimeError("audio input requires recognizer, VAD probability, and VAD state machine")
        if len(pcm16) % 2:
            raise ValueError("PCM16 payload must contain complete samples")
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            probability = self.probability.probability(samples, sample_rate)
            now = time.monotonic()
            if self.debug_partials and now - self._last_audio_diagnostic_at >= 1.0:
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
                logger.info(
                    "[AUDIO] input_rms=%.4f vad_probability=%.3f threshold=%.3f",
                    rms,
                    probability,
                    self.vad.config.threshold,
                )
                self._last_audio_diagnostic_at = now
            results: list[InteractionResult] = []
            for update in self.vad.push(samples, probability):
                log = logger.debug if update.event == VadEvent.SPEECH_CONTINUING else logger.info
                log("[VAD] %s", update.event.value)
                if update.event == VadEvent.SPEECH_STARTED:
                    self.recognizer.start()
                    self.recognizer.push_audio(update.samples, sample_rate)
                    if self.speaker.is_playing:
                        logger.info("[VAD] speech_started during playback")
                elif update.event == VadEvent.SPEECH_CONTINUING:
                    self.recognizer.push_audio(update.samples, sample_rate)
                    if self.debug_partials:
                        partial = self.recognizer.partial_transcript()
                        if partial:
                            logger.debug('[STT partial] "%s"', partial)
                elif update.event == VadEvent.SPEECH_ENDED:
                    if update.samples.size:
                        self.recognizer.push_audio(update.samples, sample_rate)
                    started = time.perf_counter()
                    transcript = normalize_transcript(self.recognizer.finalize().strip())
                    logger.info('[STT] "%s"', transcript)
                    if self.latency_metrics:
                        logger.info(
                            "[LATENCY] speech_end_to_final_stt_ms=%.1f",
                            (time.perf_counter() - started) * 1000,
                        )
                    if transcript:
                        results.append(self.handle_transcript(transcript))
            return results

    def handle_transcript(self, transcript: str) -> InteractionResult:
        with self._lock:
            decision = self.wake.evaluate(transcript)
            if not decision.accepted:
                return InteractionResult("ignored", transcript, reason=decision.reason)
            text = decision.text
            explicit = self.registry.parse_explicit(text)
            if explicit is not None:
                logger.info("[ROUTER] explicit command detected")
                return self._execute_command(text, explicit.name, interrupt=explicit.priority)
            if self.registry.uses_explicit_syntax(text):
                response = "I don't recognize that command."
                self._speak(response)
                return InteractionResult("command", text, response, accepted=False, reason="unknown_command")

            messages = [*self.history.messages(), {"role": "user", "content": text}]
            available = sorted(self.robot.available_actions()) if self.allow_tool_requests else []
            state = self.robot.state()
            try:
                reply = self.conversation.generate(messages, state, available)
            except Exception as exc:
                logger.exception("[LLM] generation failed")
                response = "My conversation system is unavailable right now."
                self._speak(response)
                return InteractionResult("conversation", text, response, reason=str(exc))

            if reply.requested_action is not None:
                if not self.allow_tool_requests:
                    logger.warning("[COMMAND] rejected LLM action because tool requests are disabled")
                    response = "Conversational actions are disabled."
                    self.history.add(text, response)
                    self._speak(response)
                    return InteractionResult(
                        "command", text, response, accepted=False, reason="tool_requests_disabled"
                    )
                try:
                    requested = self.registry.validate_action_name(reply.requested_action)
                except ValueError:
                    logger.warning("[COMMAND] rejected unapproved LLM action: %r", reply.requested_action)
                    response = "I can't perform that action."
                    self.history.add(text, response)
                    self._speak(response)
                    return InteractionResult(
                        "command", text, response, accepted=False, reason="unapproved_llm_action"
                    )
                result = self._execute_command(text, requested, interrupt=requested in {
                    CommandName.STOP, CommandName.CANCEL, CommandName.FREEZE
                })
                self.history.add(text, result.response)
                return result

            response = reply.text.strip() or "I'm not sure what to say."
            self.history.add(text, response)
            self._speak(response)
            return InteractionResult("conversation", text, response)

    def _execute_command(
        self, transcript: str, command: CommandName, *, interrupt: bool
    ) -> InteractionResult:
        if interrupt and self.speaker.is_playing:
            self.speaker.interrupt()
            logger.info("[TTS] interrupted")
        result: CommandResult = self.registry.execute(
            command, self.robot.available_actions(), self.robot.execute
        )
        logger.info("[COMMAND] %s %s", command.value, "accepted" if result.accepted else "rejected")
        self._speak(result.message)
        return InteractionResult(
            "command",
            transcript,
            result.message,
            command=command.value,
            accepted=result.accepted,
        )

    def _speak(self, text: str) -> None:
        try:
            started = time.perf_counter()
            audio = resample_pcm16(self.tts.synthesize(text), self.output_sample_rate)
            self.speaker.play(audio)
            logger.info('[TTS] "%s"', text)
            if self.latency_metrics:
                logger.info(
                    "[LATENCY] response_to_tts_start_ms=%.1f",
                    (time.perf_counter() - started) * 1000,
                )
        except Exception:
            logger.exception('[TTS] synthesis/playback failed for "%s"', text)
