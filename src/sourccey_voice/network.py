from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from .types import Speaker, SpeechAudio

logger = logging.getLogger(__name__)


class MessageType(StrEnum):
    HELLO = "hello.v1"
    HEARTBEAT = "heartbeat.v1"
    AUDIO_INPUT = "audio.input.v1"
    AUDIO_OUTPUT = "audio.output.v1"
    AUDIO_INTERRUPT = "audio.interrupt.v1"
    VOICE_EVENT = "voice.event.v1"
    ERROR = "error.v1"


@dataclass(frozen=True)
class VoiceMessage:
    type: MessageType
    sequence: int
    created_at_ms: int
    payload: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {
                "type": self.type.value,
                "sequence": self.sequence,
                "created_at_ms": self.created_at_ms,
                "payload": self.payload,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls,
        raw: str | bytes,
        *,
        max_bytes: int = 131072,
        stale_after_ms: int | None = None,
        now_ms: int | None = None,
    ) -> "VoiceMessage":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if len(raw.encode("utf-8")) > max_bytes:
            raise ValueError("message exceeds configured size limit")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("message is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("message must be an object")
        try:
            message_type = MessageType(value["type"])
            sequence = value["sequence"]
            created_at_ms = value["created_at_ms"]
            payload = value["payload"]
        except (KeyError, ValueError) as exc:
            raise ValueError("message type or required fields are invalid") from exc
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(created_at_ms, int) or created_at_ms < 0:
            raise ValueError("created_at_ms must be a non-negative integer")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if stale_after_ms is not None and message_type == MessageType.AUDIO_INPUT:
            age = (now_ms if now_ms is not None else int(time.time() * 1000)) - created_at_ms
            if age > stale_after_ms:
                raise ValueError("stale audio message")
            if age < -stale_after_ms:
                raise ValueError("audio message timestamp is too far in the future")
        message = cls(message_type, sequence, created_at_ms, payload)
        message.validate_payload()
        return message

    def validate_payload(self) -> None:
        if self.type in {MessageType.AUDIO_INPUT, MessageType.AUDIO_OUTPUT}:
            encoded = self.payload.get("pcm16_b64")
            sample_rate = self.payload.get("sample_rate")
            if not isinstance(encoded, str) or not isinstance(sample_rate, int):
                raise ValueError("audio message requires pcm16_b64 and integer sample_rate")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError("audio payload is not valid base64") from exc
            if len(decoded) % 2:
                raise ValueError("audio payload is not complete PCM16")
        elif self.type == MessageType.HELLO:
            if not isinstance(self.payload.get("token"), str):
                raise ValueError("hello message requires a token")

    @classmethod
    def create(
        cls, message_type: MessageType, sequence: int, payload: dict[str, object]
    ) -> "VoiceMessage":
        return cls(message_type, sequence, int(time.time() * 1000), payload)


def audio_payload(pcm16: bytes, sample_rate: int) -> dict[str, object]:
    return {
        "pcm16_b64": base64.b64encode(pcm16).decode("ascii"),
        "sample_rate": sample_rate,
    }


def decode_audio(message: VoiceMessage) -> tuple[bytes, int]:
    if message.type not in {MessageType.AUDIO_INPUT, MessageType.AUDIO_OUTPUT}:
        raise ValueError("message does not contain audio")
    return (
        base64.b64decode(str(message.payload["pcm16_b64"]), validate=True),
        int(message.payload["sample_rate"]),
    )


class WebSocketSpeaker(Speaker):
    def __init__(self) -> None:
        self._send: Callable[[str], Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._playing_until = 0.0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._send is not None and time.monotonic() < self._playing_until

    def attach(self, send: Callable[[str], Any], loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._send = send
            self._loop = loop
            self._playing_until = 0.0

    def detach(self) -> None:
        with self._lock:
            self._send = None
            self._loop = None
            self._playing_until = 0.0

    def play(self, audio: SpeechAudio) -> None:
        if not audio.pcm16:
            return
        duration = len(audio.pcm16) / 2 / audio.sample_rate
        # Keep each JSON/base64 frame comfortably below the configured default
        # websocket limit, even when Kokoro returns a long response in one array.
        bytes_per_chunk = audio.sample_rate  # 500 ms of mono PCM16.
        for offset in range(0, len(audio.pcm16), bytes_per_chunk):
            chunk = audio.pcm16[offset : offset + bytes_per_chunk]
            self._send_message(MessageType.AUDIO_OUTPUT, audio_payload(chunk, audio.sample_rate))
        with self._lock:
            self._playing_until = max(self._playing_until, time.monotonic()) + duration

    def interrupt(self) -> None:
        self._send_message(MessageType.AUDIO_INTERRUPT, {})
        with self._lock:
            self._playing_until = 0.0

    def _send_message(self, message_type: MessageType, payload: dict[str, object]) -> None:
        with self._lock:
            send, loop = self._send, self._loop
            self._sequence += 1
            sequence = self._sequence
        if send is None or loop is None:
            raise RuntimeError("robot audio endpoint is not connected")
        message = VoiceMessage.create(message_type, sequence, payload).to_json()
        future = asyncio.run_coroutine_threadsafe(send(message), loop)
        future.result(timeout=5.0)


class VoiceHostServer:
    def __init__(self, config: Any, runtime: Any, speaker: WebSocketSpeaker) -> None:
        self.config = config
        self.runtime = runtime
        self.speaker = speaker
        self._last_sequence = -1

    async def run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets is required to run the voice host") from exc
        logger.info("[NETWORK] voice host listening on %s:%d", self.config.host, self.config.port)
        async with websockets.serve(
            self._handle,
            self.config.host,
            self.config.port,
            max_size=self.config.max_message_bytes,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()

    async def _handle(self, websocket: Any) -> None:
        try:
            raw_hello = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            hello = VoiceMessage.from_json(raw_hello, max_bytes=self.config.max_message_bytes)
            if hello.type != MessageType.HELLO or not hmac.compare_digest(
                str(hello.payload["token"]), self.config.auth_token
            ):
                await websocket.close(code=4001, reason="authentication failed")
                return
            # The robot and desktop use independent wall clocks. Anchor audio freshness
            # to their authenticated connection rather than assuming NTP-level alignment.
            clock_offset_ms = int(time.time() * 1000) - hello.created_at_ms
            self.speaker.attach(websocket.send, asyncio.get_running_loop())
            self._last_sequence = hello.sequence
            logger.info("[NETWORK] robot audio endpoint connected")
            async for raw in websocket:
                try:
                    message = VoiceMessage.from_json(
                        raw,
                        max_bytes=self.config.max_message_bytes,
                    )
                    if message.sequence <= self._last_sequence:
                        raise ValueError("duplicate or out-of-order message")
                    self._last_sequence = message.sequence
                    if message.type == MessageType.AUDIO_INPUT:
                        received_age_ms = (
                            int(time.time() * 1000) - (message.created_at_ms + clock_offset_ms)
                        )
                        if received_age_ms > self.config.stale_after_ms:
                            raise ValueError("stale audio message")
                        if received_age_ms < -self.config.stale_after_ms:
                            raise ValueError("audio message timestamp is too far in the future")
                        pcm16, sample_rate = decode_audio(message)
                        results = await asyncio.to_thread(self.runtime.push_pcm16, pcm16, sample_rate)
                        for result in results:
                            event = VoiceMessage.create(
                                MessageType.VOICE_EVENT,
                                self._last_sequence + 1,
                                {
                                    "kind": result.kind,
                                    "transcript": result.transcript,
                                    "response": result.response,
                                    "command": result.command,
                                    "accepted": result.accepted,
                                    "reason": result.reason,
                                },
                            )
                            await websocket.send(event.to_json())
                    elif message.type != MessageType.HEARTBEAT:
                        raise ValueError(f"unexpected robot message type: {message.type.value}")
                except ValueError as exc:
                    logger.warning("[NETWORK] rejected message: %s", exc)
                    error = VoiceMessage.create(MessageType.ERROR, self._last_sequence + 1, {"error": str(exc)})
                    await websocket.send(error.to_json())
        except Exception as exc:
            logger.info("[NETWORK] robot audio endpoint disconnected: %s", exc)
        finally:
            self.speaker.detach()
