from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Any

import numpy as np

from .audio import EchoProcessor
from .network import MessageType, VoiceMessage, audio_payload, decode_audio

logger = logging.getLogger(__name__)


class RobotAudioAgent:
    def __init__(self, audio_config: Any, network_config: Any) -> None:
        self.audio_config = audio_config
        self.network_config = network_config
        self.echo = EchoProcessor(audio_config)
        self._playback = bytearray()
        self._last_far = b""
        self._lock = threading.Lock()
        self._sequence = 0
        self._input_queue: asyncio.Queue[bytes] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        try:
            import sounddevice as sd
            import websockets
        except ImportError as exc:
            raise RuntimeError("robot audio requires sourccey-voice[audio]") from exc
        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue(maxsize=100)
        blocksize = self.audio_config.sample_rate * self.audio_config.chunk_ms // 1000
        input_device = self.audio_config.input_device or None
        output_device = self.audio_config.output_device or None
        with sd.RawInputStream(
            samplerate=self.audio_config.sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="int16",
            device=input_device,
            callback=self._on_input,
        ), sd.RawOutputStream(
            samplerate=self.audio_config.sample_rate,
            blocksize=blocksize,
            channels=1,
            dtype="int16",
            device=output_device,
            callback=self._on_output,
        ):
            while True:
                try:
                    async with websockets.connect(
                        self.network_config.robot_url,
                        max_size=self.network_config.max_message_bytes,
                        ping_interval=20,
                        ping_timeout=20,
                    ) as websocket:
                        await websocket.send(self._message(MessageType.HELLO, {"token": self.network_config.auth_token}))
                        logger.info("[NETWORK] connected to voice host")
                        await asyncio.gather(self._send_audio(websocket), self._receive(websocket))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[NETWORK] voice host unavailable: %s", exc)
                    self._clear_playback()
                    await asyncio.sleep(self.network_config.reconnect_seconds)

    def _on_input(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            logger.debug("[AUDIO] input status: %s", status)
        near = bytes(indata)
        with self._lock:
            far = self._last_far
            playing = bool(self._playback) or any(far)
        if playing and self.audio_config.gate_during_playback:
            return
        if not self.echo.active and playing:
            return
        clean = self.echo.process(near, far)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue_input, clean)

    def _on_output(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        del time_info
        if status:
            logger.debug("[AUDIO] output status: %s", status)
        byte_count = frames * 2
        with self._lock:
            available = min(byte_count, len(self._playback))
            chunk = bytes(self._playback[:available])
            del self._playback[:available]
            if available < byte_count:
                chunk += b"\x00" * (byte_count - available)
            self._last_far = chunk
        outdata[:] = chunk

    def _queue_input(self, pcm16: bytes) -> None:
        assert self._input_queue is not None
        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._input_queue.put_nowait(pcm16)

    async def _send_audio(self, websocket: Any) -> None:
        assert self._input_queue is not None
        while True:
            pcm16 = await self._input_queue.get()
            await websocket.send(
                self._message(
                    MessageType.AUDIO_INPUT,
                    audio_payload(pcm16, self.audio_config.sample_rate),
                )
            )

    async def _receive(self, websocket: Any) -> None:
        async for raw in websocket:
            message = VoiceMessage.from_json(raw, max_bytes=self.network_config.max_message_bytes)
            if message.type == MessageType.AUDIO_OUTPUT:
                pcm16, sample_rate = decode_audio(message)
                if sample_rate != self.audio_config.sample_rate:
                    pcm16 = self._resample_pcm16(pcm16, sample_rate, self.audio_config.sample_rate)
                    sample_rate = self.audio_config.sample_rate
                volume = float(np.clip(self.audio_config.volume, 0.0, 1.0))
                samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) * volume
                pcm16 = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
                with self._lock:
                    self._playback.extend(pcm16)
            elif message.type == MessageType.AUDIO_INTERRUPT:
                self._clear_playback()
                self.echo.reset()
                logger.info("[AUDIO] playback interrupted")
            elif message.type == MessageType.ERROR:
                logger.warning("[NETWORK] host rejected a message: %s", message.payload.get("error"))

    def _clear_playback(self) -> None:
        with self._lock:
            self._playback.clear()
            self._last_far = b""

    @staticmethod
    def _resample_pcm16(pcm16: bytes, source_rate: int, target_rate: int) -> bytes:
        if source_rate == target_rate or not pcm16:
            return pcm16
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
        if samples.size < 2:
            return pcm16
        target_size = max(1, round(samples.size * target_rate / source_rate))
        positions = np.linspace(0, samples.size - 1, target_size)
        converted = np.interp(positions, np.arange(samples.size), samples)
        return np.clip(converted, -32768, 32767).astype("<i2").tobytes()

    def _message(self, message_type: MessageType, payload: dict[str, object]) -> str:
        self._sequence += 1
        return VoiceMessage.create(message_type, self._sequence, payload).to_json()
