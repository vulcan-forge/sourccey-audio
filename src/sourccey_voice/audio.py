from __future__ import annotations

import logging

import numpy as np

from .config import AudioConfig

logger = logging.getLogger(__name__)


class EchoProcessor:
    def __init__(self, config: AudioConfig) -> None:
        self.active = False
        self._processor = None
        if not config.aec_enabled:
            return
        try:
            from pywebrtc_audio import AudioProcessor

            self._processor = AudioProcessor(
                sample_rate=config.sample_rate,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=config.noise_suppression,
                auto_gain_control=config.automatic_gain_control,
                stream_delay_ms=config.aec_stream_delay_ms,
            )
            self.active = True
        except (ImportError, RuntimeError) as exc:
            if config.aec_required:
                raise RuntimeError(
                    "WebRTC AEC is required but unavailable; install sourccey-voice[audio]"
                ) from exc
            logger.warning("[AEC] unavailable; microphone will be gated during playback: %s", exc)

    def process(self, near_pcm16: bytes, far_pcm16: bytes) -> bytes:
        if not self.active or self._processor is None:
            return near_pcm16
        near = np.frombuffer(near_pcm16, dtype="<i2")
        far = np.frombuffer(far_pcm16, dtype="<i2")
        if far.size < near.size:
            far = np.pad(far, (0, near.size - far.size))
        elif far.size > near.size:
            far = far[: near.size]
        clean = self._processor.process(near, far)
        return np.asarray(clean, dtype="<i2").tobytes()

    def reset(self) -> None:
        if self._processor is not None:
            self._processor.reset()

