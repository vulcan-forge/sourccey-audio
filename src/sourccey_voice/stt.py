from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class MoonshineRecognizer:
    """Streaming Moonshine adapter for audio supplied by the robot transport."""

    def __init__(
        self,
        model_path: str,
        architecture: str = "medium_streaming",
        update_interval: float = 0.35,
    ) -> None:
        if not model_path:
            raise RuntimeError(
                "stt.model_path is empty; run `sourccey-voice models download stt` "
                "and set the downloaded directory in the config"
            )
        try:
            from moonshine_voice import ModelArch, Transcriber
        except ImportError as exc:
            raise RuntimeError("Moonshine is unavailable; install sourccey-voice[stt]") from exc

        try:
            arch = getattr(ModelArch, architecture.upper())
        except AttributeError as exc:
            raise ValueError(f"unsupported Moonshine architecture: {architecture}") from exc
        self._transcriber = Transcriber(
            model_path=Path(model_path), model_arch=arch, update_interval=update_interval
        )
        self._stream: Any = None
        self._sample_rate = 16000

    def start(self) -> None:
        if self._stream is not None:
            self.stop()
        self._stream = self._transcriber.create_stream()
        self._stream.start()

    def push_audio(self, samples: Sequence[float], sample_rate: int) -> None:
        if self._stream is None:
            raise RuntimeError("recognizer has not been started")
        self._sample_rate = sample_rate
        self._stream.add_audio(list(samples), sample_rate)

    def partial_transcript(self) -> str:
        if self._stream is None:
            return ""
        return _transcript_text(self._stream.update_transcription())

    def finalize(self) -> str:
        if self._stream is None:
            return ""
        stream, self._stream = self._stream, None
        try:
            transcript = stream.stop()
            if transcript is None:
                raise RuntimeError("Moonshine failed to finalize this utterance")
            return _transcript_text(transcript)
        finally:
            stream.close()

    def stop(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            # Discarded audio doesn't need one more expensive transcription.
            stream.close()

    def close(self) -> None:
        self.stop()
        self._transcriber.close()


def _transcript_text(transcript: object) -> str:
    lines = getattr(transcript, "lines", ()) or ()
    return " ".join(
        str(getattr(line, "text", "")).strip()
        for line in lines
        if str(getattr(line, "text", "")).strip()
    ).strip()
