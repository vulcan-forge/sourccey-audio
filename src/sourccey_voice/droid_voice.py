"""Offline and streaming effects for a friendly, nonhuman droid character."""

from __future__ import annotations

import argparse
import math
import tomllib
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


_EPSILON = 1e-8
_AB_PRESETS = {
    "A": "subtle_droid",
    "B": "cute_helper_droid",
    "C": "more_synthetic",
    "D": "maximum_robot",
}


@dataclass(frozen=True)
class DroidVoiceConfig:
    """Character controls. Values are intentionally conservative by default."""

    robotization_amount: float = 0.58
    pitch_semitones: float = 0.0
    formant_shift: float = 0.22
    ring_mod_frequency_hz: float = 68.0
    ring_mod_mix: float = 0.075
    spectral_mix: float = 0.18
    saturation: float = 0.08
    compression_threshold_db: float = -12.0
    compression_ratio: float = 2.1
    speaker_low_hz: float = 150.0
    speaker_high_hz: float = 7800.0
    micro_delay_ms: float = 6.0
    micro_delay_mix: float = 0.045
    bit_depth: int = 15
    sample_hold_hz: float = 0.0
    pitch_quantization_strength: float = 0.0
    earcon: bool = False

    def validate(self) -> None:
        bounded = {
            "robotization_amount": self.robotization_amount,
            "ring_mod_mix": self.ring_mod_mix,
            "spectral_mix": self.spectral_mix,
            "saturation": self.saturation,
            "micro_delay_mix": self.micro_delay_mix,
            "pitch_quantization_strength": self.pitch_quantization_strength,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not -3.0 <= self.pitch_semitones <= 3.0:
            raise ValueError("pitch_semitones must be between -3 and 3")
        if not -2.0 <= self.formant_shift <= 2.0:
            raise ValueError("formant_shift must be between -2 and 2")
        if self.ring_mod_frequency_hz < 0.0:
            raise ValueError("ring_mod_frequency_hz cannot be negative")
        if self.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be at least 1")
        if not 2 <= self.bit_depth <= 24:
            raise ValueError("bit_depth must be between 2 and 24")


def default_preset_path() -> Path:
    return Path(__file__).resolve().with_name("data") / "droid_voice.toml"


def load_presets(path: str | Path | None = None) -> dict[str, DroidVoiceConfig]:
    with Path(path or default_preset_path()).open("rb") as stream:
        raw = tomllib.load(stream)
    sections = raw.get("presets")
    if not isinstance(sections, dict):
        raise ValueError("droid voice config needs a [presets] table")
    presets: dict[str, DroidVoiceConfig] = {}
    valid = set(DroidVoiceConfig.__dataclass_fields__)
    for name, values in sections.items():
        if not isinstance(values, dict):
            raise ValueError(f"preset {name!r} must be a TOML table")
        extras = set(values) - valid
        if extras:
            raise ValueError(f"unknown droid preset settings: {', '.join(sorted(extras))}")
        preset = DroidVoiceConfig(**values)
        preset.validate()
        presets[str(name)] = preset
    return presets


class DroidVoiceProcessor:
    """Reusable offline processor with a matching stateful streaming mode."""

    def __init__(
        self,
        preset: str = "cute_helper_droid",
        *,
        config_path: str | Path | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> None:
        presets = load_presets(config_path)
        if preset not in presets:
            raise ValueError(f"unknown droid voice preset {preset!r}; choices: {', '.join(presets)}")
        values = asdict(presets[preset])
        if overrides:
            unknown = set(overrides) - set(values)
            if unknown:
                raise ValueError(f"unknown droid voice overrides: {', '.join(sorted(unknown))}")
            values.update(overrides)
        self.preset = preset
        self.config = DroidVoiceConfig(**values)
        self.config.validate()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process a complete mono buffer and preserve its duration and peak safety."""
        samples = _as_float_mono(audio)
        if samples.size == 0:
            return samples
        if self.config.pitch_semitones:
            samples = _pitch_shift(samples, self.config.pitch_semitones)
        if self.config.pitch_quantization_strength:
            samples = _quantize_global_pitch(samples, sample_rate, self.config.pitch_quantization_strength)
        if self.config.formant_shift:
            samples = _shift_spectral_envelope(samples, self.config.formant_shift)
        if self.config.spectral_mix:
            samples = _spectral_color(samples, self.config.spectral_mix)
        stream = self.stream(sample_rate)
        processed = stream.process_chunk(samples)
        return _normalize_peak(processed)

    def stream(self, sample_rate: int) -> "DroidVoiceStream":
        return DroidVoiceStream(self.config, sample_rate)

    def process_wav(self, input_path: str | Path, output_path: str | Path) -> None:
        audio, sample_rate = read_wav(input_path)
        write_wav(output_path, self.process(audio, sample_rate), sample_rate)

    def process_pcm16(self, pcm16: bytes, sample_rate: int) -> bytes:
        """Convenience adapter for a complete mono PCM16 TTS buffer."""
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return _to_pcm16(self.process(samples, sample_rate))


class DroidVoiceStream:
    """Stateful low-latency path for PCM chunks from a live TTS engine.

    The streaming path intentionally skips the look-ahead spectral stages. Its
    time-domain chain has no block buffering, so latency is limited to the
    configured micro-delay (10 ms or less in the supplied presets).
    """

    def __init__(self, config: DroidVoiceConfig, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.config = config
        self.sample_rate = sample_rate
        self._highpass_previous_input = 0.0
        self._highpass_previous_output = 0.0
        self._lowpass_previous_output = 0.0
        self._compressor_envelope = 0.0
        self._phase = 0.0
        delay_samples = max(1, round(sample_rate * config.micro_delay_ms / 1000.0))
        self._delay = np.zeros(delay_samples, dtype=np.float32)
        self._delay_index = 0
        self._held_value = 0.0
        self._hold_remaining = 0

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        samples = _as_float_mono(audio)
        if samples.size == 0:
            return samples
        output = np.empty_like(samples)
        for index, sample in enumerate(samples):
            colored = self._speaker_filter(float(sample))
            colored = self._sample_hold(colored)
            colored = self._ring_modulate(colored)
            colored = self._saturate(colored)
            colored = self._compress(colored)
            output[index] = self._micro_delay(colored)
        return np.clip(output, -1.0, 1.0)

    def process_pcm16_chunk(self, pcm16: bytes) -> bytes:
        """Process a mono PCM16 playback chunk while preserving stream state."""
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return _to_pcm16(self.process_chunk(samples))

    def begin_earcon(self) -> np.ndarray:
        return _earcon(self.sample_rate, ascending=True) if self.config.earcon else np.empty(0, np.float32)

    def end_earcon(self) -> np.ndarray:
        return _earcon(self.sample_rate, ascending=False) if self.config.earcon else np.empty(0, np.float32)

    def _speaker_filter(self, sample: float) -> float:
        amount = self.config.robotization_amount
        low = self.config.speaker_low_hz * (0.65 + 0.35 * amount)
        high = min(self.config.speaker_high_hz, self.sample_rate * 0.45)
        highpass_alpha = math.exp(-2.0 * math.pi * low / self.sample_rate)
        highpassed = highpass_alpha * (self._highpass_previous_output + sample - self._highpass_previous_input)
        self._highpass_previous_input = sample
        self._highpass_previous_output = highpassed
        lowpass_alpha = 1.0 - math.exp(-2.0 * math.pi * high / self.sample_rate)
        self._lowpass_previous_output += lowpass_alpha * (highpassed - self._lowpass_previous_output)
        return self._lowpass_previous_output

    def _sample_hold(self, sample: float) -> float:
        if self.config.sample_hold_hz <= 0.0:
            return sample
        hold_samples = max(1, round(self.sample_rate / self.config.sample_hold_hz))
        if self._hold_remaining <= 0:
            self._held_value = sample
            self._hold_remaining = hold_samples
        self._hold_remaining -= 1
        return self._held_value

    def _ring_modulate(self, sample: float) -> float:
        wet = self.config.ring_mod_mix * self.config.robotization_amount
        if wet <= 0.0:
            return sample
        carrier = math.sin(self._phase)
        self._phase = (self._phase + 2.0 * math.pi * self.config.ring_mod_frequency_hz / self.sample_rate) % (2.0 * math.pi)
        # Biased AM retains the carrier speech signal instead of deleting syllables.
        modulated = sample * (0.78 + 0.22 * carrier)
        return (1.0 - wet) * sample + wet * modulated

    def _saturate(self, sample: float) -> float:
        drive = 1.0 + 5.0 * self.config.saturation * self.config.robotization_amount
        saturated = math.tanh(sample * drive) / math.tanh(drive)
        levels = float((1 << self.config.bit_depth) - 1)
        quantized = round(saturated * levels) / levels
        mix = self.config.saturation * self.config.robotization_amount
        return (1.0 - mix) * saturated + mix * quantized

    def _compress(self, sample: float) -> float:
        attack = 1.0 - math.exp(-1.0 / (0.004 * self.sample_rate))
        release = 1.0 - math.exp(-1.0 / (0.080 * self.sample_rate))
        coefficient = attack if abs(sample) > self._compressor_envelope else release
        self._compressor_envelope += coefficient * (abs(sample) - self._compressor_envelope)
        threshold = 10.0 ** (self.config.compression_threshold_db / 20.0)
        if self._compressor_envelope <= threshold:
            return sample
        compressed_level = threshold * (self._compressor_envelope / threshold) ** (1.0 / self.config.compression_ratio)
        return sample * compressed_level / max(self._compressor_envelope, _EPSILON)

    def _micro_delay(self, sample: float) -> float:
        delayed = self._delay[self._delay_index]
        self._delay[self._delay_index] = sample
        self._delay_index = (self._delay_index + 1) % self._delay.size
        wet = self.config.micro_delay_mix * self.config.robotization_amount
        return (1.0 - wet) * sample + wet * delayed


def render_ab(input_path: str | Path, output_path: str | Path, *, config_path: str | Path | None = None) -> list[Path]:
    """Render the supplied input through the four standard droid intensities."""
    source = Path(input_path)
    destination = Path(output_path)
    audio, sample_rate = read_wav(source)
    outputs: list[Path] = []
    for label, preset in _AB_PRESETS.items():
        rendered = DroidVoiceProcessor(preset, config_path=config_path).process(audio, sample_rate)
        path = destination.with_name(f"{destination.stem}_{label}_{preset}{destination.suffix or '.wav'}")
        write_wav(path, rendered, sample_rate)
        outputs.append(path)
    return outputs


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        return _as_float_mono(audio), int(sample_rate)
    except ImportError:
        with wave.open(str(path), "rb") as stream:
            if stream.getsampwidth() != 2:
                raise RuntimeError("install soundfile to read WAV formats other than PCM16")
            raw = stream.readframes(stream.getnframes())
            audio = np.frombuffer(raw, dtype="<i2").reshape(-1, stream.getnchannels())
            return _as_float_mono(audio), stream.getframerate()


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(_as_float_mono(audio), -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(target, samples, sample_rate, subtype="PCM_16")
        return
    except ImportError:
        pass
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes((samples * 32767.0).astype("<i2").tobytes())


def _as_float_mono(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1:
        raise ValueError("audio must be a mono buffer or a frames-by-channels array")
    if np.issubdtype(samples.dtype, np.integer):
        scale = float(np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / scale
    return samples.astype(np.float32, copy=False)


def _normalize_peak(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples)))
    return samples if peak <= 0.98 else samples * (0.98 / peak)


def _to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _stft(
    samples: np.ndarray, frame_size: int = 1024, hop_size: int = 256
) -> tuple[np.ndarray, np.ndarray, int, int]:
    pad = frame_size
    padded = np.pad(samples, (pad, pad))
    remainder = (padded.size - frame_size) % hop_size
    if remainder:
        padded = np.pad(padded, (0, hop_size - remainder))
    frame_count = 1 + (padded.size - frame_size) // hop_size
    frames = np.stack([padded[index * hop_size:index * hop_size + frame_size] for index in range(frame_count)])
    window = np.hanning(frame_size).astype(np.float32)
    return np.fft.rfft(frames * window, axis=1), window, pad, samples.size


def _istft(
    spectrum: np.ndarray, window: np.ndarray, pad: int, original_size: int, hop_size: int = 256
) -> np.ndarray:
    frame_size = window.size
    output_size = hop_size * (spectrum.shape[0] - 1) + frame_size
    output = np.zeros(output_size, dtype=np.float32)
    weights = np.zeros(output_size, dtype=np.float32)
    for index, frame in enumerate(np.fft.irfft(spectrum, n=frame_size, axis=1)):
        start = index * hop_size
        output[start:start + frame_size] += frame.astype(np.float32) * window
        weights[start:start + frame_size] += window * window
    restored = output / np.maximum(weights, _EPSILON)
    return restored[pad:pad + original_size]


def _shift_spectral_envelope(samples: np.ndarray, semitones: float) -> np.ndarray:
    spectrum, window, pad, original_size = _stft(samples)
    magnitude = np.abs(spectrum)
    kernel = np.ones(25, dtype=np.float32) / 25.0
    envelope = np.stack([np.convolve(frame, kernel, mode="same") for frame in magnitude])
    ratio = 2.0 ** (semitones / 12.0)
    bins = np.arange(envelope.shape[1], dtype=np.float32)
    shifted = np.stack([np.interp(bins / ratio, bins, frame) for frame in envelope])
    detail = magnitude / np.maximum(envelope, _EPSILON)
    adjusted = detail * shifted
    return _istft(adjusted * np.exp(1j * np.angle(spectrum)), window, pad, original_size)


def _spectral_color(samples: np.ndarray, mix: float) -> np.ndarray:
    spectrum, window, pad, original_size = _stft(samples)
    magnitude = np.abs(spectrum)
    # Mild frequency-band quantization creates a compact machine coloration.
    block = 10
    colored = magnitude.copy()
    for start in range(0, magnitude.shape[1], block):
        end = min(start + block, magnitude.shape[1])
        colored[:, start:end] = magnitude[:, start:end].mean(axis=1, keepdims=True)
    blended = (1.0 - mix) * magnitude + mix * colored
    return _istft(blended * np.exp(1j * np.angle(spectrum)), window, pad, original_size)


def _pitch_shift(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Duration-preserving phase-vocoder pitch shift for small intentional changes."""
    ratio = 2.0 ** (semitones / 12.0)
    if abs(semitones) < 0.01:
        return samples
    resampled = _resample(samples, max(1, round(samples.size / ratio)))
    return _time_stretch(resampled, rate=1.0 / ratio, target_size=samples.size)


def _resample(samples: np.ndarray, target_size: int) -> np.ndarray:
    positions = np.linspace(0, samples.size - 1, target_size)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def _time_stretch(samples: np.ndarray, *, rate: float, target_size: int) -> np.ndarray:
    spectrum, window, pad, _ = _stft(samples)
    frame_count, bins = spectrum.shape
    positions = np.arange(0, frame_count - 1, rate)
    phase = np.angle(spectrum[0])
    phase_advance = 2.0 * np.pi * 256 * np.arange(bins) / 1024
    stretched = np.empty((positions.size, bins), dtype=np.complex128)
    for output_index, position in enumerate(positions):
        left = int(position)
        fraction = position - left
        first, second = spectrum[left], spectrum[left + 1]
        magnitude = (1.0 - fraction) * np.abs(first) + fraction * np.abs(second)
        delta = np.angle(second) - np.angle(first) - phase_advance
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase += phase_advance + delta
        stretched[output_index] = magnitude * np.exp(1j * phase)
    restored = _istft(stretched, window, pad, max(1, round(samples.size / rate)))
    return _resample(restored, target_size)


def _quantize_global_pitch(samples: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    """Nudge the dominant pitch toward a semitone grid without a childlike lift."""
    window = samples[: min(samples.size, sample_rate * 2)]
    if window.size < sample_rate // 20:
        return samples
    window = window - window.mean()
    correlation = np.correlate(window, window, mode="full")[window.size - 1:]
    minimum_lag, maximum_lag = max(1, sample_rate // 350), sample_rate // 80
    if correlation.size <= maximum_lag:
        return samples
    candidate = correlation[minimum_lag:maximum_lag]
    lag = minimum_lag + int(np.argmax(candidate))
    if correlation[lag] < 0.15 * correlation[0]:
        return samples
    frequency = sample_rate / lag
    midi = 69.0 + 12.0 * math.log2(frequency / 440.0)
    correction = (round(midi) - midi) * strength
    return _pitch_shift(samples, correction)


def _earcon(sample_rate: int, *, ascending: bool) -> np.ndarray:
    duration = 0.055
    count = round(sample_rate * duration)
    time = np.arange(count, dtype=np.float32) / sample_rate
    start, end = (510.0, 720.0) if ascending else (720.0, 510.0)
    frequency = start + (end - start) * time / duration
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    envelope = np.sin(np.pi * np.clip(time / duration, 0.0, 1.0))
    return (0.16 * np.sin(phase) * envelope).astype(np.float32)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply friendly utility-droid effects to a WAV file.")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("output", help="Output WAV file or A/B output base name")
    parser.add_argument("--preset", default="cute_helper_droid", help="Preset name")
    parser.add_argument("--config", help="Path to a droid_voice.toml preset file")
    parser.add_argument("--ab", action="store_true", help="Render A/B/C/D standard variations")
    args = parser.parse_args(argv)
    if args.ab:
        for output in render_ab(args.input, args.output, config_path=args.config):
            print(output)
        return 0
    DroidVoiceProcessor(args.preset, config_path=args.config).process_wav(args.input, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
