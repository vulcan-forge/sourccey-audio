"""Generate a local Qwen3-TTS VoiceDesign audition for Sourccey."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
SOURCCINA_DESCRIPTION = (
    "A small, cheerful service robot with an androgynous youthful voice. "
    "Clear and friendly with a light synthetic texture, playful but not childish, "
    "warm, expressive, and easy to understand through a compact speaker."
)
SAMPLE_TEXT = (
    "Hi, I am Sourccey. I am ready to help, explore, and keep you company."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="sourccina", help="Output voice name.")
    parser.add_argument(
        "--description",
        default=SOURCCINA_DESCRIPTION,
        help="Natural-language voice design brief.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS trial requires a CUDA-capable PyTorch installation.")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.float16,
    )
    wavs, sample_rate = model.generate_voice_design(
        text=SAMPLE_TEXT,
        language="English",
        instruct=args.description,
    )
    output = Path("artifacts/qwen-tts") / f"{args.name}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sample_rate)
    print(output.resolve())


if __name__ == "__main__":
    main()
