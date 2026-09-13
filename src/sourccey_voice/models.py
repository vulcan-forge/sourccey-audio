from __future__ import annotations

import os
from pathlib import Path

from .config import VoiceConfig


def model_cache_dir() -> Path:
    configured = os.environ.get("SOURCCEY_VOICE_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "sourccey-voice" / "models"


def download_model(kind: str, config: VoiceConfig) -> Path:
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    if kind == "stt":
        try:
            from moonshine_voice import ModelArch, get_model_for_language
        except ImportError as exc:
            raise RuntimeError("STT model setup requires sourccey-voice[stt]") from exc
        try:
            architecture = getattr(ModelArch, config.stt.architecture.upper())
        except AttributeError as exc:
            raise ValueError(f"unsupported Moonshine architecture: {config.stt.architecture}") from exc
        path, _selected_architecture = get_model_for_language(
            config.stt.language,
            architecture,
            cache_root=cache / "moonshine",
        )
        return Path(path)
    if kind == "llm":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("LLM model download requires sourccey-voice[models]") from exc
        path = hf_hub_download(
            repo_id=config.llm.model,
            filename=config.llm.model_file,
            cache_dir=cache / "huggingface",
        )
        return Path(path)
    if kind == "tts":
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("TTS model setup requires sourccey-voice[tts]") from exc
        KPipeline(lang_code=config.tts.language_code)
        return cache
    raise ValueError("model kind must be stt, llm, or tts")
