"""Standalone local voice interaction module for Sourccey."""

from .commands import CommandName, CommandRegistry
from .config import VoiceConfig, load_config
from .runtime import InteractionResult, VoiceRuntime

__all__ = [
    "CommandName",
    "CommandRegistry",
    "InteractionResult",
    "VoiceConfig",
    "VoiceRuntime",
    "load_config",
]

__version__ = "0.1.0"
