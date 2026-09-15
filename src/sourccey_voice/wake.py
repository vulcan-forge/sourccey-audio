from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Iterable

from .commands import CommandRegistry, PRIORITY_COMMANDS


# These are spelling hints, not acoustic confidence or a speaker identity check.
DEFAULT_ALIASES = ("source see", "source c", "source sea", "source-see", "sourcey", "sourcy", "sorsi", "searcy")
DEFAULT_CONTEXTUAL_ALIASES = (
    "cersei", "circe", "sorcerer", "sorcery", "horsey", "horsie", "horsy",
    "mercy", "mersey", "soros", "sorosy", "sourcing", "source", "sourced seed", "sir see",
    "source and tv", "source n tv",
)
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_REQUEST = re.compile(
    r"(?:(?:can|could|would|will|do|did|are|have)\s+you\b|"
    r"please\b|(?:what|where|when|why|how|who)\b|"
    r"(?:help|tell|show|explain|grab|fetch|bring|follow|stop|cancel|freeze|command)\b|"
    r"\b(?:you|your|you're|youre)\b)",
    re.IGNORECASE,
)
_DIRECT_REQUEST = re.compile(r"^(?:(?:can|could|would|will|do|did|are|have)\s+you\b|please\b|(?:what|where|when|why|how|who)\b|(?:help|tell|show|explain|get|grab|fetch|bring|make|play|put|take|follow|stop|cancel|freeze|command)\b)", re.IGNORECASE)
_DIRECT_REFERENCE = re.compile(r"\b(?:you|your|you're|youre)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PrefixMatch:
    matched: str
    remainder: str
    reason: str
    similarity: float = 1.0


class WakeMatcher:
    """Match only a leading address; preserve every word of the request."""

    def __init__(
        self, phrases: Iterable[str], *, aliases: Iterable[str] = DEFAULT_ALIASES,
        contextual_aliases: Iterable[str] = DEFAULT_CONTEXTUAL_ALIASES,
        fuzzy_threshold: float = 0.65,
    ) -> None:
        self.phrases = self._tokenize(phrases)
        self.aliases = self._tokenize(aliases)
        self.contextual_aliases = self._tokenize(contextual_aliases)
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _tokenize(phrases: Iterable[str]) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted({tuple(_TOKEN.findall(p.casefold())) for p in phrases if p.strip()}, key=len, reverse=True))

    def match(self, text: str) -> PrefixMatch | None:
        tokens = list(_TOKEN.finditer(text))
        if not tokens:
            return None
        if text[:tokens[0].start()].strip(" \t\r\n\"'.,!?-"):
            return None
        words = tuple(t.group().casefold() for t in tokens)
        # Greeting tolerance is restricted to a leading "hey" or "hello".
        offsets = (0, 1) if words[0] in {"hey", "hello"} else (0,)
        for offset in offsets:
            for choices, reason in (
                (self.phrases, "exact"), (self.aliases, "alias"),
                (self.contextual_aliases, "contextual_alias"),
            ):
                for phrase in choices:
                    end = offset + len(phrase)
                    if not phrase or words[offset:end] != phrase:
                        continue
                    remainder = text[tokens[end - 1].end():].lstrip(" \t\r\n,.:;!?-")
                    if reason == "contextual_alias" and not (
                        _DIRECT_REQUEST.match(remainder) or _DIRECT_REFERENCE.search(remainder)
                    ):
                        continue
                    return PrefixMatch(text[:tokens[end - 1].end()].strip(), remainder, reason)
            if offset >= len(tokens):
                continue
            candidate = words[offset]
            remainder = text[tokens[offset].end():].lstrip(" \t\r\n,.:;!?-")
            if len(candidate) < 5 or len(candidate) > 9 or not _DIRECT_REQUEST.match(remainder):
                continue
            # Tight orthographic fallback only against canonical names, never
            # against ambiguous aliases (which would multiply false matches).
            score = max((SequenceMatcher(None, candidate, p[0]).ratio()
                         for p in self.phrases if len(p) == 1), default=0.0)
            if score >= self.fuzzy_threshold:
                return PrefixMatch(text[:tokens[offset].end()].strip(), remainder, "close_spelling", score)
        return None


@dataclass(frozen=True)
class WakeDecision:
    accepted: bool
    text: str
    woke: bool = False
    reason: str = ""
    matched: str = ""
    similarity: float = 0.0


class WakeSession:
    def __init__(
        self, enabled: bool, phrases: Iterable[str], engaged_seconds: float,
        emergency_bypass: bool, registry: CommandRegistry, *,
        clock: Callable[[], float] = time.monotonic,
        aliases: Iterable[str] = DEFAULT_ALIASES,
        contextual_aliases: Iterable[str] = DEFAULT_CONTEXTUAL_ALIASES,
        fuzzy_threshold: float = 0.65,
        continuation_seconds: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.phrases = tuple(phrases)
        self.engaged_seconds = engaged_seconds
        self.emergency_bypass = emergency_bypass
        self.registry = registry
        self._clock = clock
        self.continuation_seconds = continuation_seconds
        self.matcher = WakeMatcher(self.phrases, aliases=aliases,
                                   contextual_aliases=contextual_aliases, fuzzy_threshold=fuzzy_threshold)
        self.reset()

    def reset(self) -> None:
        self._engaged_until = float("-inf")
        self._pending_until = float("-inf")
        self._continuing = False

    def begin_utterance(self) -> None:
        # Claim the one-use continuation when speech starts, not after a long
        # request has finished decoding. An empty/noisy turn consumes it too.
        self._continuing = self._clock() < self._pending_until
        self._pending_until = float("-inf")

    @property
    def engaged(self) -> bool:
        return not self.enabled or self._clock() < self._engaged_until

    def evaluate(self, transcript: str) -> WakeDecision:
        text = transcript.strip()
        continuation = self._continuing or self._clock() < self._pending_until
        self._continuing = False
        self._pending_until = float("-inf")
        if not text:
            return WakeDecision(False, text, reason="empty")
        if not self.enabled:
            return WakeDecision(True, text, reason="wake_disabled")
        explicit = self.registry.parse_explicit(text)
        if self.emergency_bypass and explicit is not None and explicit.name in PRIORITY_COMMANDS:
            self._engage()
            return WakeDecision(True, text, reason="emergency_bypass")
        match = self.matcher.match(text)
        if match:
            if not match.remainder:
                self._pending_until = self._clock() + self.continuation_seconds
                return WakeDecision(False, "", True, "awaiting_request", match.matched, match.similarity)
            self._engage()
            return WakeDecision(True, match.remainder, True, match.reason, match.matched, match.similarity)
        if continuation:
            self._engage()
            return WakeDecision(True, text, reason="wake_continuation")
        if self.engaged:
            self._engage()
            return WakeDecision(True, text, reason="engaged_window")
        return WakeDecision(False, text, reason="wake_required")

    def _engage(self) -> None:
        self._engaged_until = self._clock() + self.engaged_seconds
