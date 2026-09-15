import numpy as np
import pytest

from sourccey_voice.commands import CommandRegistry
from sourccey_voice.config import VadConfig
from sourccey_voice.vad import AdaptiveEnergyGate, VadEvent, VoiceActivityDetector
from sourccey_voice.wake import WakeMatcher, WakeSession


def test_vad_transitions_and_preroll_postroll():
    config = VadConfig(
        threshold=0.5,
        min_speech_ms=40,
        silence_timeout_ms=40,
        pre_roll_ms=20,
        post_roll_ms=20,
    )
    vad = VoiceActivityDetector(config, 1000)
    frame = np.ones(20, dtype=np.float32)
    assert vad.push(frame * 0, 0.0) == []
    assert vad.push(frame, 0.9) == []
    started = vad.push(frame, 0.9)
    assert [update.event for update in started] == [VadEvent.SPEECH_STARTED]
    assert started[0].samples.size == 60
    assert vad.push(frame * 0, 0.1) == []
    ended = vad.push(frame * 0, 0.1)
    assert ended[0].event == VadEvent.SPEECH_ENDED
    assert ended[0].samples.size == 20


def test_short_noise_never_starts_speech():
    vad = VoiceActivityDetector(VadConfig(min_speech_ms=60, pre_roll_ms=0), 1000)
    assert vad.push(np.ones(20), 0.9) == []
    assert vad.push(np.zeros(20), 0.0) == []


def test_wake_phrase_opens_engaged_window():
    now = [10.0]
    session = WakeSession(True, ["sourccey", "hey sourccey"], 5, True, CommandRegistry(), clock=lambda: now[0])
    first = session.evaluate("Hey Sourccey, who are you?")
    assert first.accepted and first.woke and first.text == "who are you?"
    now[0] = 14.0
    assert session.evaluate("and what can you do?").accepted
    now[0] = 20.0
    assert not session.evaluate("still there?").accepted


def test_emergency_command_bypasses_wake_word():
    session = WakeSession(True, ["hey sourccey"], 5, True, CommandRegistry(), clock=lambda: 10.0)
    decision = session.evaluate("Command: stop")
    assert decision.accepted and decision.reason == "emergency_bypass"


def test_non_emergency_command_does_not_bypass_wake_word():
    session = WakeSession(True, ["hey sourccey"], 5, True, CommandRegistry(), clock=lambda: 10.0)
    assert not session.evaluate("Command: follow me").accepted


@pytest.mark.parametrize("prefix", [
    "Sourccey", "Source see", "Source C", "Source sea", "Source-see", "Hey, Sourccey", "Hello Sourcy",
    "Sorsi", "Sourccy", "Sourcceyy", "Sorcerer", "Cersei", "Horsey", "Mercy", "Soros",
    "Source and TV", "Sir, see", "Sourced seed",
])
def test_observed_name_variants_in_direct_address(prefix):
    match = WakeMatcher(["sourccey"]).match(f"{prefix}, can you grab me a beer?")
    assert match is not None
    assert match.remainder == "can you grab me a beer?"


@pytest.mark.parametrize("text", [
    "Sorry, can you help?", "Surely this works", "Source code needs fixing",
    "Mercy is important", "Cersei was in that show", "Soros was on the news",
    "Horsey", "I told Sourccey to stop", "We discussed source see yesterday",
    "Of course, can you grab me a beer?", "Please get me the beer", "Thank you",
    "123 Sourccey help", "Sourcceyish can you help?", "Sorcery is interesting",
])
def test_background_conversation_does_not_wake(text):
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry())
    assert not session.evaluate(text).accepted


def test_ambiguous_aliases_can_be_disabled():
    matcher = WakeMatcher(["sourccey"], contextual_aliases=())
    assert matcher.match("Mercy, please help") is None


def test_sorosy_observed_request_and_background_mentions():
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry())
    decision = session.evaluate("Sorosy, can you tell Nick to check the answers?")
    assert decision.accepted
    assert decision.reason == "contextual_alias"
    assert decision.text == "can you tell Nick to check the answers?"
    assert not session.evaluate("Sorosy was mentioned earlier.").accepted
    assert not session.evaluate("I asked Sorosy to check the answers.").accepted


def test_source_variant_with_second_person_statement_is_addressed():
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry())
    result = session.evaluate("Source, it is your voice sound to speak.")
    assert result.accepted
    assert result.text == "it is your voice sound to speak."
    assert not session.evaluate("Source code needs a voice.").accepted


def test_name_alone_claims_one_continuation_at_speech_start():
    now = [0.0]
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry(), clock=lambda: now[0])
    assert not session.evaluate("hi").accepted  # No engaged window at clock zero.
    assert session.evaluate("Sourccey").reason == "awaiting_request"
    now[0] = 1.0
    session.begin_utterance()
    now[0] = 6.0  # The full request can take longer than the start deadline.
    result = session.evaluate("can you grab me a beer?")
    assert result.accepted and result.reason == "wake_continuation"
    assert not session.evaluate("and some chips").accepted


def test_expired_or_empty_continuation_does_not_leak():
    now = [10.0]
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry(), clock=lambda: now[0])
    session.evaluate("Sourccey")
    now[0] = 13.0
    assert not session.evaluate("get me a beer").accepted
    session.evaluate("Sourccey")
    session.begin_utterance()
    session.evaluate("")
    assert not session.evaluate("get me a beer").accepted


def test_full_request_does_not_open_continuation():
    session = WakeSession(True, ["sourccey"], 0, False, CommandRegistry())
    assert session.evaluate("Sourccey hello").accepted
    assert not session.evaluate("please help").accepted


def test_adaptive_energy_preserves_soft_voice_and_detects_relative_silence():
    gate = AdaptiveEnergyGate(VadConfig())
    for _ in range(400):
        gate.apply(0.004, 0.0, False, 0.02)
    assert gate.apply(0.006, 0.95, False, 0.02) == 0.95
    for _ in range(50):
        assert gate.apply(0.006, 0.95, True, 0.02) == 0.95
    assert gate.apply(0.0003, 0.95, True, 0.02) == 0.0


def test_speaking_does_not_raise_estimated_room_floor():
    gate = AdaptiveEnergyGate(VadConfig())
    original_floor = gate.noise_rms
    for _ in range(200):
        gate.apply(0.4, 0.99, True, 0.02)
    assert gate.noise_rms == original_floor


def test_short_pause_keeps_one_utterance_and_original_audio():
    vad = VoiceActivityDetector(VadConfig(min_speech_ms=20, silence_timeout_ms=100,
                                         pre_roll_ms=0), 1000)
    speech = np.ones(20, dtype=np.float32)
    assert vad.push(speech, 0.9)[0].event == VadEvent.SPEECH_STARTED
    for _ in range(4):
        assert vad.push(speech * 0, 0.0) == []
    updates = vad.push(speech, 0.9)
    assert all(u.event == VadEvent.SPEECH_CONTINUING for u in updates)
    assert sum(u.samples.size for u in updates) == 100
