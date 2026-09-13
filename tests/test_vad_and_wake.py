import numpy as np

from sourccey_voice.commands import CommandRegistry
from sourccey_voice.config import VadConfig
from sourccey_voice.vad import VadEvent, VoiceActivityDetector
from sourccey_voice.wake import WakeSession


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

