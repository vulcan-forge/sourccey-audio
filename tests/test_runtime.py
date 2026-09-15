import numpy as np
import pytest

from sourccey_voice.commands import CommandRegistry
from sourccey_voice.config import VadConfig
from sourccey_voice.conversation import ConversationHistory
from sourccey_voice.runtime import VoiceRuntime, normalize_transcript, pronunciation_text
from sourccey_voice.types import ConversationReply
from sourccey_voice.vad import VoiceActivityDetector
from sourccey_voice.wake import WakeSession

from .fakes import FakeConversation, FakeProbability, FakeRecognizer, FakeRobot, FakeSpeaker, FakeTts


def test_tts_uses_pronunciation_spelling_without_changing_canonical_text():
    assert pronunciation_text("Sourccey is ready. Tell Sourccey's friend.") == "Soarsee is ready. Tell Soarsee's friend."


def test_normalization_preserves_request_words():
    assert normalize_transcript("Cersei, tell me about mercy.") == "Sourccey, tell me about mercy."
    assert normalize_transcript("We talked about Cersei and sorcery.") == "We talked about Cersei and sorcery."


def make_runtime(
    *,
    robot=None,
    conversation=None,
    tts=None,
    speaker=None,
    recognizer=None,
    probability=None,
    vad=None,
    allow_tool_requests=True,
):
    registry = CommandRegistry()
    return VoiceRuntime(
        registry=registry,
        robot=robot or FakeRobot(),
        conversation=conversation or FakeConversation(),
        tts=tts or FakeTts(),
        speaker=speaker or FakeSpeaker(),
        history=ConversationHistory(3, 1000, 60),
        wake=WakeSession(False, ["hey sourccey"], 10, True, registry),
        output_sample_rate=16000,
        recognizer=recognizer,
        probability=probability,
        vad=vad,
        allow_tool_requests=allow_tool_requests,
    )


def test_explicit_command_works_when_llm_is_down():
    robot = FakeRobot({"STOP"})
    llm = FakeConversation(error=RuntimeError("offline"))
    result = make_runtime(robot=robot, conversation=llm).handle_transcript("Command: stop")
    assert result.accepted is True
    assert robot.executed == ["STOP"]
    assert llm.calls == 0


def test_unknown_explicit_command_never_reaches_llm():
    llm = FakeConversation()
    result = make_runtime(conversation=llm).handle_transcript("Command: drive forward")
    assert result.accepted is False
    assert result.reason == "unknown_command"
    assert llm.calls == 0


def test_unapproved_llm_action_never_reaches_robot():
    robot = FakeRobot({"STOP", "FOLLOW_ME"})
    llm = FakeConversation(ConversationReply("Doing it.", "DRIVE_FORWARD"))
    result = make_runtime(robot=robot, conversation=llm).handle_transcript("go over there")
    assert result.accepted is False
    assert result.reason == "unapproved_llm_action"
    assert robot.executed == []


def test_approved_but_unavailable_llm_action_reports_failure():
    robot = FakeRobot({"STOP"})
    llm = FakeConversation(ConversationReply("Sure.", "FOLLOW_ME"))
    result = make_runtime(robot=robot, conversation=llm).handle_transcript("can you follow me?")
    assert result.accepted is False
    assert "not available" in result.response
    assert robot.executed == []


def test_llm_tool_requests_can_be_disabled():
    robot = FakeRobot({"STOP"})
    llm = FakeConversation(ConversationReply("Stopping.", "STOP"))
    result = make_runtime(
        robot=robot, conversation=llm, allow_tool_requests=False
    ).handle_transcript("please stop")
    assert result.reason == "tool_requests_disabled"
    assert robot.executed == []


def test_priority_command_interrupts_tts_before_execution():
    robot = FakeRobot({"FREEZE"})
    speaker = FakeSpeaker(playing=True)
    result = make_runtime(robot=robot, speaker=speaker).handle_transcript("Command: freeze")
    assert result.accepted is True
    assert speaker.interruptions == 1
    assert robot.executed == ["FREEZE"]


def test_command_still_executes_when_tts_fails():
    robot = FakeRobot({"CANCEL"})
    result = make_runtime(robot=robot, tts=FakeTts(error=RuntimeError("no voice"))).handle_transcript(
        "Command: cancel"
    )
    assert result.accepted is True
    assert robot.executed == ["CANCEL"]


def test_command_handler_exception_is_not_reported_as_success():
    robot = FakeRobot({"STOP"}, error=RuntimeError("link lost"))
    result = make_runtime(robot=robot).handle_transcript("Command: stop")
    assert result.accepted is False
    assert "link lost" in result.response


def test_mock_audio_to_stt_to_command_to_tts_integration():
    recognizer = FakeRecognizer("Command: stop")
    probability = FakeProbability([0.9, 0.1, 0.1])
    vad = VoiceActivityDetector(
        VadConfig(min_speech_ms=20, silence_timeout_ms=40, pre_roll_ms=0, post_roll_ms=20),
        16000,
    )
    robot = FakeRobot({"STOP"})
    tts = FakeTts()
    runtime = make_runtime(
        robot=robot, recognizer=recognizer, probability=probability, vad=vad, tts=tts
    )
    frame = (np.ones(320) * 1000).astype("<i2").tobytes()
    silence = np.zeros(320, dtype="<i2").tobytes()
    assert runtime.push_pcm16(frame, 16000) == []
    assert runtime.push_pcm16(silence, 16000) == []
    results = runtime.push_pcm16(silence, 16000)
    assert results[0].command == "STOP"
    assert robot.executed == ["STOP"]
    assert tts.texts == ["done"]


def test_low_energy_ends_turn_when_silero_score_stays_high():
    recognizer = FakeRecognizer("hello")
    probability = FakeProbability([0.9, 0.9, 0.9])
    vad = VoiceActivityDetector(
        VadConfig(
            min_speech_ms=20,
            silence_timeout_ms=40,
            pre_roll_ms=0,
            post_roll_ms=20,
            silence_rms_threshold=0.01,
        ),
        16000,
    )
    runtime = make_runtime(recognizer=recognizer, probability=probability, vad=vad)
    speech = (np.ones(320) * 2000).astype("<i2").tobytes()
    silence = np.zeros(320, dtype="<i2").tobytes()
    assert runtime.push_pcm16(speech, 16000) == []
    assert runtime.push_pcm16(silence, 16000) == []
    result = runtime.push_pcm16(silence, 16000)
    assert result[0].transcript == "hello"


def test_overlong_turn_never_routes_partial_command():
    vad = VoiceActivityDetector(VadConfig(min_speech_ms=20, max_utterance_ms=1000), 16000)
    llm = FakeConversation()
    runtime = make_runtime(
        recognizer=FakeRecognizer("Command: follow me"),
        probability=FakeProbability([0.9] * 55), vad=vad, conversation=llm,
    )
    frame = (np.ones(320) * 2000).astype("<i2").tobytes()
    results = [result for _ in range(50) for result in runtime.push_pcm16(frame, 16000)]
    assert [r.reason for r in results] == ["utterance_too_long"]
    assert llm.calls == 0
    assert runtime.robot.executed == []


def test_old_window_option_cannot_cut_off_request():
    vad = VoiceActivityDetector(VadConfig(
        always_listen_window_ms=100, min_speech_ms=20, silence_timeout_ms=40,
    ), 16000)
    runtime = make_runtime(recognizer=FakeRecognizer("hello"),
                           probability=FakeProbability([0.9] * 22), vad=vad)
    speech = (np.ones(320) * 2000).astype("<i2").tobytes()
    for _ in range(20):
        assert runtime.push_pcm16(speech, 16000) == []
    silence = np.zeros(320, dtype="<i2").tobytes()
    assert runtime.push_pcm16(silence, 16000) == []
    assert runtime.push_pcm16(silence, 16000)[0].transcript == "hello"


def test_wrong_audio_rate_is_rejected_before_recognition():
    runtime = make_runtime(recognizer=FakeRecognizer(""),
                           probability=FakeProbability([]), vad=VoiceActivityDetector(VadConfig(), 16000))
    with pytest.raises(ValueError, match="sample rate"):
        runtime.push_pcm16(b"\x00\x00", 48000)


def test_raw_name_decision_reaches_llm_without_corrupting_request():
    runtime = make_runtime()
    runtime.wake = WakeSession(True, ["sourccey"], 0, False, runtime.registry)
    assert runtime.handle_transcript("Sorry, could you help me?").kind == "ignored"
    result = runtime.handle_transcript("Source see, tell me about Cersei.")
    assert result.transcript == "tell me about Cersei."
    assert runtime.conversation.calls == 1


def test_reset_discards_pending_name_continuation():
    runtime = make_runtime()
    runtime.wake = WakeSession(True, ["sourccey"], 0, False, runtime.registry)
    runtime.handle_transcript("Sourccey")
    runtime.reset_audio()
    assert runtime.handle_transcript("can you help?").kind == "ignored"
