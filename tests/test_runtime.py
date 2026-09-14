import numpy as np

from sourccey_voice.commands import CommandRegistry
from sourccey_voice.config import VadConfig
from sourccey_voice.conversation import ConversationHistory
from sourccey_voice.runtime import VoiceRuntime, normalize_transcript
from sourccey_voice.types import ConversationReply
from sourccey_voice.vad import VoiceActivityDetector
from sourccey_voice.wake import WakeSession

from .fakes import FakeConversation, FakeProbability, FakeRecognizer, FakeRobot, FakeSpeaker, FakeTts


def test_sourccey_stt_aliases_are_normalized():
    assert normalize_transcript("Hello, sourcing.") == "Hello, Sourccey."
    assert normalize_transcript("Hi sourcey") == "Hi Sourccey"
    assert normalize_transcript("Hello, Sorsi.") == "Hello, Sourccey."
    assert normalize_transcript("Hey, Sourcy.") == "Hey, Sourccey."
    assert normalize_transcript("Hi, Searcy.") == "Hi, Sourccey."
    assert normalize_transcript("Hello, Cersei.") == "Hello, Sourccey."
    assert normalize_transcript("Source and TV, help me.") == "Sourccey, help me."
    assert normalize_transcript("Sourced seed, are you ready?") == "Sourccey, are you ready?"
    assert normalize_transcript("Sir, see, are you ready?") == "Sourccey, are you ready?"


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
