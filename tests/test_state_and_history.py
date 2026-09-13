import json

import pytest

from sourccey_voice.config import config_from_mapping, default_config_path, load_config
from sourccey_voice.conversation import ConversationHistory
from sourccey_voice.state import RobotState, state_from_sourccey_observation


def test_history_is_bounded_by_turns():
    history = ConversationHistory(2, 1000, 60)
    history.add("one", "a")
    history.add("two", "b")
    history.add("three", "c")
    assert [item["content"] for item in history.messages()] == ["two", "b", "three", "c"]


def test_history_is_bounded_by_characters():
    history = ConversationHistory(10, 128, 60)
    history.add("x" * 100, "a")
    history.add("y" * 100, "b")
    assert history.character_count <= 128


def test_history_expires_after_inactivity():
    now = [0.0]
    history = ConversationHistory(3, 1000, 5, clock=lambda: now[0])
    history.add("hello", "hi")
    now[0] = 6.0
    assert history.messages() == []


def test_robot_state_serializes_only_real_supplied_fields():
    state = state_from_sourccey_observation({"x.vel": 0.2, "z.pos": 17.0, "camera": object()})
    assert json.loads(state.to_json()) == {"x_velocity": 0.2, "z_position": 17.0}


def test_unknown_robot_state_field_is_rejected():
    with pytest.raises(ValueError, match="unknown robot state"):
        RobotState({"imaginary_sensor": True})


def test_packaged_default_config_is_available():
    assert default_config_path().is_file()
    config = load_config(validate=False)
    assert config.audio.sample_rate == 16000
    assert config.stt.architecture == "medium_streaming"


def test_packaged_and_repository_configs_match():
    repository_config = default_config_path().parents[3] / "config" / "default.toml"
    if repository_config.is_file():
        assert default_config_path().read_bytes() == repository_config.read_bytes()


@pytest.mark.parametrize(
    "audio",
    [
        {"chunk_ms": 0},
        {"volume": 1.1},
    ],
)
def test_invalid_audio_config_is_rejected(audio):
    with pytest.raises(ValueError):
        config_from_mapping(
            {"audio": audio, "network": {"auth_token": "test-secret"}}, validate=True
        )
