import base64
import json

import pytest

from sourccey_voice.network import MessageType, VoiceMessage, audio_payload, decode_audio


def test_audio_message_round_trip():
    message = VoiceMessage(MessageType.AUDIO_INPUT, 7, 1000, audio_payload(b"\x01\x00", 16000))
    parsed = VoiceMessage.from_json(message.to_json(), stale_after_ms=100, now_ms=1050)
    assert decode_audio(parsed) == (b"\x01\x00", 16000)


def test_stale_audio_is_rejected():
    message = VoiceMessage(MessageType.AUDIO_INPUT, 1, 1000, audio_payload(b"\x00\x00", 16000))
    with pytest.raises(ValueError, match="stale"):
        VoiceMessage.from_json(message.to_json(), stale_after_ms=100, now_ms=1200)


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        json.dumps({"type": "evil.v1", "sequence": 1, "created_at_ms": 1, "payload": {}}),
        json.dumps({"type": "audio.input.v1", "sequence": -1, "created_at_ms": 1, "payload": {}}),
    ],
)
def test_malformed_messages_are_rejected(raw):
    with pytest.raises(ValueError):
        VoiceMessage.from_json(raw)


def test_invalid_audio_base64_is_rejected():
    raw = VoiceMessage(
        MessageType.AUDIO_INPUT, 1, 1, {"pcm16_b64": "%%%", "sample_rate": 16000}
    ).to_json()
    with pytest.raises(ValueError, match="base64"):
        VoiceMessage.from_json(raw)


def test_message_size_limit_is_enforced():
    raw = VoiceMessage.create(MessageType.HEARTBEAT, 1, {"padding": "x" * 100}).to_json()
    with pytest.raises(ValueError, match="size limit"):
        VoiceMessage.from_json(raw, max_bytes=20)

