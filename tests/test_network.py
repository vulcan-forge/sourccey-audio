import base64
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sourccey_voice.network import MessageType, VoiceMessage, audio_payload, decode_audio
from sourccey_voice.network import VoiceHostServer


def test_audio_message_round_trip():
    message = VoiceMessage(MessageType.AUDIO_INPUT, 7, 1000, audio_payload(b"\x01\x00", 16000))
    parsed = VoiceMessage.from_json(message.to_json(), stale_after_ms=100, now_ms=1050)
    assert decode_audio(parsed) == (b"\x01\x00", 16000)


def test_unauthorized_connection_cannot_detach_active_robot():
    class Socket:
        async def recv(self):
            return VoiceMessage.create(MessageType.HELLO, 1, {"token": "wrong"}).to_json()

        async def close(self, **kwargs):
            pass

    speaker = Mock()
    runtime = Mock()
    server = VoiceHostServer(SimpleNamespace(max_message_bytes=131072, auth_token="secret"), runtime, speaker)
    server._connected = True
    asyncio.run(server._handle(Socket()))
    speaker.detach.assert_not_called()
    runtime.reset_audio.assert_not_called()
    assert server._connected


def test_disconnect_resets_recognition_and_wake_state():
    class Socket:
        async def recv(self):
            return VoiceMessage.create(MessageType.HELLO, 1, {"token": "secret"}).to_json()

        async def send(self, text):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    speaker, runtime = Mock(), Mock()
    server = VoiceHostServer(SimpleNamespace(max_message_bytes=131072, auth_token="secret"), runtime, speaker)
    asyncio.run(server._handle(Socket()))
    assert runtime.reset_audio.call_count == 2
    speaker.detach.assert_called_once()
    assert not server._connected


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
