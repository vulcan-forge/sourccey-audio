import wave
from dataclasses import replace

from sourccey_voice.config import VoiceConfig, VadConfig
from sourccey_voice.replay import replay_wav

from .fakes import FakeProbability, FakeRecognizer


def test_replay_uses_audio_clock_and_never_requires_llm_or_tts(tmp_path, monkeypatch):
    import sourccey_voice.replay as replay

    path = tmp_path / "request.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\xd0\x07" * 16000)
    recognizer = FakeRecognizer("Source see, can you grab me a beer?")
    recognizer.close = lambda: None
    monkeypatch.setattr(replay, "MoonshineRecognizer", lambda *args: recognizer)
    monkeypatch.setattr(replay, "SileroProbability", lambda: FakeProbability([0.95] * 100))
    config = replace(VoiceConfig(), vad=VadConfig(min_speech_ms=20, silence_timeout_ms=40))
    result = replay_wav(path, config)
    assert result["results"][0]["transcript"] == "can you grab me a beer?"
    assert result["results"][0]["response"] == "Replay only."
