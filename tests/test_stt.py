from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sourccey_voice.stt import MoonshineRecognizer


@pytest.mark.parametrize("fails", [False, True])
def test_native_stream_is_freed_even_if_finalization_fails(fails):
    recognizer = MoonshineRecognizer.__new__(MoonshineRecognizer)
    stream = Mock()
    recognizer._stream = stream
    if fails:
        stream.stop.side_effect = RuntimeError("decoder failed")
        with pytest.raises(RuntimeError):
            recognizer.finalize()
    else:
        stream.stop.return_value = SimpleNamespace(lines=[SimpleNamespace(text="hello")])
        assert recognizer.finalize() == "hello"
    stream.close.assert_called_once()
    assert recognizer._stream is None


def test_discard_frees_stream_without_decoding_truncated_audio():
    recognizer = MoonshineRecognizer.__new__(MoonshineRecognizer)
    stream = Mock()
    recognizer._stream = stream
    recognizer.stop()
    stream.close.assert_called_once()
    stream.stop.assert_not_called()
