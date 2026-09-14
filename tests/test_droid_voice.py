import numpy as np

from sourccey_voice.droid_voice import DroidVoiceProcessor, load_presets, read_wav, render_ab, write_wav


def _adult_voice_like_signal(sample_rate: int = 24000) -> np.ndarray:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    return 0.42 * np.sin(2 * np.pi * 155 * time) + 0.16 * np.sin(2 * np.pi * 310 * time)


def test_default_processor_preserves_shape_and_safety():
    source = _adult_voice_like_signal()
    processed = DroidVoiceProcessor().process(source, 24000)
    assert processed.shape == source.shape
    assert np.isfinite(processed).all()
    assert np.max(np.abs(processed)) <= 1.0
    assert not np.allclose(processed, source)


def test_streaming_processor_keeps_state_across_chunks():
    source = _adult_voice_like_signal()
    stream = DroidVoiceProcessor().stream(24000)
    processed = np.concatenate([stream.process_chunk(source[:8000]), stream.process_chunk(source[8000:])])
    assert processed.shape == source.shape
    assert np.isfinite(processed).all()


def test_pcm16_streaming_adapter_preserves_frame_count():
    source = _adult_voice_like_signal()[:1600]
    pcm16 = (source * 32767.0).astype("<i2").tobytes()
    processed = DroidVoiceProcessor().stream(24000).process_pcm16_chunk(pcm16)
    assert len(processed) == len(pcm16)


def test_preset_loading_includes_standard_ab_variants():
    presets = load_presets()
    assert {"subtle_droid", "cute_helper_droid", "more_synthetic", "maximum_robot"} <= set(presets)
    assert presets["cute_helper_droid"].pitch_semitones == 0.0


def test_ab_renderer_writes_all_variations(tmp_path):
    source = tmp_path / "source.wav"
    write_wav(source, _adult_voice_like_signal(), 24000)
    outputs = render_ab(source, tmp_path / "robot.wav")
    assert len(outputs) == 4
    for output in outputs:
        rendered, sample_rate = read_wav(output)
        assert sample_rate == 24000
        assert rendered.size > 0
