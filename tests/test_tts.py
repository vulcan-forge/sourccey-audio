import numpy as np

from sourccey_voice.tts import _apply_robotic_treatment


def test_robotic_treatment_preserves_sample_shape_and_bounds():
    samples = np.linspace(-0.8, 0.8, 512, dtype=np.float32)
    treated = _apply_robotic_treatment(samples, sample_rate=24000)
    assert treated.shape == samples.shape
    assert np.max(np.abs(treated)) <= 1.0
    assert not np.array_equal(treated, samples)
