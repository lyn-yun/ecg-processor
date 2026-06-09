"""Tests for the preprocessing pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.preprocessing import (
    PreprocessingConfig,
    preprocess,
    remove_baseline_wander,
    remove_high_freq_noise,
    remove_powerline_noise,
)

SAMPLING_RATE: int = 360


def _make_test_signal(n_samples: int = 3600, seed: int = 42) -> np.ndarray:
    """Create a simple test signal with low-freq drift + 50 Hz + noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples / SAMPLING_RATE, n_samples, endpoint=False)
    signal = (
        np.sin(2 * np.pi * 1.0 * t)           # 1 Hz — "ECG-like"
        + 0.2 * np.sin(2 * np.pi * 0.2 * t)   # baseline wander
        + 0.1 * np.sin(2 * np.pi * 50 * t)    # power-line
        + 0.05 * rng.normal(0, 1, n_samples)  # EMG
    )
    return signal


class TestPreprocessingConfig:
    """Tests for the PreprocessingConfig dataclass."""

    def test_default_config(self) -> None:
        cfg = PreprocessingConfig(sampling_rate=360)
        assert cfg.sampling_rate == 360
        assert cfg.filter_type == "iir"
        assert "baseline" in cfg.enabled_stages

    def test_invalid_filter_type_raises(self) -> None:
        with pytest.raises(ValueError):
            PreprocessingConfig(filter_type="invalid")  # type: ignore[arg-type]

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValueError):
            PreprocessingConfig(enabled_stages=["unknown"])  # type: ignore[list-item]

    def test_custom_config(self) -> None:
        cfg = PreprocessingConfig(
            sampling_rate=500, filter_type="fir", notch_freq=60,
            enabled_stages=["baseline", "notch"],
        )
        assert cfg.sampling_rate == 500
        assert cfg.notch_freq == 60
        assert "lowpass" not in cfg.enabled_stages


class TestPreprocessingPipeline:
    """Integration tests for the full preprocessing pipeline."""

    def test_output_keys(self) -> None:
        signal = _make_test_signal()
        result = preprocess(signal, SAMPLING_RATE)
        for key in ["raw", "baseline_removed", "notch_filtered", "filtered"]:
            assert key in result, f"Missing key: {key}"

    def test_output_same_length(self) -> None:
        signal = _make_test_signal()
        result = preprocess(signal, SAMPLING_RATE)
        for arr in result.values():
            assert len(arr) == len(signal)

    def test_baseline_suppressed(self) -> None:
        """Baseline wander removal should reduce low-frequency power."""
        signal = _make_test_signal(3600)
        filtered = remove_baseline_wander(signal, SAMPLING_RATE, filter_type="iir")
        # Low-frequency (< 0.3 Hz) energy should decrease
        from src.filters import compute_spectrum
        freqs_raw, amp_raw = compute_spectrum(signal, SAMPLING_RATE)
        freqs_filt, amp_filt = compute_spectrum(filtered, SAMPLING_RATE)
        low_idx = np.where(freqs_raw < 0.3)[0]
        assert amp_filt[low_idx].mean() < amp_raw[low_idx].mean()

    def test_notch_suppresses_50hz(self) -> None:
        """The notch filter should reduce 50 Hz component."""
        signal = _make_test_signal()
        filtered = remove_powerline_noise(
            signal, SAMPLING_RATE, notch_freq=50, filter_type="iir",
        )
        from src.filters import compute_spectrum
        freqs_raw, amp_raw = compute_spectrum(signal, SAMPLING_RATE)
        freqs_filt, amp_filt = compute_spectrum(filtered, SAMPLING_RATE)
        idx_50 = np.argmin(np.abs(freqs_raw - 50.0))
        assert amp_filt[idx_50] < amp_raw[idx_50], \
            f"50 Hz should be suppressed: raw={amp_raw[idx_50]:.4f}, filt={amp_filt[idx_50]:.4f}"

    def test_lowpass_suppresses_high_freq(self) -> None:
        """Low-pass filter should reduce > 40 Hz content."""
        signal = _make_test_signal()
        filtered = remove_high_freq_noise(signal, SAMPLING_RATE, filter_type="iir")
        from src.filters import compute_spectrum
        freqs_raw, amp_raw = compute_spectrum(signal, SAMPLING_RATE)
        freqs_filt, amp_filt = compute_spectrum(filtered, SAMPLING_RATE)
        high_idx = np.where(freqs_raw > 45.0)[0]
        if len(high_idx) > 0:
            assert amp_filt[high_idx].mean() < amp_raw[high_idx].mean()

    def test_fir_pipeline(self) -> None:
        """Pipeline should work with FIR filters too."""
        signal = _make_test_signal()
        cfg = PreprocessingConfig(sampling_rate=SAMPLING_RATE, filter_type="fir")
        result = preprocess(signal, SAMPLING_RATE, cfg)
        assert len(result["filtered"]) == len(signal)

    def test_selective_stages(self) -> None:
        """Only enabled stages should affect the signal."""
        signal = _make_test_signal()
        # Only baseline removal
        cfg = PreprocessingConfig(
            sampling_rate=SAMPLING_RATE,
            enabled_stages=["baseline"],
        )
        result = preprocess(signal, SAMPLING_RATE, cfg)
        # baseline_removed != raw, but notch == baseline_removed (no change)
        assert not np.allclose(result["raw"], result["baseline_removed"])
        assert np.allclose(result["baseline_removed"], result["notch_filtered"])
        assert np.allclose(result["baseline_removed"], result["filtered"])

    def test_finite_output(self) -> None:
        """Pipeline output should have no NaN or inf."""
        signal = _make_test_signal()
        result = preprocess(signal, SAMPLING_RATE)
        for key, arr in result.items():
            assert np.all(np.isfinite(arr)), f"{key} contains NaN/inf"
