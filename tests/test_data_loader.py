"""Tests for the data loader and synthetic ECG generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

import numpy as np
import pytest

from src.data_loader import (
    generate_ecg_signal,
    load_csv,
    save_csv,
)

SAMPLING_RATE: int = 360


class TestGenerateECG:
    """Tests for the synthetic ECG signal generator."""

    def test_output_shapes(self) -> None:
        t, clean, noisy, sr = generate_ecg_signal(duration=10.0, sampling_rate=360)
        assert len(t) == len(clean) == len(noisy) == 3600
        assert sr == 360

    def test_clean_signal_is_periodic(self) -> None:
        """The clean signal should contain repeating R-peaks."""
        t, clean, _, sr = generate_ecg_signal(
            duration=5.0, sampling_rate=360, heart_rate=60, add_noise=False,
        )
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(clean, height=0.5, distance=int(sr * 0.5))
        # At 60 BPM over 5 s, we expect ~5 beats
        assert 4 <= len(peaks) <= 6, f"Expected ~5 beats, got {len(peaks)}"

    def test_noisy_signal_differs_from_clean(self) -> None:
        _, clean, noisy, _ = generate_ecg_signal(add_noise=True)
        assert not np.allclose(clean, noisy)

    def test_no_noise_option(self) -> None:
        _, clean, noisy, _ = generate_ecg_signal(add_noise=False)
        assert np.allclose(clean, noisy)

    def test_reproducibility(self) -> None:
        _, _, noisy1, _ = generate_ecg_signal(seed=42)
        _, _, noisy2, _ = generate_ecg_signal(seed=42)
        assert np.allclose(noisy1, noisy2)

    def test_reproducibility_different_seeds(self) -> None:
        _, _, noisy1, _ = generate_ecg_signal(seed=42)
        _, _, noisy2, _ = generate_ecg_signal(seed=99)
        assert not np.allclose(noisy1, noisy2)

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_ecg_signal(duration=-1)
        with pytest.raises(ValueError):
            generate_ecg_signal(duration=0)

    def test_invalid_heart_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_ecg_signal(heart_rate=0)
        with pytest.raises(ValueError):
            generate_ecg_signal(heart_rate=400)

    def test_default_parameters(self) -> None:
        t, clean, noisy, sr = generate_ecg_signal()
        assert sr == 360
        assert len(t) == 3600  # 10 s default
        assert np.all(np.isfinite(clean))
        assert np.all(np.isfinite(noisy))

    def test_custom_heart_rate(self) -> None:
        """Higher heart rate → more beats in the same duration."""
        t, clean_60, _, _ = generate_ecg_signal(duration=5.0, heart_rate=60, add_noise=False)
        _, clean_120, _, _ = generate_ecg_signal(duration=5.0, heart_rate=120, add_noise=False)
        from scipy.signal import find_peaks
        peaks_60, _ = find_peaks(clean_60, height=0.5, distance=50)
        peaks_120, _ = find_peaks(clean_120, height=0.5, distance=50)
        assert len(peaks_120) > len(peaks_60)


class TestCSVIO:
    """Tests for CSV file import / export."""

    def test_save_and_load_roundtrip(self) -> None:
        t = np.linspace(0, 1, 100, endpoint=False)
        signal = np.sin(2 * np.pi * 5 * t)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            save_csv(tmp_path, t, signal)
            t_loaded, sig_loaded, sr = load_csv(tmp_path)
            assert len(t_loaded) == len(sig_loaded) == len(signal)
            assert np.allclose(signal, sig_loaded, atol=1e-4)
            assert sr > 0
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv(Path("/nonexistent/ecg.csv"))
