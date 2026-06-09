"""Tests for the filter design and application module."""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.filters import (
    apply_fir_filter,
    apply_iir_filter,
    compute_frequency_response,
    compute_spectrum,
    design_fir_bandstop,
    design_fir_highpass,
    design_fir_lowpass,
    design_iir_highpass,
    design_iir_lowpass,
    design_iir_notch,
)

SAMPLING_RATE: int = 360

# ---------------------------------------------------------------------------
# FIR filter tests
# ---------------------------------------------------------------------------


class TestFIRFilters:
    """Tests for FIR filter design functions."""

    def test_highpass_coeffs_shape(self) -> None:
        taps = design_fir_highpass(cutoff=0.5, sampling_rate=SAMPLING_RATE, numtaps=101)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 101
        assert np.all(np.isfinite(taps))

    def test_lowpass_coeffs_shape(self) -> None:
        taps = design_fir_lowpass(cutoff=40.0, sampling_rate=SAMPLING_RATE, numtaps=101)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 101
        assert np.all(np.isfinite(taps))

    def test_bandstop_coeffs_shape(self) -> None:
        taps = design_fir_bandstop(center=50.0, sampling_rate=SAMPLING_RATE, numtaps=101)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 101
        assert np.all(np.isfinite(taps))

    def test_highpass_rejects_dc(self) -> None:
        """A high-pass filter should attenuate DC (0 Hz).

        With a 0.5 Hz cut-off at 360 Hz sampling, even a high-order FIR
        provides modest DC attenuation — this is a narrow-band design.
        We verify that DC is measurably lower than the pass-band.
        """
        taps = design_fir_highpass(0.5, SAMPLING_RATE, numtaps=151)
        freqs, mag_db = compute_frequency_response(taps, SAMPLING_RATE)
        # DC should be attenuated relative to pass-band (10 Hz)
        dc_idx = np.argmin(np.abs(freqs))
        pass_idx = np.argmin(np.abs(freqs - 10.0))
        assert mag_db[dc_idx] < mag_db[pass_idx], (
            f"DC ({mag_db[dc_idx]:.1f} dB) should be lower than "
            f"10 Hz ({mag_db[pass_idx]:.1f} dB)"
        )

    def test_highpass_passes_high_freq(self) -> None:
        """The high-pass should pass frequencies well above cut-off."""
        taps = design_fir_highpass(0.5, SAMPLING_RATE, numtaps=151)
        freqs, mag_db = compute_frequency_response(taps, SAMPLING_RATE)
        # Find magnitude at 10 Hz
        idx_10hz = np.argmin(np.abs(freqs - 10.0))
        assert mag_db[idx_10hz] > -3, f"10 Hz should pass well, got {mag_db[idx_10hz]:.1f} dB"

    def test_lowpass_rejects_nyquist(self) -> None:
        """A low-pass should strongly attenuate near Nyquist."""
        taps = design_fir_lowpass(40.0, SAMPLING_RATE, numtaps=151)
        freqs, mag_db = compute_frequency_response(taps, SAMPLING_RATE)
        idx_nyq = np.argmin(np.abs(freqs - SAMPLING_RATE / 2))
        assert mag_db[idx_nyq] < -30, f"Nyquist attenuation should be strong, got {mag_db[idx_nyq]:.1f} dB"

    def test_odd_taps_enforced(self) -> None:
        """Even numtaps should be bumped to odd."""
        taps = design_fir_highpass(0.5, SAMPLING_RATE, numtaps=100)  # even
        assert len(taps) % 2 == 1

    def test_invalid_cutoff_raises(self) -> None:
        with pytest.raises(ValueError):
            design_fir_highpass(0, SAMPLING_RATE)  # DC = invalid
        with pytest.raises(ValueError):
            design_fir_highpass(200, SAMPLING_RATE)  # > Nyquist


# ---------------------------------------------------------------------------
# IIR filter tests
# ---------------------------------------------------------------------------


class TestIIRFilters:
    """Tests for IIR (Butterworth) filter design."""

    def test_highpass_returns_ba(self) -> None:
        b, a = design_iir_highpass(0.5, SAMPLING_RATE)
        assert isinstance(b, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert len(b) > 0 and len(a) > 0
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))

    def test_lowpass_returns_ba(self) -> None:
        b, a = design_iir_lowpass(40.0, SAMPLING_RATE)
        assert isinstance(b, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))

    def test_notch_returns_ba(self) -> None:
        b, a = design_iir_notch(50.0, SAMPLING_RATE)
        assert isinstance(b, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))

    def test_notch_attenuates_at_center(self) -> None:
        b, a = design_iir_notch(50.0, SAMPLING_RATE, q=30)
        freqs, mag_db = compute_frequency_response((b, a), SAMPLING_RATE)
        idx = np.argmin(np.abs(freqs - 50.0))
        assert mag_db[idx] < -10, f"Notch should attenuate at 50 Hz, got {mag_db[idx]:.1f} dB"

    def test_highpass_stability(self) -> None:
        """IIR filter poles should be inside the unit circle."""
        b, a = design_iir_highpass(0.5, SAMPLING_RATE, order=4)
        poles = np.roots(a)
        assert np.all(np.abs(poles) < 1.0), f"Unstable poles: {poles}"


# ---------------------------------------------------------------------------
# Filter application tests
# ---------------------------------------------------------------------------


class TestFilterApplication:
    """Tests for filter application (filtfilt)."""

    def test_fir_output_same_length(self) -> None:
        taps = design_fir_highpass(0.5, SAMPLING_RATE, numtaps=101)
        data = np.random.default_rng(42).normal(0, 1, 1000)
        out = apply_fir_filter(data, taps)
        assert len(out) == len(data)

    def test_iir_output_same_length(self) -> None:
        b, a = design_iir_highpass(0.5, SAMPLING_RATE)
        data = np.random.default_rng(42).normal(0, 1, 1000)
        out = apply_iir_filter(data, b, a)
        assert len(out) == len(data)

    def test_filtered_signal_is_finite(self) -> None:
        """Output of filtering should be all-finite."""
        data = np.random.default_rng(42).normal(0, 1, 1000)
        taps = design_fir_lowpass(40.0, SAMPLING_RATE)
        out = apply_fir_filter(data, taps)
        assert np.all(np.isfinite(out))

    def test_filter_does_not_crash_on_short_signal(self) -> None:
        """Filtering very short signals should not raise (filtfilt padlen check)."""
        data = np.array([0.1, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0] * 20, dtype=float)
        taps = design_fir_highpass(0.5, SAMPLING_RATE, numtaps=31)
        out = apply_fir_filter(data, taps)
        assert len(out) == len(data)


# ---------------------------------------------------------------------------
# Spectrum helper tests
# ---------------------------------------------------------------------------


class TestSpectrum:
    """Tests for the FFT spectrum helper."""

    def test_spectrum_shape(self) -> None:
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 360))
        freqs, amp = compute_spectrum(data, 360)
        assert freqs.shape == amp.shape
        assert len(freqs) == 360 // 2 + 1

    def test_spectrum_peak_at_signal_freq(self) -> None:
        """A pure 10 Hz sine should peak near 10 Hz."""
        sr = 360
        freq = 10.0
        t = np.linspace(0, 1, sr, endpoint=False)
        data = np.sin(2 * np.pi * freq * t)
        freqs, amp = compute_spectrum(data, sr)
        peak_idx = np.argmax(amp)
        assert abs(freqs[peak_idx] - freq) < 1.0, \
            f"Expected peak near {freq} Hz, got {freqs[peak_idx]:.1f} Hz"


# ---------------------------------------------------------------------------
# Frequency response tests
# ---------------------------------------------------------------------------


class TestFrequencyResponse:
    """Tests for the frequency response computation."""

    def test_response_has_correct_shape(self) -> None:
        taps = design_fir_lowpass(40.0, SAMPLING_RATE)
        freqs, mag_db = compute_frequency_response(taps, SAMPLING_RATE, n_freqs=512)
        assert len(freqs) == 512
        assert len(mag_db) == 512

    def test_response_mag_is_negative_or_zero(self) -> None:
        """dB magnitude for a passive digital filter should be <= 0."""
        taps = design_fir_lowpass(40.0, SAMPLING_RATE)
        _, mag_db = compute_frequency_response(taps, SAMPLING_RATE)
        assert np.all(mag_db <= 1.0), f"Max magnitude: {mag_db.max():.1f} dB"
