"""Digital filter design module for ECG signal processing.

Provides FIR (window method) and IIR (Butterworth) filter design
with zero-phase filtering (filtfilt).  Also includes frequency-
response visualisation utilities.

Design principles:
    - Use ``scipy.signal.filtfilt`` for zero-phase (forward-backward) filtering
      to preserve ECG waveform morphology.
    - FIR filters have linear phase but higher order; IIR filters are more
      efficient with non-linear phase (corrected by filtfilt).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.signal as signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default filter order
DEFAULT_FIR_ORDER: int = 101   # FIR filter taps (must be odd for bandpass etc.)
DEFAULT_IIR_ORDER: int = 4     # IIR filter order

# Cut-off frequencies in Hz
DEFAULT_LOWCUT: float = 0.5    # High-pass cut-off for baseline removal
DEFAULT_HIGHCUT: float = 40.0  # Low-pass cut-off for EMG removal
DEFAULT_NOTCH: float = 50.0    # Notch centre for power-line (50 or 60)
DEFAULT_NOTCH_Q: float = 30.0  # Quality factor for notch


# ---------------------------------------------------------------------------
# FIR filter design
# ---------------------------------------------------------------------------

def design_fir_highpass(
    cutoff: float = DEFAULT_LOWCUT,
    sampling_rate: int = 360,
    numtaps: int = DEFAULT_FIR_ORDER,
    window: str = "hamming",
) -> np.ndarray:
    """Design a linear-phase FIR high-pass filter using the window method.

    Args:
        cutoff: Cut-off frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        numtaps: Number of filter taps (must be odd for type I).
        window: Window function name (passed to ``scipy.signal.firwin``).

    Returns:
        Array of filter coefficients (length ``numtaps``).

    Raises:
        ValueError: If ``cutoff`` is not in (0, Nyquist).
    """
    nyquist: float = sampling_rate / 2.0
    if not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff {cutoff} Hz must be in (0, {nyquist})")

    numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1  # Ensure odd
    taps: np.ndarray = signal.firwin(
        numtaps, cutoff, pass_zero="highpass", fs=sampling_rate, window=window,
    )
    logger.debug("FIR high-pass: cutoff=%.1f Hz, taps=%d, window=%s", cutoff, numtaps, window)
    return taps


def design_fir_lowpass(
    cutoff: float = DEFAULT_HIGHCUT,
    sampling_rate: int = 360,
    numtaps: int = DEFAULT_FIR_ORDER,
    window: str = "hamming",
) -> np.ndarray:
    """Design a linear-phase FIR low-pass filter.

    Args:
        cutoff: Cut-off frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        numtaps: Number of filter taps.
        window: Window function name.

    Returns:
        Array of filter coefficients.
    """
    nyquist: float = sampling_rate / 2.0
    if not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff {cutoff} Hz must be in (0, {nyquist})")

    numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1
    taps: np.ndarray = signal.firwin(
        numtaps, cutoff, pass_zero="lowpass", fs=sampling_rate, window=window,
    )
    logger.debug("FIR low-pass: cutoff=%.1f Hz, taps=%d", cutoff, numtaps)
    return taps


def design_fir_bandstop(
    center: float = DEFAULT_NOTCH,
    sampling_rate: int = 360,
    numtaps: int = DEFAULT_FIR_ORDER,
    width: float = 2.0,
    window: str = "hamming",
) -> np.ndarray:
    """Design an FIR band-stop (notch) filter.

    Args:
        center: Centre frequency in Hz (e.g., 50 or 60).
        sampling_rate: Sampling rate in Hz.
        numtaps: Number of filter taps.
        width: Stop-band width in Hz.
        window: Window function name.

    Returns:
        Array of filter coefficients.
    """
    nyquist: float = sampling_rate / 2.0
    low: float = (center - width / 2) / nyquist
    high: float = (center + width / 2) / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"Band-stop edges [{low*nyquist:.1f}, {high*nyquist:.1f}] out of range")

    numtaps = numtaps if numtaps % 2 == 1 else numtaps + 1
    taps: np.ndarray = signal.firwin(
        numtaps, [low, high], pass_zero="bandstop", window=window,
    )
    logger.debug("FIR band-stop: %.1f Hz ± %.1f Hz, taps=%d", center, width / 2, numtaps)
    return taps


# ---------------------------------------------------------------------------
# IIR filter design (Butterworth)
# ---------------------------------------------------------------------------

def design_iir_highpass(
    cutoff: float = DEFAULT_LOWCUT,
    sampling_rate: int = 360,
    order: int = DEFAULT_IIR_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth IIR high-pass filter.

    Args:
        cutoff: Cut-off frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Tuple (b, a) of numerator / denominator coefficients.
    """
    nyquist: float = sampling_rate / 2.0
    wn: float = cutoff / nyquist
    if not 0 < wn < 1:
        raise ValueError(f"Normalised cut-off {wn} must be in (0, 1)")
    b, a = signal.butter(order, wn, btype="high", analog=False)
    logger.debug("IIR high-pass: order=%d, cutoff=%.1f Hz", order, cutoff)
    return b, a


def design_iir_lowpass(
    cutoff: float = DEFAULT_HIGHCUT,
    sampling_rate: int = 360,
    order: int = DEFAULT_IIR_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth IIR low-pass filter.

    Args:
        cutoff: Cut-off frequency in Hz.
        sampling_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Tuple (b, a) of coefficients.
    """
    nyquist: float = sampling_rate / 2.0
    wn: float = cutoff / nyquist
    if not 0 < wn < 1:
        raise ValueError(f"Normalised cut-off {wn} must be in (0, 1)")
    b, a = signal.butter(order, wn, btype="low", analog=False)
    logger.debug("IIR low-pass: order=%d, cutoff=%.1f Hz", order, cutoff)
    return b, a


def design_iir_notch(
    center: float = DEFAULT_NOTCH,
    sampling_rate: int = 360,
    q: float = DEFAULT_NOTCH_Q,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design an IIR notch (band-stop) filter using a second-order section.

    Args:
        center: Centre frequency in Hz (e.g., 50).
        sampling_rate: Sampling rate in Hz.
        q: Quality factor (higher = narrower notch).

    Returns:
        Tuple (b, a) of coefficients.
    """
    nyquist: float = sampling_rate / 2.0
    w0: float = center / nyquist
    if not 0 < w0 < 1:
        raise ValueError(f"Normalised centre {w0} must be in (0, 1)")
    b, a = signal.iirnotch(w0, q)
    logger.debug("IIR notch: center=%.1f Hz, Q=%.1f", center, q)
    return b, a


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_fir_filter(
    data: np.ndarray,
    taps: np.ndarray,
) -> np.ndarray:
    """Apply an FIR filter with zero-phase (forward-backward) filtering.

    Args:
        data: 1-D input signal.
        taps: FIR filter coefficients.

    Returns:
        Filtered signal (same length as input).
    """
    return signal.filtfilt(taps, 1.0, data)  # type: ignore[no-any-return]


def apply_iir_filter(
    data: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply an IIR filter with zero-phase (filtfilt).

    Args:
        data: 1-D input signal.
        b: Numerator coefficients.
        a: Denominator coefficients.

    Returns:
        Filtered signal.
    """
    return signal.filtfilt(b, a, data)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------

def compute_frequency_response(
    taps_or_ba: np.ndarray | Tuple[np.ndarray, np.ndarray],
    sampling_rate: int = 360,
    n_freqs: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frequency response of a filter.

    Args:
        taps_or_ba: FIR taps array, or (b, a) tuple for IIR.
        sampling_rate: Sampling rate in Hz.
        n_freqs: Number of frequency points.

    Returns:
        Tuple (freqs, magnitude_dB) — frequencies in Hz, magnitude in dB.
    """
    if isinstance(taps_or_ba, tuple):
        b, a = taps_or_ba
        w, h = signal.freqz(b, a, worN=n_freqs)
    else:
        w, h = signal.freqz(taps_or_ba, 1.0, worN=n_freqs)

    freqs: np.ndarray = w * sampling_rate / (2 * np.pi)
    mag_db: np.ndarray = 20 * np.log10(np.abs(h) + 1e-12)
    return freqs, mag_db


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def compute_spectrum(
    data: np.ndarray,
    sampling_rate: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided amplitude spectrum via FFT.

    Args:
        data: 1-D time-domain signal.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Tuple (freqs, amplitude) where amplitude is in the same units as
        the input.
    """
    n: int = len(data)
    fft_vals: np.ndarray = np.fft.rfft(data)
    freqs: np.ndarray = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    amplitude: np.ndarray = np.abs(fft_vals) / n * 2.0
    return freqs, amplitude
