"""Visualisation module for ECG signal processing results.

Produces publication-quality plots:
- Raw vs. filtered signal comparison (time domain)
- Filter frequency response curves (magnitude + phase)
- Power spectrum (FFT) before and after filtering
- Multi-stage processing overview

All figures are saved to a configurable output directory (default: ``results/``).
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import filters as filt_module

# Use a non-interactive backend when running headless (e.g., CI, Streamlit)
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Colour palette (colour-blind friendly)
COLOR_RAW: str = "#7f7f7f"       # grey — raw noisy signal
COLOR_CLEAN: str = "#1f77b4"     # blue — clean / reference
COLOR_FILTERED: str = "#d62728"  # red — filtered output
COLOR_STAGE1: str = "#ff7f0e"    # orange — after baseline removal
COLOR_STAGE2: str = "#2ca02c"    # green — after notch

# Figure defaults
FIG_DPI: int = 150
FIG_SIZE: Tuple[float, float] = (12, 5)
FIG_SIZE_SQUARE: Tuple[float, float] = (8, 6)

# Default output directory
RESULTS_DIR: Path = Path("results")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ensure_dir(save_path: Optional[Path]) -> Optional[Path]:
    """Create parent directories if a save path is given."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def _save_or_show(fig: plt.Figure, save_path: Optional[Path]) -> None:
    """Save figure to disk and close it."""
    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Time-domain signal comparison
# ---------------------------------------------------------------------------

def plot_signal_comparison(
    t: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    title: str = "ECG Signal: Raw vs. Filtered",
    save_path: Optional[Path] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot raw and filtered ECG signals overlaid in the time domain.

    Args:
        t: Time array (seconds).
        raw: Raw (noisy) ECG signal.
        filtered: Processed (clean) ECG signal.
        title: Figure title.
        save_path: If given, save the figure to this path (PNG).
        time_range: Optional (t_min, t_max) to zoom in.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    ax.plot(t, raw, color=COLOR_RAW, alpha=0.7, linewidth=0.8, label="Raw (noisy)")
    ax.plot(t, filtered, color=COLOR_FILTERED, linewidth=1.2, label="Filtered")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (mV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if time_range is not None:
        ax.set_xlim(time_range)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Multi-stage processing overview
# ---------------------------------------------------------------------------

def plot_processing_stages(
    t: np.ndarray,
    raw: np.ndarray,
    baseline_removed: np.ndarray,
    notch_filtered: np.ndarray,
    filtered: np.ndarray,
    title: str = "ECG Processing Stages",
    save_path: Optional[Path] = None,
    time_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Show the ECG signal after each preprocessing stage.

    Four sub-panels: Raw, After High-Pass, After Notch, After Low-Pass.

    Args:
        t: Time array.
        raw: Original noisy signal.
        baseline_removed: After baseline wander removal.
        notch_filtered: After power-line notch.
        filtered: Final output (after low-pass).
        title: Overall figure title.
        save_path: Optional save path.
        time_range: Optional (t_min, t_max) zoom.

    Returns:
        The Figure.
    """
    stages: list[tuple[str, np.ndarray, str]] = [
        ("1. Raw (Noisy)", raw, COLOR_RAW),
        ("2. Baseline Removed (HP 0.5 Hz)", baseline_removed, COLOR_STAGE1),
        ("3. Notch Filtered (50 Hz)", notch_filtered, COLOR_STAGE2),
        ("4. Low-Pass Filtered (40 Hz)", filtered, COLOR_FILTERED),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=FIG_DPI, sharex=True)

    for ax, (label, data, color) in zip(axes, stages):
        ax.plot(t, data, color=color, linewidth=0.8)
        ax.set_ylabel("mV", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if time_range is not None:
            ax.set_xlim(time_range)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Frequency response of filters
# ---------------------------------------------------------------------------

def plot_filter_response(
    sampling_rate: int = 360,
    filter_type: str = "iir",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot the frequency response (magnitude) of all three filters.

    Shows high-pass (0.5 Hz), notch (50 Hz), and low-pass (40 Hz) responses
    on a single axes.

    Args:
        sampling_rate: Sampling rate in Hz.
        filter_type: ``"fir"`` or ``"iir"``.
        save_path: Optional save path for the figure.

    Returns:
        The Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=FIG_DPI)

    configs = [
        ("High-Pass (0.5 Hz)", "highpass", 0.5, "#1f77b4"),
        ("Notch (50 Hz)", "notch", 50.0, "#ff7f0e"),
        ("Low-Pass (40 Hz)", "lowpass", 40.0, "#2ca02c"),
    ]

    for ax, (title, ftype, cutoff, color) in zip(axes, configs):
        if filter_type == "fir":
            if ftype == "highpass":
                coeffs = filt_module.design_fir_highpass(cutoff, sampling_rate)
            elif ftype == "notch":
                coeffs = filt_module.design_fir_bandstop(cutoff, sampling_rate)
            else:
                coeffs = filt_module.design_fir_lowpass(cutoff, sampling_rate)
        else:
            if ftype == "highpass":
                coeffs = filt_module.design_iir_highpass(cutoff, sampling_rate)
            elif ftype == "notch":
                coeffs = filt_module.design_iir_notch(cutoff, sampling_rate)
            else:
                coeffs = filt_module.design_iir_lowpass(cutoff, sampling_rate)

        freqs, mag_db = filt_module.compute_frequency_response(coeffs, sampling_rate)
        ax.plot(freqs, mag_db, color=color, linewidth=1.5)
        ax.axvline(cutoff, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"{cutoff} Hz")
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Magnitude (dB)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(-60, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Filter Frequency Responses ({filter_type.upper()})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Spectrum (FFT) comparison
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(
    raw: np.ndarray,
    filtered: np.ndarray,
    sampling_rate: int = 360,
    title: str = "ECG Power Spectrum: Before vs. After Filtering",
    save_path: Optional[Path] = None,
    max_freq: float = 80.0,
) -> plt.Figure:
    """Compare the power spectra of raw and filtered signals.

    Args:
        raw: Raw (noisy) ECG signal.
        filtered: Processed ECG signal.
        sampling_rate: Sampling rate in Hz.
        title: Figure title.
        save_path: Optional save path.
        max_freq: Maximum frequency to display (Hz).

    Returns:
        The Figure.
    """
    freqs_raw, amp_raw = filt_module.compute_spectrum(raw, sampling_rate)
    freqs_filt, amp_filt = filt_module.compute_spectrum(filtered, sampling_rate)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    ax.plot(freqs_raw, amp_raw, color=COLOR_RAW, alpha=0.7, linewidth=0.8,
            label="Raw (noisy)")
    ax.plot(freqs_filt, amp_filt, color=COLOR_FILTERED, linewidth=1.2,
            label="Filtered")

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Amplitude (mV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, max_freq)
    ax.grid(True, alpha=0.3)

    # Annotate key landmarks
    ax.axvline(0.5, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.5,
               label="0.5 Hz cut-off")
    ax.axvline(50.0, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.5,
               label="50 Hz notch")
    ax.axvline(40.0, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.5,
               label="40 Hz cut-off")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Comprehensive overview figure
# ---------------------------------------------------------------------------

def plot_comprehensive_overview(
    t: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    baseline_removed: np.ndarray,
    notch_filtered: np.ndarray,
    sampling_rate: int = 360,
    save_path: Optional[Path] = None,
    time_range: Optional[Tuple[float, float]] = (0, 2.5),
    filter_type: str = "iir",
) -> plt.Figure:
    """Generate a single comprehensive figure with four panels:

    (A) Time-domain raw vs filtered
    (B) Processing stages (zoomed)
    (C) Frequency response of filters
    (D) Spectrum before/after

    Args:
        t: Time array.
        raw: Raw signal.
        filtered: Final filtered signal.
        baseline_removed: After HP filter.
        notch_filtered: After notch filter.
        sampling_rate: Sampling rate in Hz.
        save_path: Optional save path.
        time_range: Zoom range for panels A and B.
        filter_type: ``"fir"`` or ``"iir"``.

    Returns:
        The Figure.
    """
    fig = plt.figure(figsize=(16, 12), dpi=FIG_DPI)

    # (A) Top-left: Time domain overlay
    ax_a = fig.add_subplot(2, 2, 1)
    ax_a.plot(t, raw, color=COLOR_RAW, alpha=0.7, linewidth=0.8, label="Raw")
    ax_a.plot(t, filtered, color=COLOR_FILTERED, linewidth=1.2, label="Filtered")
    if time_range:
        ax_a.set_xlim(time_range)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Amplitude (mV)")
    ax_a.set_title("A. Raw vs. Filtered (Zoomed)")
    ax_a.legend(fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # (B) Top-right: Stages
    ax_b = fig.add_subplot(2, 2, 2)
    ax_b.plot(t, raw, color=COLOR_RAW, alpha=0.5, linewidth=0.6, label="Raw")
    ax_b.plot(t, baseline_removed, color=COLOR_STAGE1, linewidth=1.0, label="After HP")
    ax_b.plot(t, notch_filtered, color=COLOR_STAGE2, linewidth=1.0, label="After Notch")
    ax_b.plot(t, filtered, color=COLOR_FILTERED, linewidth=1.2, label="After LP")
    if time_range:
        ax_b.set_xlim(time_range)
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Amplitude (mV)")
    ax_b.set_title("B. Processing Stages")
    ax_b.legend(fontsize=7)
    ax_b.grid(True, alpha=0.3)

    # (C) Bottom-left: Filter responses
    ax_c = fig.add_subplot(2, 2, 3)
    for ftype, cutoff, color, label in [
        ("highpass", 0.5, "#1f77b4", "HP 0.5 Hz"),
        ("lowpass", 40.0, "#2ca02c", "LP 40 Hz"),
    ]:
        if filter_type == "fir":
            coeffs = (filt_module.design_fir_highpass if ftype == "highpass"
                      else filt_module.design_fir_lowpass)(cutoff, sampling_rate)
        else:
            coeffs = (filt_module.design_iir_highpass if ftype == "highpass"
                      else filt_module.design_iir_lowpass)(cutoff, sampling_rate)
        freqs, mag_db = filt_module.compute_frequency_response(coeffs, sampling_rate)
        ax_c.plot(freqs, mag_db, color=color, linewidth=1.2, label=label)

    # Add notch separately
    if filter_type == "fir":
        notch_coeffs = filt_module.design_fir_bandstop(50.0, sampling_rate)
    else:
        notch_coeffs = filt_module.design_iir_notch(50.0, sampling_rate)
    freqs_n, mag_n = filt_module.compute_frequency_response(notch_coeffs, sampling_rate)
    ax_c.plot(freqs_n, mag_n, color="#ff7f0e", linewidth=1.2, label="Notch 50 Hz")

    ax_c.set_xlabel("Frequency (Hz)")
    ax_c.set_ylabel("Magnitude (dB)")
    ax_c.set_title(f"C. Filter Responses ({filter_type.upper()})")
    ax_c.set_ylim(-60, 5)
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # (D) Bottom-right: Spectrum
    ax_d = fig.add_subplot(2, 2, 4)
    freqs_raw, amp_raw = filt_module.compute_spectrum(raw, sampling_rate)
    freqs_filt, amp_filt = filt_module.compute_spectrum(filtered, sampling_rate)
    ax_d.plot(freqs_raw, amp_raw, color=COLOR_RAW, alpha=0.7, linewidth=0.8, label="Raw")
    ax_d.plot(freqs_filt, amp_filt, color=COLOR_FILTERED, linewidth=1.2, label="Filtered")
    ax_d.set_xlabel("Frequency (Hz)")
    ax_d.set_ylabel("Amplitude (mV)")
    ax_d.set_title("D. Power Spectrum")
    ax_d.set_xlim(0, 80)
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    fig.suptitle("ECG Signal Processing — Comprehensive Overview",
                 fontsize=15, fontweight="bold", y=0.998)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig
