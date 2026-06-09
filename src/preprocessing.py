"""ECG signal preprocessing pipeline.

Implements three essential denoising stages for clinical ECG analysis:

1. **Baseline wander removal** – high-pass filter (0.5 Hz cut-off) to remove
   low-frequency drift caused by respiration and electrode motion.
2. **Power-line notch** – band-stop filter at 50 / 60 Hz to suppress mains
   interference.
3. **High-frequency noise reduction** – low-pass filter (40 Hz cut-off) to
   attenuate myographic (muscle) noise and other high-frequency artifacts.

All stages use zero-phase filtering (``scipy.signal.filtfilt``) to avoid
distorting ECG waveform morphology.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import filters as filt_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Configuration for the ECG preprocessing pipeline.

    Attributes:
        sampling_rate: Signal sampling rate in Hz.
        filter_type: ``"fir"`` or ``"iir"`` — which filter family to use.
        lowcut: High-pass cut-off for baseline wander removal (Hz).
        highcut: Low-pass cut-off for high-frequency noise (Hz).
        notch_freq: Mains frequency to notch out (Hz).
        fir_taps: Number of taps for FIR filters.
        iir_order: Order for IIR filters.
        enabled_stages: Which stages to apply; subset of
            {``"baseline"``, ``"notch"``, ``"lowpass"``}.
    """

    sampling_rate: int = 360
    filter_type: str = "iir"
    lowcut: float = 0.5
    highcut: float = 40.0
    notch_freq: float = 50.0
    fir_taps: int = 101
    iir_order: int = 4
    enabled_stages: list[str] = field(
        default_factory=lambda: ["baseline", "notch", "lowpass"]
    )

    def __post_init__(self) -> None:
        valid_types = {"fir", "iir"}
        if self.filter_type not in valid_types:
            raise ValueError(f"filter_type must be one of {valid_types}")
        valid_stages = {"baseline", "notch", "lowpass"}
        for stage in self.enabled_stages:
            if stage not in valid_stages:
                raise ValueError(f"Unknown stage '{stage}'; valid: {valid_stages}")


# ---------------------------------------------------------------------------
# Filter application helpers
# ---------------------------------------------------------------------------

def _get_filter_applier(filter_type: str) -> Callable:
    """Return the appropriate ``apply_*`` function for the filter type."""
    if filter_type == "fir":
        return filt_module.apply_fir_filter
    return filt_module.apply_iir_filter


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def remove_baseline_wander(
    data: np.ndarray,
    sampling_rate: int,
    filter_type: str = "iir",
    fir_taps: int = 101,
    iir_order: int = 4,
    lowcut: float = 0.5,
) -> np.ndarray:
    """Remove baseline wander using a high-pass filter.

    Args:
        data: Raw ECG signal.
        sampling_rate: Sampling rate in Hz.
        filter_type: ``"fir"`` or ``"iir"``.
        fir_taps: FIR filter order (only for ``filter_type="fir"``).
        iir_order: IIR filter order (only for ``filter_type="iir"``).
        lowcut: High-pass cut-off frequency (Hz).

    Returns:
        Signal with baseline wander removed.
    """
    apply_fn = _get_filter_applier(filter_type)

    if filter_type == "fir":
        coeffs = filt_module.design_fir_highpass(lowcut, sampling_rate, fir_taps)
        filtered = apply_fn(data, coeffs)
    else:
        b, a = filt_module.design_iir_highpass(lowcut, sampling_rate, iir_order)
        filtered = apply_fn(data, b, a)

    logger.info("Baseline wander removed (HP %.1f Hz, %s)", lowcut, filter_type.upper())
    return filtered  # type: ignore[no-any-return]


def remove_powerline_noise(
    data: np.ndarray,
    sampling_rate: int,
    notch_freq: float = 50.0,
    filter_type: str = "iir",
    fir_taps: int = 101,
) -> np.ndarray:
    """Remove power-line interference with a notch (band-stop) filter.

    Args:
        data: ECG signal.
        sampling_rate: Sampling rate in Hz.
        notch_freq: Mains frequency (50 or 60 Hz).
        filter_type: ``"fir"`` or ``"iir"``.
        fir_taps: FIR filter order.

    Returns:
        Signal with power-line interference attenuated.
    """
    apply_fn = _get_filter_applier(filter_type)

    if filter_type == "fir":
        coeffs = filt_module.design_fir_bandstop(notch_freq, sampling_rate, fir_taps)
        filtered = apply_fn(data, coeffs)
    else:
        b, a = filt_module.design_iir_notch(notch_freq, sampling_rate)
        filtered = apply_fn(data, b, a)

    logger.info("Power-line noise removed (notch %.1f Hz, %s)", notch_freq, filter_type.upper())
    return filtered  # type: ignore[no-any-return]


def remove_high_freq_noise(
    data: np.ndarray,
    sampling_rate: int,
    filter_type: str = "iir",
    fir_taps: int = 101,
    iir_order: int = 4,
    highcut: float = 40.0,
) -> np.ndarray:
    """Attenuate high-frequency noise (EMG) with a low-pass filter.

    Args:
        data: ECG signal.
        sampling_rate: Sampling rate in Hz.
        filter_type: ``"fir"`` or ``"iir"``.
        fir_taps: FIR filter order.
        iir_order: IIR filter order.
        highcut: Low-pass cut-off frequency (Hz).

    Returns:
        Low-pass filtered signal.
    """
    apply_fn = _get_filter_applier(filter_type)

    if filter_type == "fir":
        coeffs = filt_module.design_fir_lowpass(highcut, sampling_rate, fir_taps)
        filtered = apply_fn(data, coeffs)
    else:
        b, a = filt_module.design_iir_lowpass(highcut, sampling_rate, iir_order)
        filtered = apply_fn(data, b, a)

    logger.info("High-frequency noise removed (LP %.1f Hz, %s)", highcut, filter_type.upper())
    return filtered  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    data: np.ndarray,
    sampling_rate: int,
    config: Optional[PreprocessingConfig] = None,
) -> dict[str, np.ndarray]:
    """Run the full ECG preprocessing pipeline.

    Stages are applied in order: baseline wander → notch → low-pass.
    Intermediate results are returned so they can be inspected.

    Args:
        data: Raw ECG signal (1-D array).
        sampling_rate: Sampling rate in Hz.
        config: Pipeline configuration.  Uses defaults if ``None``.

    Returns:
        A dictionary with keys:
        - ``"raw"``: original input signal
        - ``"baseline_removed"``: after high-pass filter
        - ``"notch_filtered"``: after power-line notch
        - ``"filtered"``: final output after low-pass
    """
    if config is None:
        config = PreprocessingConfig(sampling_rate=sampling_rate)

    results: dict[str, np.ndarray] = {"raw": data.copy()}
    current: np.ndarray = data.copy()

    # Stage 1: Baseline wander
    if "baseline" in config.enabled_stages:
        current = remove_baseline_wander(
            current, sampling_rate,
            filter_type=config.filter_type,
            fir_taps=config.fir_taps,
            iir_order=config.iir_order,
            lowcut=config.lowcut,
        )
    results["baseline_removed"] = current.copy()

    # Stage 2: Power-line notch
    if "notch" in config.enabled_stages:
        current = remove_powerline_noise(
            current, sampling_rate,
            notch_freq=config.notch_freq,
            filter_type=config.filter_type,
            fir_taps=config.fir_taps,
        )
    results["notch_filtered"] = current.copy()

    # Stage 3: High-frequency noise
    if "lowpass" in config.enabled_stages:
        current = remove_high_freq_noise(
            current, sampling_rate,
            filter_type=config.filter_type,
            fir_taps=config.fir_taps,
            iir_order=config.iir_order,
            highcut=config.highcut,
        )
    results["filtered"] = current.copy()

    logger.info("Preprocessing pipeline complete (%s)", config.filter_type.upper())
    return results
