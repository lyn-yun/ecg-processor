"""ECG data loading and synthetic signal generation module.

Supports:
    - Loading ECG data from CSV files (.csv)
    - Loading PhysioNet-compatible WFDB format (.dat/.hea)
    - Generating realistic synthetic ECG signals with configurable parameters
      for demonstration and testing purposes.
"""

import csv
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (no magic numbers)
# ---------------------------------------------------------------------------

# Default sampling parameters
DEFAULT_SAMPLING_RATE: int = 360  # Hz (standard for MIT-BIH)
DEFAULT_DURATION: float = 10.0    # seconds
DEFAULT_HEART_RATE: int = 72      # bpm

# Synthetic ECG waveform shape parameters
ECG_P_WAVE_AMPLITUDE: float = 0.15   # mV
ECG_Q_WAVE_AMPLITUDE: float = -0.08  # mV
ECG_R_WAVE_AMPLITUDE: float = 1.0    # mV
ECG_S_WAVE_AMPLITUDE: float = -0.3   # mV
ECG_T_WAVE_AMPLITUDE: float = 0.25   # mV

# Noise parameters
BASELINE_WANDER_FREQ: float = 0.3     # Hz
BASELINE_WANDER_AMPLITUDE: float = 0.15  # mV
POWER_LINE_NOISE_FREQ: float = 50.0   # Hz (China/Europe: 50, US: 60)
POWER_LINE_NOISE_AMPLITUDE: float = 0.05  # mV
EMG_NOISE_STD: float = 0.03            # mV (Gaussian muscle noise)


# ---------------------------------------------------------------------------
# Synthetic ECG generation
# ---------------------------------------------------------------------------

def _generate_single_beat(
    t_beat: np.ndarray,
    beat_duration: float,
) -> np.ndarray:
    """Generate a single cardiac cycle (P-Q-R-S-T complex).

    Uses Gaussian-like approximations to model each wave component.

    Args:
        t_beat: Time array for this beat (seconds, 0 to beat_duration).
        beat_duration: Duration of one beat in seconds.

    Returns:
        ECG voltage values (mV) for one beat.
    """
    beat: np.ndarray = np.zeros_like(t_beat)

    # Normalised position within the beat [0, 1]
    tau: np.ndarray = t_beat / beat_duration

    # --- P wave (atrial depolarisation) at ~0.1 ---
    p_center: float = 0.10
    p_width: float = 0.03
    beat += ECG_P_WAVE_AMPLITUDE * np.exp(-0.5 * ((tau - p_center) / p_width) ** 2)

    # --- Q wave (septal depolarisation) at ~0.16 ---
    q_center: float = 0.16
    q_width: float = 0.01
    beat += ECG_Q_WAVE_AMPLITUDE * np.exp(-0.5 * ((tau - q_center) / q_width) ** 2)

    # --- R wave (ventricular depolarisation) at ~0.20 ---
    r_center: float = 0.20
    r_width: float = 0.012
    beat += ECG_R_WAVE_AMPLITUDE * np.exp(-0.5 * ((tau - r_center) / r_width) ** 2)

    # --- S wave (terminal depolarisation) at ~0.25 ---
    s_center: float = 0.25
    s_width: float = 0.015
    beat += ECG_S_WAVE_AMPLITUDE * np.exp(-0.5 * ((tau - s_center) / s_width) ** 2)

    # --- T wave (ventricular repolarisation) at ~0.45 ---
    t_center: float = 0.45
    t_width: float = 0.06
    beat += ECG_T_WAVE_AMPLITUDE * np.exp(-0.5 * ((tau - t_center) / t_width) ** 2)

    return beat


def _add_baseline_wander(
    signal: np.ndarray,
    t: np.ndarray,
    amplitude: float = BASELINE_WANDER_AMPLITUDE,
    freq: float = BASELINE_WANDER_FREQ,
) -> np.ndarray:
    """Add low-frequency baseline wander to a signal.

    Args:
        signal: Clean ECG signal.
        t: Time array (seconds).
        amplitude: Wander amplitude in mV.
        freq: Wander frequency in Hz.

    Returns:
        Signal with added baseline wander.
    """
    wander: np.ndarray = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + wander


def _add_power_line_noise(
    signal: np.ndarray,
    t: np.ndarray,
    amplitude: float = POWER_LINE_NOISE_AMPLITUDE,
    freq: float = POWER_LINE_NOISE_FREQ,
) -> np.ndarray:
    """Add power-line (mains) interference.

    Args:
        signal: ECG signal.
        t: Time array (seconds).
        amplitude: Noise amplitude in mV.
        freq: Mains frequency (50 or 60 Hz).

    Returns:
        Signal with added power-line interference.
    """
    noise: np.ndarray = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + noise


def _add_emg_noise(
    signal: np.ndarray,
    std: float = EMG_NOISE_STD,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add high-frequency myographic (muscle) noise as Gaussian random noise.

    Args:
        signal: ECG signal.
        std: Standard deviation of Gaussian noise in mV.
        rng: NumPy random generator for reproducibility.

    Returns:
        Signal with added EMG noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise: np.ndarray = rng.normal(0, std, size=signal.shape)
    return signal + noise


def generate_ecg_signal(
    duration: float = DEFAULT_DURATION,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    heart_rate: int = DEFAULT_HEART_RATE,
    add_noise: bool = True,
    power_line_freq: float = POWER_LINE_NOISE_FREQ,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realistic synthetic ECG signal with optional noise.

    The signal is built by repeating a single P-Q-R-S-T beat at the given
    heart rate.  Optional noise components (baseline wander, power-line
    interference, EMG) are added for realism.

    Args:
        duration: Signal duration in seconds.
        sampling_rate: Sampling frequency in Hz.
        heart_rate: Heart rate in beats per minute.
        add_noise: If True, add realistic noise sources.
        power_line_freq: Mains frequency in Hz (50 or 60).
        seed: Random seed for reproducible noise.

    Returns:
        Tuple of (t, clean_ecg, noisy_ecg, sampling_rate), where:
            t          – time array (seconds)
            clean_ecg  – clean ECG signal (mV)
            noisy_ecg  – ECG signal with noise (mV)

    Raises:
        ValueError: If any parameter is out of its valid range.
    """
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    if heart_rate <= 0 or heart_rate > 300:
        raise ValueError(f"heart_rate must be in (0, 300], got {heart_rate}")

    rng: np.random.Generator = np.random.default_rng(seed)

    # Time axis
    n_samples: int = int(duration * sampling_rate)
    t: np.ndarray = np.linspace(0, duration, n_samples, endpoint=False)
    dt: float = 1.0 / sampling_rate

    # Beat timing
    beat_period: float = 60.0 / heart_rate  # seconds per beat
    n_beats: int = int(np.ceil(duration / beat_period)) + 1

    # Generate the clean signal by stitching beats together
    clean_ecg: np.ndarray = np.zeros(n_samples, dtype=np.float64)
    for beat_idx in range(n_beats):
        beat_start: float = beat_idx * beat_period
        start_sample: int = int(beat_start * sampling_rate)
        end_sample: int = min(
            int((beat_start + beat_period) * sampling_rate),
            n_samples,
        )
        if start_sample >= n_samples:
            break

        n_beat_samples: int = end_sample - start_sample
        t_beat: np.ndarray = np.linspace(0, beat_period, n_beat_samples, endpoint=False)
        beat_wave: np.ndarray = _generate_single_beat(t_beat, beat_period)
        clean_ecg[start_sample:end_sample] = beat_wave

    # Add noise layers
    if add_noise:
        noisy_ecg: np.ndarray = clean_ecg.copy()
        noisy_ecg = _add_baseline_wander(noisy_ecg, t)
        noisy_ecg = _add_power_line_noise(noisy_ecg, t, freq=power_line_freq)
        noisy_ecg = _add_emg_noise(noisy_ecg, rng=rng)
    else:
        noisy_ecg = clean_ecg.copy()

    logger.info(
        "Generated synthetic ECG: %.1f s, %d Hz, %d BPM%s",
        duration, sampling_rate, heart_rate,
        " (with noise)" if add_noise else "",
    )
    return t, clean_ecg, noisy_ecg, sampling_rate


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_csv(
    filepath: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load ECG signal from a CSV file.

    Expected format: two columns — time (s) and amplitude (mV).
    If only one column is present, time is inferred from a default
    sampling rate.

    Args:
        filepath: Path to a .csv file.

    Returns:
        Tuple (t, signal, sampling_rate).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV format is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    data: list[list[str]] = []
    first_row_skipped: bool = False
    with open(filepath, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            # Skip empty / comment / header lines
            if not row or row[0].startswith("#"):
                continue
            # Detect and skip a text header row (e.g., "time_s,amplitude_mv")
            if not first_row_skipped:
                first_row_skipped = True
                try:
                    float(row[0])
                except ValueError:
                    continue  # Skip header row
            data.append(row)

    if not data:
        raise ValueError(f"CSV file is empty: {filepath}")

    arr: np.ndarray = np.array(data, dtype=np.float64)

    if arr.shape[1] == 1:
        # Single column — assume amplitude only at default sampling rate
        signal: np.ndarray = arr[:, 0]
        sr: int = DEFAULT_SAMPLING_RATE
        t: np.ndarray = np.arange(len(signal)) / sr
    elif arr.shape[1] >= 2:
        t = arr[:, 0]
        signal = arr[:, 1]
        sr = int(np.round(1.0 / np.median(np.diff(t))))
    else:
        raise ValueError(f"Unexpected CSV column count: {arr.shape[1]}")

    logger.info("Loaded CSV: %s (%d samples, %d Hz)", filepath, len(signal), sr)
    return t, signal, sr


def load_wfdb(
    dat_path: Path,
    hea_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a WFDB-compatible .dat file with its .hea header.

    This is a minimal parser for PhysioNet / MIT-BIH format.  It supports
    Format 212 (two leads, 12-bit) and Format 16 (single lead, 16-bit).

    Args:
        dat_path: Path to the .dat binary file.
        hea_path: Path to the .hea header file.  If None, the .hea is
                  assumed to share the same stem as dat_path.

    Returns:
        Tuple (t, signal, sampling_rate).  If two leads exist, only the
        first lead is returned.

    Raises:
        FileNotFoundError: If either file is missing.
        RuntimeError: If the header cannot be parsed.
    """
    if hea_path is None:
        hea_path = dat_path.with_suffix(".hea")

    if not dat_path.exists():
        raise FileNotFoundError(f".dat file not found: {dat_path}")
    if not hea_path.exists():
        raise FileNotFoundError(f".hea file not found: {hea_path}")

    # --- Parse .hea header ---
    with open(hea_path, "r") as fh:
        header_lines: list[str] = [line.strip() for line in fh if line.strip()]

    if not header_lines:
        raise RuntimeError(f"Empty header file: {hea_path}")

    # First line: record name, number of leads, ...
    first_parts: list[str] = header_lines[0].split()
    n_signals: int = int(first_parts[1])
    sampling_rate: int = int(first_parts[2])

    # Signal specification line(s)
    fmt: int | None = None
    n_samples: int = 0
    for line in header_lines[1:]:
        if line.startswith("#"):
            continue
        parts: list[str] = line.split()
        if len(parts) < 8:
            continue
        n_samples = int(parts[3])
        fmt = int(parts[1])
        break  # Read only first lead

    if fmt is None:
        raise RuntimeError("Could not determine signal format from header")

    # --- Read binary data ---
    raw: bytes = dat_path.read_bytes()

    if fmt == 16:
        # 16-bit, single lead
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    elif fmt == 212:
        # MIT-BIH Format 212: two 12-bit samples packed into 3 bytes
        # For simplicity, unpack using struct
        signal = _decode_format_212(raw)
    elif fmt == 8:
        signal = np.frombuffer(raw, dtype=np.int8).astype(np.float64)
    elif fmt == 61:
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    else:
        raise RuntimeError(f"Unsupported WFDB format: {fmt}")

    if n_samples and len(signal) > n_samples:
        signal = signal[:n_samples]

    t: np.ndarray = np.arange(len(signal)) / sampling_rate

    logger.info("Loaded WFDB: %s (%d samples, %d Hz)", dat_path, len(signal), sampling_rate)
    return t, signal, sampling_rate


def _decode_format_212(raw: bytes) -> np.ndarray:
    """Decode MIT-BIH Format 212 (two 12-bit values in 3 bytes).

    Bytes layout: [b1 b2 b3] → sample1 = (b1 << 4) | (b2 >> 4),
                             sample2 = ((b2 & 0x0F) << 8) | b3

    Returns only the first lead's samples (every other sample).
    """
    samples: list[int] = []
    for i in range(0, len(raw) - 2, 3):
        b1, b2, b3 = raw[i], raw[i + 1], raw[i + 2]
        s1: int = ((b1 & 0xFF) << 4) | ((b2 >> 4) & 0x0F)
        s2: int = ((b2 & 0x0F) << 8) | (b3 & 0xFF)
        # Sign-extend 12-bit
        if s1 >= 0x800:
            s1 -= 0x1000
        if s2 >= 0x800:
            s2 -= 0x1000
        samples.extend([s1, s2])

    return np.array(samples, dtype=np.float64)[::2]  # Return lead 1 only


def load_ecg(
    filepath: Optional[str] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Unified ECG loading interface.

    If ``filepath`` is provided the function tries to load the file in the
    appropriate format.  Otherwise a synthetic signal is generated with the
    keyword arguments passed to :func:`generate_ecg_signal`.

    Args:
        filepath: Path to a .csv or .dat file, or None for synthetic data.
        **kwargs: Passed through to :func:`generate_ecg_signal`.

    Returns:
        Tuple (t, signal, sampling_rate).
    """
    if filepath is None:
        logger.info("No file provided — generating synthetic ECG signal")
        t, clean, noisy, sr = generate_ecg_signal(**kwargs)
        return t, noisy, sr

    path: Path = Path(filepath)
    suffix: str = path.suffix.lower()

    if suffix in (".csv", ".txt"):
        return load_csv(path)
    elif suffix == ".dat":
        return load_wfdb(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: .csv, .dat (+ .hea)"
        )


def save_csv(
    filepath: Path,
    t: np.ndarray,
    signal: np.ndarray,
) -> None:
    """Save a time–amplitude ECG array as a CSV file.

    Args:
        filepath: Destination path.
        t: Time array (seconds).
        signal: ECG amplitude array (mV).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "amplitude_mv"])
        for ti, vi in zip(t, signal):
            writer.writerow([f"{ti:.6f}", f"{vi:.6f}"])
    logger.info("Saved CSV: %s (%d rows)", filepath, len(t))
