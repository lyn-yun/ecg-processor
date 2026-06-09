#!/usr/bin/env python3
"""ECG Signal Processor — Main entry point.

Command-line interface for the ECG signal processing pipeline::

    python main.py                        # run with synthetic data and defaults
    python main.py --file data/sample.csv # load from a CSV file
    python main.py --filter fir           # use FIR instead of IIR filters
    python main.py --noise-freq 60        # notch at 60 Hz (US mains)
    python main.py --heart-rate 80        # synthetic signal at 80 BPM
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import generate_ecg_signal, load_ecg, save_csv
from src.preprocessing import PreprocessingConfig, preprocess
from src.filters import compute_frequency_response, design_iir_highpass, design_iir_lowpass, design_iir_notch
from src.visualization import (
    plot_comprehensive_overview,
    plot_filter_response,
    plot_processing_stages,
    plot_signal_comparison,
    plot_spectrum_comparison,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with a consistent format.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level: int = logging.DEBUG if verbose else logging.INFO
    fmt: str = "%(asctime)s [%(levelname)-7s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="ECG Signal Processor — filter and visualise ECG signals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --filter fir --duration 15
  python main.py --file data/sample.csv --noise-freq 60
  python main.py --heart-rate 80 --output-dir my_results
        """,
    )

    # Input options
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to ECG data file (.csv or .dat). If omitted, synthetic data is generated.",
    )

    # Synthetic signal options
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Duration of synthetic signal in seconds (default: 10).",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=360,
        help="Sampling rate in Hz (default: 360).",
    )
    parser.add_argument(
        "--heart-rate", type=int, default=72,
        help="Heart rate in BPM for synthetic signal (default: 72).",
    )
    parser.add_argument(
        "--noise-freq", type=float, default=50.0,
        help="Power-line frequency in Hz: 50 (default) or 60.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible noise (default: 42).",
    )

    # Filter options
    parser.add_argument(
        "--filter", type=str, choices=["fir", "iir"], default="iir",
        help="Filter type: fir or iir (default: iir).",
    )
    parser.add_argument(
        "--fir-taps", type=int, default=101,
        help="Number of taps for FIR filters (default: 101).",
    )
    parser.add_argument(
        "--iir-order", type=int, default=4,
        help="Order for IIR filters (default: 4).",
    )

    # Preprocessing stages
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline wander removal.",
    )
    parser.add_argument(
        "--skip-notch", action="store_true",
        help="Skip power-line notch filter.",
    )
    parser.add_argument(
        "--skip-lowpass", action="store_true",
        help="Skip high-frequency noise removal.",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for output figures (default: results/).",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Save processed signal as CSV in the output directory.",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse args, run pipeline, generate figures."""
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ECG Signal Processor — starting pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load or generate data
    # ------------------------------------------------------------------
    try:
        t, raw_signal, sr = load_ecg(
            filepath=args.file,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            heart_rate=args.heart_rate,
            add_noise=True,
            power_line_freq=args.noise_freq,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Failed to load data: %s", exc)
        sys.exit(1)

    logger.info("Signal: %d samples, %d Hz, %.1f s", len(raw_signal), sr, t[-1])

    # ------------------------------------------------------------------
    # 2. Build preprocessing config
    # ------------------------------------------------------------------
    enabled_stages: list[str] = []
    if not args.skip_baseline:
        enabled_stages.append("baseline")
    if not args.skip_notch:
        enabled_stages.append("notch")
    if not args.skip_lowpass:
        enabled_stages.append("lowpass")

    config = PreprocessingConfig(
        sampling_rate=sr,
        filter_type=args.filter,
        lowcut=0.5,
        highcut=40.0,
        notch_freq=args.noise_freq,
        fir_taps=args.fir_taps,
        iir_order=args.iir_order,
        enabled_stages=enabled_stages,
    )

    # ------------------------------------------------------------------
    # 3. Run preprocessing pipeline
    # ------------------------------------------------------------------
    logger.info("Running preprocessing pipeline (%s)...", args.filter.upper())
    try:
        results = preprocess(raw_signal, sr, config)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)

    filtered_signal = results["filtered"]
    logger.info("Pipeline complete. Output signal length: %d", len(filtered_signal))

    # ------------------------------------------------------------------
    # 4. Generate visualisations
    # ------------------------------------------------------------------
    logger.info("Generating figures...")

    try:
        plot_signal_comparison(
            t, raw_signal, filtered_signal,
            title=f"ECG Signal: Raw vs. Filtered ({args.filter.upper()})",
            save_path=output_dir / "01_raw_vs_filtered.png",
            time_range=(0, 3.0),
        )

        plot_processing_stages(
            t, raw_signal,
            results["baseline_removed"],
            results["notch_filtered"],
            filtered_signal,
            title=f"ECG Processing Stages ({args.filter.upper()})",
            save_path=output_dir / "02_processing_stages.png",
            time_range=(0, 3.0),
        )

        plot_filter_response(
            sampling_rate=sr,
            filter_type=args.filter,
            save_path=output_dir / "03_filter_responses.png",
        )

        plot_spectrum_comparison(
            raw_signal, filtered_signal,
            sampling_rate=sr,
            title=f"ECG Power Spectrum: Before vs. After ({args.filter.upper()})",
            save_path=output_dir / "04_spectrum_comparison.png",
        )

        plot_comprehensive_overview(
            t, raw_signal, filtered_signal,
            results["baseline_removed"],
            results["notch_filtered"],
            sampling_rate=sr,
            filter_type=args.filter,
            save_path=output_dir / "05_comprehensive_overview.png",
            time_range=(0, 2.0),
        )
    except Exception as exc:
        logger.error("Visualisation failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Optionally save CSV
    # ------------------------------------------------------------------
    if args.save_csv:
        csv_path = output_dir / "processed_ecg.csv"
        try:
            save_csv(csv_path, t, filtered_signal)
        except Exception as exc:
            logger.error("CSV export failed: %s", exc)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("All done! Figures saved to: %s", output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
