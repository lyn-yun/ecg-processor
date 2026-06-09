"""ECG Signal Processor — Streamlit web application.

Launch with::

    streamlit run app.py

Provides an interactive web UI for:
- Uploading a CSV ECG file or using synthetic data
- Adjusting filter parameters in the sidebar
- Viewing real-time before/after plots
- Downloading processed data
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data_loader import generate_ecg_signal, load_csv, save_csv
from src.preprocessing import PreprocessingConfig, preprocess
from src.visualization import (
    plot_comprehensive_overview,
    plot_filter_response,
    plot_processing_stages,
    plot_signal_comparison,
    plot_spectrum_comparison,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ECG Signal Processor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🫀 ECG Signal Processor")
st.markdown(
    "A digital signal processing toolkit for electrocardiogram (ECG) signals. "
    "Apply **FIR / IIR filters** to remove baseline wander, power-line interference, "
    "and high-frequency noise. All filtering uses **zero-phase (filtfilt)** processing "
    "to preserve waveform morphology."
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

# Data source
st.sidebar.subheader("📂 Data Source")
data_option = st.sidebar.radio(
    "Choose data source:",
    ["Synthetic (generated)", "Upload CSV file"],
)

uploaded_file = None
if data_option == "Upload CSV file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload ECG CSV file",
        type=["csv", "txt"],
        help="CSV with columns: time (s), amplitude (mV)",
    )

# Synthetic parameters (only when no upload)
if uploaded_file is None:
    with st.sidebar.expander("Synthetic Signal Parameters", expanded=True):
        duration = st.slider("Duration (s)", 5.0, 30.0, 10.0, 1.0)
        heart_rate = st.slider("Heart Rate (BPM)", 40, 120, 72, 1)
        sampling_rate = st.selectbox("Sampling Rate (Hz)", [250, 360, 500, 1000], index=1)
        noise_freq = st.selectbox("Power-line Frequency (Hz)", [50, 60], index=0)
        seed = st.number_input("Random Seed", 0, 9999, 42)

# Filter settings
st.sidebar.subheader("🔧 Filter Settings")
filter_type = st.sidebar.radio("Filter Type", ["iir", "fir"], index=0)

with st.sidebar.expander("Advanced Filter Options", expanded=False):
    if filter_type == "fir":
        fir_taps = st.slider("FIR Taps", 31, 301, 101, 10)
        iir_order = 4
    else:
        fir_taps = 101
        iir_order = st.slider("IIR Order", 2, 10, 4, 1)

lowcut = st.number_input("High-Pass Cut-off (Hz)", 0.1, 5.0, 0.5, 0.1,
                         help="Removes baseline wander below this frequency")
highcut = st.number_input("Low-Pass Cut-off (Hz)", 10.0, 100.0, 40.0, 1.0,
                          help="Removes high-frequency (EMG) noise above this frequency")

# Processing stages
st.sidebar.subheader("🧪 Processing Stages")
apply_baseline = st.sidebar.checkbox("Baseline Wander Removal (HP)", value=True)
apply_notch = st.sidebar.checkbox("Power-line Notch Filter", value=True)
apply_lowpass = st.sidebar.checkbox("Low-Pass Filter (EMG Removal)", value=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

# --- Load data ---
if uploaded_file is not None:
    try:
        content = uploaded_file.read()
        # Write to a temp path then use load_csv
        tmp_path = Path("data") / uploaded_file.name
        tmp_path.parent.mkdir(exist_ok=True)
        tmp_path.write_bytes(content)
        t, raw_signal, sr = load_csv(tmp_path)
        st.success(f"Loaded **{uploaded_file.name}**: {len(raw_signal)} samples @ {sr} Hz")
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
        st.stop()
else:
    with st.spinner("Generating synthetic ECG signal..."):
        t, clean_signal, raw_signal, sr = generate_ecg_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=heart_rate,
            add_noise=True,
            power_line_freq=noise_freq,
            seed=seed,
        )
    st.info(f"Synthetic signal: {len(raw_signal)} samples @ {sr} Hz, "
            f"HR={heart_rate} BPM, {duration:.0f} s")

# --- Process ---
enabled = []
if apply_baseline:
    enabled.append("baseline")
if apply_notch:
    enabled.append("notch")
if apply_lowpass:
    enabled.append("lowpass")

config = PreprocessingConfig(
    sampling_rate=sr,
    filter_type=filter_type,
    lowcut=lowcut,
    highcut=highcut,
    notch_freq=noise_freq if uploaded_file is None else 50.0,
    fir_taps=fir_taps,
    iir_order=iir_order,
    enabled_stages=enabled,
)

with st.spinner("Processing..."):
    results = preprocess(raw_signal, sr, config)
filtered = results["filtered"]

# --- Tabs ---
tabs = st.tabs([
    "📈 Signal Comparison",
    "🔄 Processing Stages",
    "📊 Frequency Response",
    "📡 Power Spectrum",
    "🧾 Comprehensive",
])

with tabs[0]:
    st.subheader("Raw vs. Filtered ECG Signal")
    time_zoom = st.slider("Time window (s)", 1.0, float(t[-1]), 3.0, 0.5,
                          key="zoom1")
    fig = plot_signal_comparison(
        t, raw_signal, filtered,
        title=f"ECG Signal: Raw vs. Filtered ({filter_type.upper()})",
        time_range=(0, time_zoom),
    )
    st.pyplot(fig)

with tabs[1]:
    st.subheader("Processing Stages")
    fig = plot_processing_stages(
        t, raw_signal,
        results["baseline_removed"],
        results["notch_filtered"],
        filtered,
        title=f"ECG Processing Stages ({filter_type.upper()})",
        time_range=(0, min(5.0, t[-1])),
    )
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Filter Frequency Responses")
    fig = plot_filter_response(sr, filter_type)
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Power Spectrum: Before vs. After")
    fig = plot_spectrum_comparison(
        raw_signal, filtered, sr,
        title=f"Power Spectrum ({filter_type.upper()})",
    )
    st.pyplot(fig)

with tabs[4]:
    st.subheader("Comprehensive Overview")
    fig = plot_comprehensive_overview(
        t, raw_signal, filtered,
        results["baseline_removed"],
        results["notch_filtered"],
        sampling_rate=sr,
        filter_type=filter_type,
        time_range=(0, min(2.5, t[-1])),
    )
    st.pyplot(fig)

# --- Download ---
st.sidebar.subheader("💾 Export")
st.sidebar.markdown("Download the filtered signal as a CSV file.")

buf = io.StringIO()
np.savetxt(buf, np.column_stack([t, filtered]),
           delimiter=",", header="time_s,amplitude_mv", comments="")
st.sidebar.download_button(
    label="📥 Download Processed CSV",
    data=buf.getvalue(),
    file_name="processed_ecg.csv",
    mime="text/csv",
)

# --- Metrics ---
st.sidebar.subheader("📊 Signal Metrics")
col_a, col_b, col_c = st.sidebar.columns(3)
col_a.metric("Raw STD", f"{np.std(raw_signal):.4f}")
col_b.metric("Filtered STD", f"{np.std(filtered):.4f}")
col_c.metric("SNR Gain", f"{np.std(raw_signal) / (np.std(filtered) + 1e-12):.1f}×")
