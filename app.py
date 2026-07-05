#!/usr/bin/env python3
"""Streamlit entrypoint for the PA Spectral Analysis Workbench."""

from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fft_core import (
    baseline_correct,
    check_time_uniformity,
    compute_fft,
    create_fft_spectrum_figure,
    create_time_domain_figure,
    find_bw50,
    find_dominant_frequency,
    normalize_spectrum,
    read_comsol_csv,
    split_by_parameter,
)
from workbench_ui import render_workbench


st.set_page_config(
    page_title="PA Spectral Analysis Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main-title {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 55%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: .15rem; letter-spacing: -.5px;
      }
      .sub-title { color: #8888aa; font-size: 1.02rem; margin-bottom: 1.2rem; }
      div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102,126,234,.3); border-radius: 12px;
        padding: 14px 18px;
      }
      .stDownloadButton > button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">📊 PA Spectral Analysis Workbench</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">COMSOL waveform collections, signal roles, derived '
    'differences, FFT spectra, detector filtering, overlay plots, and '
    'reproducible export.</p>',
    unsafe_allow_html=True,
)


def _fig_to_png(fig, dpi=300):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    return buffer.getvalue()


def _read_uploaded_csv(upload):
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as handle:
            handle.write(upload.getvalue())
            path = handle.name
        return read_comsol_csv(path)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def _single_report(metrics, settings):
    lines = [
        "# Single Signal FFT Report",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Settings",
    ]
    lines.extend(f"- {key}: {value}" for key, value in settings.items())
    lines.extend(["", "## Results", ""])
    for metric in metrics:
        lines.append(
            f"- {metric['point']} ({metric.get('param_value', '')}): "
            f"dominant={metric['dominant_freq']:.6g} MHz, "
            f"BW50={metric['bandwidth']:.6g} MHz"
        )
    return "\n".join(lines)


def render_single_signal():
    st.markdown("### Single Signal FFT")
    upload = st.file_uploader(
        "Upload a COMSOL CSV file",
        type=["csv"],
        help="Metadata rows beginning with % are supported.",
        key="single_upload",
    )
    if upload is None:
        st.info("Upload a CSV file to begin.")
        return
    try:
        frame = _read_uploaded_csv(upload)
    except Exception as exc:
        st.error(f"CSV could not be parsed: {exc}")
        return

    st.success(f"Parsed {len(frame)} rows and {len(frame.columns)} columns.")
    with st.expander("Preview"):
        st.dataframe(frame.head(20), use_container_width=True)

    columns = list(frame.columns)
    left, right = st.columns(2)
    with left:
        point_name = st.text_input("Point name", "Detector P3")
        parameter_options = ["None (single signal)"] + columns
        parameter_selection = st.selectbox("Parameter column", parameter_options)
        parameter_column = (
            None
            if parameter_selection == "None (single signal)"
            else parameter_selection
        )
        time_default = next(
            (index for index, name in enumerate(columns) if "time" in str(name).lower()),
            0,
        )
        time_column = st.selectbox("Time column", columns, index=time_default)
        signal_default = next(
            (
                index for index, name in enumerate(columns)
                if "pressure" in str(name).lower()
                or "photoacoustic" in str(name).lower()
            ),
            min(len(columns) - 1, 1),
        )
        signal_column = st.selectbox(
            "Signal / pressure column", columns, index=signal_default
        )
    with right:
        baseline_end = st.number_input(
            "Baseline end (ns)", value=10.0, min_value=0.0
        )
        frequency_max = st.number_input(
            "Plot frequency max (MHz)", value=150.0, min_value=1.0
        )
        dpi = int(st.number_input(
            "Figure DPI", value=300, min_value=72, max_value=600, step=25
        ))

    run_key = (
        upload.name,
        upload.size,
        point_name,
        parameter_column,
        time_column,
        signal_column,
        baseline_end,
        frequency_max,
        dpi,
    )
    if st.button("Run FFT analysis", type="primary", use_container_width=True):
        data = split_by_parameter(
            frame,
            {
                "time": time_column,
                "signal": signal_column,
                "param": parameter_column,
            },
        )
        fft_results, metrics, spectra_rows = {}, [], []
        for parameter_value, (time_ns, signal) in data.items():
            check_time_uniformity(time_ns)
            corrected, _ = baseline_correct(time_ns, signal, baseline_end)
            frequency, amplitude = compute_fft(time_ns, corrected)
            normalized, non_dc_max = normalize_spectrum(amplitude)
            dominant, peak_index = find_dominant_frequency(frequency, amplitude)
            bandwidth, low, high = find_bw50(
                frequency, normalized, peak_index
            )
            fft_results[parameter_value] = {
                "freq_mhz": frequency,
                "amplitude": amplitude,
                "normalized": normalized,
                "non_dc_max": non_dc_max,
                "dominant_freq": dominant,
                "peak_idx": peak_index,
                "bandwidth": bandwidth,
                "bw_f_low": low,
                "bw_f_high": high,
            }
            metrics.append({
                "point": point_name,
                "param_name": parameter_column or "parameter",
                "param_value": parameter_value,
                "dominant_freq": dominant,
                "bandwidth": bandwidth,
                "bw_f_low": low,
                "bw_f_high": high,
                "non_dc_max": non_dc_max,
                "freq_resolution": (
                    frequency[1] - frequency[0] if len(frequency) > 1 else float("nan")
                ),
            })
            spectra_rows.extend({
                "parameter": parameter_value,
                "frequency_mhz": frequency[index],
                "normalized_amplitude": normalized[index],
            } for index in range(len(frequency)))

        time_figure = create_time_domain_figure(
            data, point_name, parameter_column or "parameter"
        )
        spectrum_figure = create_fft_spectrum_figure(
            fft_results, point_name, frequency_max, parameter_column or "parameter"
        )
        settings = {
            "point_name": point_name,
            "parameter_column": parameter_column,
            "time_column": time_column,
            "signal_column": signal_column,
            "baseline_end_ns": baseline_end,
            "plot_frequency_max_mhz": frequency_max,
            "figure_dpi": dpi,
        }
        st.session_state.single_result = {
            "run_key": run_key,
            "metrics": metrics,
            "spectra": spectra_rows,
            "time_png": _fig_to_png(time_figure, dpi),
            "spectrum_png": _fig_to_png(spectrum_figure, dpi),
            "report": _single_report(metrics, settings),
        }
        plt.close(time_figure)
        plt.close(spectrum_figure)

    result = st.session_state.get("single_result")
    if not result:
        return
    if result["run_key"] != run_key:
        st.warning("Settings changed after the last run. Run the analysis again.")

    st.markdown("## Results")
    metric_columns = st.columns(min(len(result["metrics"]), 6))
    for column, metric in zip(metric_columns, result["metrics"][:6]):
        column.metric(
            f"{metric['point']} ({metric.get('param_value', '')})",
            f"{metric['dominant_freq']:.3f} MHz",
            f"BW50 {metric['bandwidth']:.3f} MHz",
        )
    time_tab, spectrum_tab, table_tab, report_tab = st.tabs(
        ["Time domain", "FFT spectrum", "Metrics", "Report"]
    )
    with time_tab:
        st.image(result["time_png"], use_container_width=True)
        st.download_button(
            "Download time-domain PNG", result["time_png"],
            "single_time_domain.png", "image/png"
        )
    with spectrum_tab:
        st.image(result["spectrum_png"], use_container_width=True)
        st.download_button(
            "Download FFT spectrum PNG", result["spectrum_png"],
            "single_fft_spectrum.png", "image/png"
        )
    with table_tab:
        metrics_frame = pd.DataFrame(result["metrics"])
        st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download metrics CSV",
            metrics_frame.to_csv(index=False).encode("utf-8"),
            "single_fft_metrics.csv",
            "text/csv",
        )
        st.download_button(
            "Download spectra CSV",
            pd.DataFrame(result["spectra"]).to_csv(index=False).encode("utf-8"),
            "single_fft_spectra.csv",
            "text/csv",
        )
    with report_tab:
        st.code(result["report"], language=None)
        st.download_button(
            "Download report", result["report"],
            "single_fft_report.md", "text/markdown"
        )


with st.sidebar:
    st.markdown("## Analysis mode")
    mode = st.radio(
        "Select mode",
        ["Mode 1 — Single Signal FFT", "Mode 2 — General PA Workbench"],
        label_visibility="collapsed",
    )
    st.divider()

if mode.startswith("Mode 1"):
    render_single_signal()
else:
    render_workbench(st, _fig_to_png)
