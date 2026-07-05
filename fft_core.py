#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMSOL FFT Analysis Tool — Core Library
=========================================

Reusable functions for parsing COMSOL time-domain pressure CSV exports,
performing FFT-based frequency-domain analysis, and generating plots
and reports.

Designed to support both single-run and parametric-sweep exports
(Stage 2 and Stage 3).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
#  1. CSV Parsing
# ═══════════════════════════════════════════════════════════════════════

def read_comsol_csv(filepath, time_column=None, signal_column=None,
                    param_column=None):
    """Read a COMSOL-exported CSV file, skipping metadata lines
    starting with '%'.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    time_column : str or int, optional
        Column name or 0-based index for time data.
    signal_column : str or int, optional
        Column name or 0-based index for the signal (pressure) data.
    param_column : str or int, optional
        Column name or 0-based index for the parametric sweep column.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with detected or user-specified columns.
    """
    # Read lines, skip COMSOL metadata (lines starting with %)
    data_lines = []
    header_line = None
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "":
                continue
            if stripped.startswith("%"):
                # Last metadata line is often the header description
                header_line = stripped
                continue
            data_lines.append(stripped)

    if not data_lines:
        raise ValueError(f"No data found in {filepath}")

    # Check if header_line contains column names
    col_names = None
    if header_line:
        # Remove leading '% ' and split
        hdr = header_line.lstrip("% ").strip()
        parts = [p.strip().strip('"') for p in hdr.split(",")]
        if len(parts) >= 2:
            col_names = parts

    # Parse data into DataFrame
    from io import StringIO
    raw_text = "\n".join(data_lines)
    if col_names:
        df = pd.read_csv(StringIO(raw_text), header=None,
                         names=col_names[:len(data_lines[0].split(","))])
    else:
        df = pd.read_csv(StringIO(raw_text), header=None)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    return df


def detect_columns(df, time_column=None, signal_column=None,
                   param_column=None):
    """Auto-detect time, signal, and parameter columns.

    Parameters
    ----------
    df : pandas.DataFrame
    time_column, signal_column, param_column : str or int, optional
        User overrides.

    Returns
    -------
    dict
        Keys: 'time', 'signal', 'param' (param may be None).
    """
    cols = list(df.columns)

    def _resolve(spec, keywords, label):
        if spec is not None:
            if isinstance(spec, int):
                if spec < len(cols):
                    return cols[spec]
                raise ValueError(f"{label} index {spec} out of range. "
                                 f"Available columns: {cols}")
            if spec in cols:
                return spec
            raise ValueError(f"{label} '{spec}' not found. "
                             f"Available columns: {cols}")
        # Auto-detect by keyword
        for kw in keywords:
            for c in cols:
                if kw.lower() in c.lower():
                    return c
        return None

    time_kw = ["Time (ns)", "time", "t "]
    signal_kw = ["pressure", "Photoacoustic", "p "]
    param_kw = ["tau_p", "S3scale", "d_M"]

    t_col = _resolve(time_column, time_kw, "Time column")
    s_col = _resolve(signal_column, signal_kw, "Signal column")
    p_col = _resolve(param_column, param_kw, "Parameter column")

    if t_col is None:
        raise ValueError(f"Cannot auto-detect time column. "
                         f"Available columns: {cols}")
    if s_col is None:
        raise ValueError(f"Cannot auto-detect signal column. "
                         f"Available columns: {cols}")

    return {"time": t_col, "signal": s_col, "param": p_col}


# ═══════════════════════════════════════════════════════════════════════
#  2. Parametric Sweep Handling
# ═══════════════════════════════════════════════════════════════════════

def split_by_parameter(df, col_map):
    """Group DataFrame by the parameter column.

    Returns
    -------
    dict
        {param_value: (time_array, signal_array)}
        If no parameter column, returns {None: (time, signal)}.
    """
    t_col = col_map["time"]
    s_col = col_map["signal"]
    p_col = col_map["param"]

    if p_col is None:
        return {None: (df[t_col].values, df[s_col].values)}

    result = {}
    unique_vals = np.unique(np.round(df[p_col].values, 1))
    for val in unique_vals:
        mask = np.abs(df[p_col].values - val) < 0.5
        key = int(round(val)) if float(val) == round(val) else round(val, 2)
        result[key] = (df[t_col].values[mask], df[s_col].values[mask])

    return result


# ═══════════════════════════════════════════════════════════════════════
#  3. Signal Processing
# ═══════════════════════════════════════════════════════════════════════

def check_time_uniformity(time_ns):
    """Warn if time step variation exceeds 1%."""
    dt = np.diff(time_ns)
    dt_mean = np.mean(dt)
    dt_var = np.max(np.abs(dt - dt_mean)) / dt_mean * 100
    if dt_var > 1.0:
        warnings.warn(f"Non-uniform time step detected: "
                      f"max variation = {dt_var:.2f}%")
    return dt_mean


def baseline_correct(time_ns, signal, baseline_end_ns=10.0):
    """Subtract the mean signal in [0, baseline_end_ns].

    Returns
    -------
    corrected : ndarray
    baseline_value : float
    """
    mask = time_ns <= baseline_end_ns
    baseline = np.mean(signal[mask])
    return signal - baseline, baseline


def compute_fft(time_ns, signal):
    """Apply Hann window, compute real FFT.

    Returns
    -------
    freq_mhz : ndarray
    amplitude : ndarray
    """
    N = len(signal)
    dt = (time_ns[-1] - time_ns[0]) / (N - 1)  # ns
    dt_s = dt * 1e-9  # seconds

    window = np.hanning(N)
    fft_vals = np.fft.rfft(signal * window)
    amplitude = np.abs(fft_vals)
    freq_mhz = np.fft.rfftfreq(N, d=dt_s) / 1e6

    return freq_mhz, amplitude


def normalize_spectrum(amplitude):
    """Normalize by non-DC maximum amplitude.

    Returns
    -------
    normalized : ndarray
    non_dc_max : float
    """
    non_dc_max = np.max(amplitude[1:])
    return amplitude / non_dc_max, non_dc_max


def find_dominant_frequency(freq_mhz, amplitude):
    """Return (dominant_freq_MHz, bin_index), excluding DC."""
    idx = np.argmax(amplitude[1:]) + 1
    return freq_mhz[idx], idx


def find_bw50(freq_mhz, norm_amp, peak_idx):
    """Half-maximum bandwidth via linear interpolation of 0.5 crossings.

    Returns
    -------
    bandwidth : float
    f_low : float
    f_high : float
    """
    threshold = 0.5

    # Left crossing
    f_low = freq_mhz[1]
    for i in range(peak_idx, 1, -1):
        if norm_amp[i - 1] < threshold <= norm_amp[i]:
            x0, x1 = freq_mhz[i - 1], freq_mhz[i]
            y0, y1 = norm_amp[i - 1], norm_amp[i]
            f_low = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            break

    # Right crossing
    f_high = freq_mhz[-1]
    for i in range(peak_idx, len(norm_amp) - 1):
        if norm_amp[i] >= threshold > norm_amp[i + 1]:
            x0, x1 = freq_mhz[i], freq_mhz[i + 1]
            y0, y1 = norm_amp[i], norm_amp[i + 1]
            f_high = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
            break

    return f_high - f_low, f_low, f_high


# ═══════════════════════════════════════════════════════════════════════
#  4. Plotting
# ═══════════════════════════════════════════════════════════════════════

# Default colour palette for parametric sweeps
_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_time_domain(data_dict, point_name, output_path,
                     param_name="param", dpi=300):
    """Plot raw time-domain pressure for all parameter values.

    Parameters
    ----------
    data_dict : dict
        {param_value: (time_ns, signal)} or {None: (time, signal)}.
    point_name : str
        Display name, e.g. "Detector P3".
    output_path : str
    param_name : str
        Name of the parameter for legend labels.
    dpi : int
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (pval, (t, s)) in enumerate(sorted(data_dict.items(),
                                               key=lambda x: (x[0] is None,
                                                               x[0]))):
        color = _COLORS[i % len(_COLORS)]
        if pval is not None:
            label = f"{param_name} = {pval}"
        else:
            label = point_name
        ax.plot(t, s, label=label, color=color, linewidth=1.2)

    ax.set_xlabel("Time (ns)", fontsize=13)
    ax.set_ylabel("Photoacoustic pressure (Pa)", fontsize=13)
    ax.set_title(f"Raw Time-Domain Pressure at {point_name}", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


def plot_fft_spectrum(results_dict, point_name, output_path,
                      freq_max=150.0, param_name="param", dpi=300):
    """Plot normalized FFT spectrum with f_dom and BW50 in legend.

    Parameters
    ----------
    results_dict : dict
        {param_value: dict with 'freq_mhz', 'normalized',
         'dominant_freq', 'bandwidth'}
    point_name : str
    output_path : str
    freq_max : float
    param_name : str
    dpi : int
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (pval, res) in enumerate(sorted(results_dict.items(),
                                           key=lambda x: (x[0] is None,
                                                          x[0]))):
        color = _COLORS[i % len(_COLORS)]
        mask = res["freq_mhz"] <= freq_max
        if pval is not None:
            lbl = (f"{param_name} = {pval}  |  "
                   f"f_dom = {res['dominant_freq']:.3f} MHz  |  "
                   f"BW50 = {res['bandwidth']:.3f} MHz")
        else:
            lbl = (f"f_dom = {res['dominant_freq']:.3f} MHz  |  "
                   f"BW50 = {res['bandwidth']:.3f} MHz")
        ax.plot(res["freq_mhz"][mask], res["normalized"][mask],
                label=lbl, color=color, linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
               label="Half-maximum (0.5)")
    ax.set_xlabel("Frequency (MHz)", fontsize=13)
    ax.set_ylabel("Normalized Amplitude", fontsize=13)
    ax.set_title(f"Normalized FFT Spectrum at {point_name} "
                 f"(0\u2013{int(freq_max)} MHz)", fontsize=15)
    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


def create_time_domain_figure(data_dict, point_name, param_name="param"):
    """Create and return a raw time-domain figure (without saving).

    Same logic as plot_time_domain but returns the figure for
    embedding in Streamlit or other UIs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (pval, (t, s)) in enumerate(sorted(
            data_dict.items(), key=lambda x: (x[0] is None, x[0]))):
        color = _COLORS[i % len(_COLORS)]
        label = f"{param_name} = {pval}" if pval is not None else point_name
        ax.plot(t, s, label=label, color=color, linewidth=1.2)
    ax.set_xlabel("Time (ns)", fontsize=13)
    ax.set_ylabel("Photoacoustic pressure (Pa)", fontsize=13)
    ax.set_title(f"Raw Time-Domain Pressure at {point_name}", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_fft_spectrum_figure(results_dict, point_name,
                               freq_max=150.0, param_name="param"):
    """Create and return an FFT spectrum figure (without saving).

    Same logic as plot_fft_spectrum but returns the figure for
    embedding in Streamlit or other UIs.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (pval, res) in enumerate(sorted(
            results_dict.items(), key=lambda x: (x[0] is None, x[0]))):
        color = _COLORS[i % len(_COLORS)]
        mask = res["freq_mhz"] <= freq_max
        if pval is not None:
            lbl = (f"{param_name} = {pval}  |  "
                   f"f_dom = {res['dominant_freq']:.3f} MHz  |  "
                   f"BW50 = {res['bandwidth']:.3f} MHz")
        else:
            lbl = (f"f_dom = {res['dominant_freq']:.3f} MHz  |  "
                   f"BW50 = {res['bandwidth']:.3f} MHz")
        ax.plot(res["freq_mhz"][mask], res["normalized"][mask],
                label=lbl, color=color, linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
               label="Half-maximum (0.5)")
    ax.set_xlabel("Frequency (MHz)", fontsize=13)
    ax.set_ylabel("Normalized Amplitude", fontsize=13)
    ax.set_title(f"Normalized FFT Spectrum at {point_name} "
                 f"(0\u2013{int(freq_max)} MHz)", fontsize=15)
    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  5. Export Functions
# ═══════════════════════════════════════════════════════════════════════

def export_metrics_csv(results, path):
    """Write summary metrics CSV.

    Parameters
    ----------
    results : list of dict
        Each dict has keys: point, param_name, param_value,
        dominant_freq, bandwidth, bw_f_low, bw_f_high,
        non_dc_max, freq_resolution.
    path : str
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Point,Param_Name,Param_Value,"
                 "Dominant_Frequency_MHz,Half_Max_Bandwidth_MHz,"
                 "BW_F_Low_MHz,BW_F_High_MHz,"
                 "Non_DC_Max_Amplitude,Freq_Resolution_MHz\n")
        for r in results:
            pv = r.get("param_value", "")
            pn = r.get("param_name", "")
            fh.write(f"{r['point']},{pn},{pv},"
                     f"{r['dominant_freq']:.6f},"
                     f"{r['bandwidth']:.6f},"
                     f"{r['bw_f_low']:.6f},"
                     f"{r['bw_f_high']:.6f},"
                     f"{r['non_dc_max']:.10e},"
                     f"{r['freq_resolution']:.6f}\n")
    print(f"  [SAVED] {path}")


def export_spectra_csv(spectra_data, path, freq_max=500.0):
    """Write normalized spectra (0 to freq_max MHz) to CSV.

    Parameters
    ----------
    spectra_data : list of dict
        Each dict has: label, freq_mhz, normalized.
    path : str
    freq_max : float
    """
    header = ["Frequency_MHz"]
    cols = {}
    freq_ref = None

    for sd in spectra_data:
        name = sd["label"]
        header.append(name)
        mask = sd["freq_mhz"] <= freq_max
        cols[name] = sd["normalized"][mask]
        if freq_ref is None:
            freq_ref = sd["freq_mhz"][mask]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(len(freq_ref)):
            row = [f"{freq_ref[i]:.6f}"]
            for name in header[1:]:
                if i < len(cols[name]):
                    row.append(f"{cols[name][i]:.10e}")
                else:
                    row.append("")
            fh.write(",".join(row) + "\n")
    print(f"  [SAVED] {path}")


def write_analysis_report(results, config, path):
    """Write a text analysis report.

    Parameters
    ----------
    results : list of dict
        Metric results (same format as export_metrics_csv input).
    config : dict
        Analysis configuration settings.
    path : str
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("COMSOL FFT Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Tool: comsol-photoacoustic-fft-tool\n\n")

        f.write("ANALYSIS SETTINGS\n")
        f.write("-" * 40 + "\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("FFT METHOD\n")
        f.write("-" * 40 + "\n")
        f.write("  Baseline correction: mean subtraction for t <= "
                f"{config.get('baseline_end_ns', 10.0)} ns\n")
        f.write("  Window: Hann (numpy.hanning)\n")
        f.write("  FFT: numpy.fft.rfft (real FFT)\n")
        f.write("  Normalization: non-DC maximum amplitude\n\n")

        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        hdr = f"{'Point':<20s} {'Param':<12s} {'f_dom (MHz)':>12s} "
        hdr += f"{'BW50 (MHz)':>12s} {'BW range (MHz)':>24s}\n"
        f.write(hdr)
        for r in results:
            pv = r.get("param_value", "-")
            f.write(f"{r['point']:<20s} {str(pv):<12s} "
                    f"{r['dominant_freq']:>12.3f} "
                    f"{r['bandwidth']:>12.3f} "
                    f"[{r['bw_f_low']:>8.3f} -- {r['bw_f_high']:>8.3f}]\n")
        f.write("\n")

    print(f"  [SAVED] {path}")


# ═══════════════════════════════════════════════════════════════════════
#  6. High-Level Pipeline
# ═══════════════════════════════════════════════════════════════════════

def analyze_single(csv_path, point_name, output_dir,
                   param_column=None, time_column=None,
                   signal_column=None, param_name=None,
                   baseline_end_ns=10.0, freq_max=150.0,
                   export_freq_max=500.0, dpi=300):
    """Full analysis pipeline for one CSV file.

    Returns
    -------
    metrics : list of dict
    spectra : list of dict
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = point_name.replace(" ", "_")

    print(f"\n{'=' * 70}")
    print(f"  Analyzing: {point_name}")
    print(f"  Input:     {os.path.basename(csv_path)}")
    print(f"{'=' * 70}")

    # Parse CSV
    df = read_comsol_csv(csv_path, time_column=time_column,
                         signal_column=signal_column,
                         param_column=param_column)
    col_map = detect_columns(df, time_column=time_column,
                             signal_column=signal_column,
                             param_column=param_column)

    print(f"  Detected columns: time='{col_map['time']}', "
          f"signal='{col_map['signal']}', param='{col_map['param']}'")
    print(f"  Total rows: {len(df)}")

    # Split by parameter
    data_dict = split_by_parameter(df, col_map)
    detected_param_name = param_name or col_map["param"] or "param"
    print(f"  Parameter values: {sorted(k for k in data_dict if k is not None)}")

    # Validate time steps
    for pval, (t, s) in data_dict.items():
        dt_mean = check_time_uniformity(t)
        print(f"  {detected_param_name}={pval}: N={len(t)}, "
              f"t=[{t[0]:.1f}, {t[-1]:.1f}] ns, dt={dt_mean:.4f} ns")

    # Time-domain plot
    plot_time_domain(data_dict, point_name,
                     os.path.join(output_dir,
                                  f"{safe_name}_raw_time_domain_pressure.png"),
                     param_name=detected_param_name, dpi=dpi)

    # FFT per parameter value
    fft_results = {}
    metrics = []
    spectra = []

    for pval, (time_ns, signal) in sorted(data_dict.items(),
                                          key=lambda x: (x[0] is None,
                                                         x[0])):
        corrected, bl = baseline_correct(time_ns, signal, baseline_end_ns)
        freq, amp = compute_fft(time_ns, corrected)
        norm, ndc = normalize_spectrum(amp)
        dom_f, pk = find_dominant_frequency(freq, amp)
        bw, fl, fh = find_bw50(freq, norm, pk)
        df_res = freq[1] - freq[0]

        print(f"\n  [{detected_param_name}={pval}] baseline={bl:.6e}, "
              f"f_dom={dom_f:.3f} MHz, BW50={bw:.3f} MHz "
              f"[{fl:.3f}--{fh:.3f}]")

        fft_results[pval] = dict(
            freq_mhz=freq, amplitude=amp, normalized=norm,
            non_dc_max=ndc, dominant_freq=dom_f, peak_idx=pk,
            bandwidth=bw, bw_f_low=fl, bw_f_high=fh)

        metrics.append(dict(
            point=point_name, param_name=detected_param_name,
            param_value=pval, dominant_freq=dom_f, bandwidth=bw,
            bw_f_low=fl, bw_f_high=fh, non_dc_max=ndc,
            freq_resolution=df_res))

        label = (f"{safe_name}_{detected_param_name}{pval}_normalized"
                 if pval is not None else f"{safe_name}_normalized")
        spectra.append(dict(label=label, freq_mhz=freq, normalized=norm))

    # FFT spectrum plot
    plot_fft_spectrum(fft_results, point_name,
                      os.path.join(output_dir,
                                   f"{safe_name}_normalized_fft_spectrum"
                                   f"_0_{int(freq_max)}MHz.png"),
                      freq_max=freq_max,
                      param_name=detected_param_name, dpi=dpi)

    return metrics, spectra


def analyze_batch(config_path):
    """Run analysis from a JSON batch configuration file.

    Parameters
    ----------
    config_path : str
        Path to JSON config file.

    Returns
    -------
    all_metrics : list of dict
    all_spectra : list of dict
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    defaults = config.get("defaults", {})
    analyses = config.get("analyses", [])

    if not analyses:
        raise ValueError("No 'analyses' entries found in config file.")

    all_metrics = []
    all_spectra = []

    for entry in analyses:
        csv_path = entry["csv_path"]
        # Resolve relative paths against config file directory
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(config_path), csv_path)

        point_name = entry.get("point_name", "Unknown")
        output_dir = entry.get("output_dir",
                               defaults.get("output_dir", "./outputs"))
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(config_path),
                                      output_dir)

        m, s = analyze_single(
            csv_path=csv_path,
            point_name=point_name,
            output_dir=output_dir,
            param_column=entry.get("param_column"),
            time_column=entry.get("time_column"),
            signal_column=entry.get("signal_column"),
            param_name=entry.get("param_name"),
            baseline_end_ns=entry.get("baseline_end_ns",
                                      defaults.get("baseline_end_ns", 10.0)),
            freq_max=entry.get("freq_max_mhz",
                               defaults.get("freq_max_mhz", 150.0)),
            export_freq_max=entry.get("export_freq_max_mhz",
                                      defaults.get("export_freq_max_mhz",
                                                    500.0)),
            dpi=entry.get("figure_dpi",
                          defaults.get("figure_dpi", 300)),
        )
        all_metrics.extend(m)
        all_spectra.extend(s)

    # Export combined outputs
    combined_dir = defaults.get("output_dir", "./outputs")
    if not os.path.isabs(combined_dir):
        combined_dir = os.path.join(os.path.dirname(config_path),
                                    combined_dir)
    os.makedirs(combined_dir, exist_ok=True)

    export_metrics_csv(all_metrics,
                       os.path.join(combined_dir, "fft_metrics.csv"))
    export_spectra_csv(all_spectra,
                       os.path.join(combined_dir, "fft_spectra.csv"),
                       freq_max=defaults.get("export_freq_max_mhz", 500.0))

    report_config = {
        "baseline_end_ns": defaults.get("baseline_end_ns", 10.0),
        "freq_max_mhz": defaults.get("freq_max_mhz", 150.0),
        "export_freq_max_mhz": defaults.get("export_freq_max_mhz", 500.0),
        "figure_dpi": defaults.get("figure_dpi", 300),
        "config_file": config_path,
        "num_analyses": len(analyses),
    }
    write_analysis_report(all_metrics, report_config,
                          os.path.join(combined_dir, "analysis_report.txt"))

    return all_metrics, all_spectra
