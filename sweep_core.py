#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_core.py — COMSOL Wide-Format Sweep Spectral Analysis Engine
==================================================================

Parses COMSOL "all-cases" wide CSV exports (parametric sweep output where
each column represents one time-point / parameter combination) and performs
Hysi-style photoacoustic spectral analysis:

  1. Parse wide CSV header  →  {(pair_case, Phi_local): (t_ns, p_array)}
  2. Build OFF / ON / FORCE / DIFF signal sets
  3. Apply time gate + Hamming window
  4. Compute power spectrum (dB)
  5. Apply Gaussian detector response
  6. Extract spectral metrics

Column header format expected from COMSOL:
    p @ t=0; pair_case=3; Phi_local=20
    p @ t=0.5; pair_case=3; Phi_local=20
    ...

Usage
-----
    from sweep_core import parse_wide_comsol_csv, build_signal_set, ...
"""

import re
import warnings
import numpy as np
import pandas as pd
from io import StringIO


# ═══════════════════════════════════════════════════════════════════════
#  1. Wide CSV Parser
# ═══════════════════════════════════════════════════════════════════════

def parse_wide_comsol_csv(filepath):
    """Parse a COMSOL wide-format (all-cases) parametric sweep CSV.

    Expected format
    ---------------
    Row 0  : spatial coordinates header  (X, Y, Z, ...)
    Row 1  : column descriptions like    p @ t=0; pair_case=3; Phi_local=20
    Rows 2+: numeric data (one row per spatial point; we assume a single
             detector point, so only the first data row is used)

    Returns
    -------
    data : dict
        { (pair_case, Phi_local) : {"t_ns": np.ndarray, "p": np.ndarray} }
    meta : dict
        Parsed metadata: unique pair_cases, unique phi_locals, n_columns
    raw_df : pd.DataFrame
        Raw parsed DataFrame for preview.
    """
    raw_lines = []
    with open(filepath, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            raw_lines.append(stripped)

    if len(raw_lines) < 2:
        raise ValueError("CSV has fewer than 2 rows — cannot parse.")

    # ── Detect header rows ──────────────────────────────────────────
    # Find the row that contains 't=' pattern (column description row)
    desc_row_idx = None
    for i, ln in enumerate(raw_lines):
        if re.search(r"t\s*=\s*[\d.eE+\-]+", ln):
            desc_row_idx = i
            break

    if desc_row_idx is None:
        raise ValueError(
            "Cannot find column description row with 't=...' pattern. "
            "Expected COMSOL wide-format with headers like: "
            "'p @ t=0; pair_case=3; Phi_local=20'"
        )

    # Parse column descriptions
    desc_line = raw_lines[desc_row_idx]
    col_descs = [c.strip().strip('"') for c in desc_line.split(",")]

    # Data rows (everything after desc row, skip any remaining header rows)
    data_rows = []
    for ln in raw_lines[desc_row_idx + 1:]:
        stripped = ln.strip()
        if not stripped:
            continue
        # Skip if line doesn't start with a number (might be sub-headers)
        first_tok = stripped.split(",")[0].strip()
        try:
            float(first_tok)
            data_rows.append(stripped)
        except ValueError:
            continue

    if not data_rows:
        raise ValueError("No numeric data rows found after header.")

    # Parse data as DataFrame
    raw_df = pd.read_csv(
        StringIO("\n".join(data_rows)),
        header=None,
        names=range(len(col_descs)),
    )

    # ── Parse column descriptions ───────────────────────────────────
    # Regex patterns for extracting parameters
    _t_pat       = re.compile(r"t\s*=\s*([\d.eE+\-]+)")
    _pc_pat      = re.compile(r"pair_case\s*=\s*([\d.eE+\-]+)")
    _phi_pat     = re.compile(r"Phi_local\s*=\s*([\d.eE+\-]+)")

    col_meta = []  # list of dicts per column
    for i, desc in enumerate(col_descs):
        t_m   = _t_pat.search(desc)
        pc_m  = _pc_pat.search(desc)
        phi_m = _phi_pat.search(desc)

        if t_m and pc_m and phi_m:
            col_meta.append({
                "col_idx":   i,
                "t_ns":      float(t_m.group(1)),
                "pair_case": float(pc_m.group(1)),
                "Phi_local": float(phi_m.group(1)),
            })
        else:
            col_meta.append(None)  # spatial / ignored column

    # ── Group into (pair_case, Phi_local) → (t_ns[], p[]) ──────────
    # Use the first spatial data row (index 0) as the detector point.
    # If multiple rows exist (multiple spatial points), they can be
    # handled by the caller via the raw_df.
    if len(raw_df) == 0:
        raise ValueError("Data DataFrame is empty.")

    # Build lookup: (pair_case, phi) → sorted list of (t_ns, value)
    groups = {}
    for cm in col_meta:
        if cm is None:
            continue
        key = (cm["pair_case"], cm["Phi_local"])
        if key not in groups:
            groups[key] = []
        # Average across spatial rows (or take first row)
        col_values = raw_df.iloc[:, cm["col_idx"]].values
        mean_val = float(np.mean(col_values))
        groups[key].append((cm["t_ns"], mean_val))

    # Sort by time and convert to arrays
    data = {}
    for key, pairs in groups.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        t_arr = np.array([p[0] for p in pairs_sorted])
        p_arr = np.array([p[1] for p in pairs_sorted])
        data[key] = {"t_ns": t_arr, "p": p_arr}

    # ── Build metadata summary ──────────────────────────────────────
    all_pair_cases = sorted(set(k[0] for k in data))
    all_phi_locals = sorted(set(k[1] for k in data))

    meta = {
        "pair_cases":  all_pair_cases,
        "phi_locals":  all_phi_locals,
        "n_keys":      len(data),
        "n_t_points":  len(next(iter(data.values()))["t_ns"]) if data else 0,
    }

    return data, meta, raw_df


def get_available_keys(data):
    """Return sorted list of (pair_case, Phi_local) tuples in data."""
    return sorted(data.keys())


# ═══════════════════════════════════════════════════════════════════════
#  2. Signal Builder
# ═══════════════════════════════════════════════════════════════════════

def build_signal_set(data, phi_local, off_case, on_case, force_case,
                     force_phi=None):
    """Extract OFF, ON, FORCE waveforms and compute DIFF = ON - OFF.

    Parameters
    ----------
    data : dict
        Output of parse_wide_comsol_csv.
    phi_local : float
        Phi_local value for OFF and ON signals.
    off_case : float
        pair_case value for OFF (background) signal.
    on_case : float
        pair_case value for ON (photoacoustic) signal.
    force_case : float
        pair_case value for FORCE-only reference signal.
    force_phi : float or None
        Phi_local to use for FORCE signal.
        If None, uses phi_local (same as ON/OFF).

    Returns
    -------
    signals : dict
        {
          "t_ns": np.ndarray,
          "OFF":  np.ndarray,
          "ON":   np.ndarray,
          "FORCE": np.ndarray or None,
          "DIFF": np.ndarray,
        }
    missing : list of str
        Names of signals that could not be found in data.
    """
    f_phi = force_phi if force_phi is not None else phi_local
    missing = []

    off_key   = (float(off_case),   float(phi_local))
    on_key    = (float(on_case),    float(phi_local))
    force_key = (float(force_case), float(f_phi))

    # Try to retrieve each signal
    off_entry   = data.get(off_key)
    on_entry    = data.get(on_key)
    force_entry = data.get(force_key)

    if off_entry is None:
        missing.append(f"OFF (pair_case={off_case}, Phi_local={phi_local})")
    if on_entry is None:
        missing.append(f"ON (pair_case={on_case}, Phi_local={phi_local})")
    if force_entry is None:
        missing.append(f"FORCE (pair_case={force_case}, Phi_local={f_phi})")

    if off_entry is None or on_entry is None:
        return None, missing

    t_ns = off_entry["t_ns"]

    # Align lengths if they differ (shouldn't happen, but safety)
    n = min(len(off_entry["p"]), len(on_entry["p"]))
    off_sig  = off_entry["p"][:n]
    on_sig   = on_entry["p"][:n]
    t_ns     = t_ns[:n]

    diff_sig = on_sig - off_sig

    force_sig = None
    if force_entry is not None:
        nf = min(n, len(force_entry["p"]))
        force_sig = force_entry["p"][:nf]
        if nf < n:
            # Pad or truncate
            force_sig = np.pad(force_sig, (0, n - nf))

    signals = {
        "t_ns":  t_ns,
        "OFF":   off_sig,
        "ON":    on_sig,
        "FORCE": force_sig,
        "DIFF":  diff_sig,
    }
    return signals, missing


# ═══════════════════════════════════════════════════════════════════════
#  3. Signal Processing
# ═══════════════════════════════════════════════════════════════════════

def apply_time_gate(t_ns, signal, t_start_ns=30.0, t_end_ns=200.0):
    """Extract the time-gated portion of a signal.

    Parameters
    ----------
    t_ns : np.ndarray
    signal : np.ndarray
    t_start_ns : float
    t_end_ns : float

    Returns
    -------
    t_gated : np.ndarray
    sig_gated : np.ndarray
    gate_mask : np.ndarray (bool)
    """
    mask = (t_ns >= t_start_ns) & (t_ns <= t_end_ns)
    return t_ns[mask], signal[mask], mask


def _check_uniform_dt(t_ns):
    """Return mean dt and warn if non-uniform (>1% variation)."""
    if len(t_ns) < 2:
        return 0.0
    dt = np.diff(t_ns)
    dt_mean = np.mean(dt)
    if dt_mean == 0:
        return 0.0
    dt_var = np.max(np.abs(dt - dt_mean)) / dt_mean * 100
    if dt_var > 1.0:
        warnings.warn(f"Non-uniform time step: max variation = {dt_var:.2f}%")
    return float(dt_mean)


def compute_hamming_fft(t_ns, signal):
    """Apply Hamming window and compute real FFT power spectrum.

    Parameters
    ----------
    t_ns : np.ndarray   — time axis in nanoseconds
    signal : np.ndarray — pressure signal

    Returns
    -------
    freq_mhz  : np.ndarray   — frequency axis in MHz
    power     : np.ndarray   — power spectrum  |FFT|²
    amplitude : np.ndarray   — amplitude spectrum |FFT|
    """
    N = len(signal)
    if N < 4:
        raise ValueError(f"Signal too short for FFT: N={N}")

    dt_ns = _check_uniform_dt(t_ns)
    if dt_ns <= 0:
        dt_ns = (t_ns[-1] - t_ns[0]) / max(N - 1, 1)

    dt_s = dt_ns * 1e-9  # convert ns → s

    window    = np.hamming(N)
    win_norm  = np.sum(window ** 2)  # for power normalization
    fft_vals  = np.fft.rfft(signal * window)
    amplitude = np.abs(fft_vals)
    power     = (amplitude ** 2) / win_norm
    freq_mhz  = np.fft.rfftfreq(N, d=dt_s) / 1e6

    return freq_mhz, power, amplitude


def power_to_db(power, ref=None, floor_db=-80.0):
    """Convert power spectrum to dB scale.

    Parameters
    ----------
    power : np.ndarray
    ref : float or None
        Reference power. If None, uses the maximum of `power`.
    floor_db : float
        Minimum dB value (prevents -inf). Default -80 dB.

    Returns
    -------
    power_db : np.ndarray
    ref_used : float
    """
    if ref is None:
        ref = np.max(power)
    if ref <= 0:
        ref = 1.0
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(power / ref, 1e-30))
    power_db = np.maximum(db, floor_db)
    return power_db, ref


def apply_detector_response(freq_mhz, power_db,
                            center_mhz=5.0, bw_fraction=0.60,
                            rolloff_db=-6.0):
    """Apply a Gaussian detector frequency response (transfer function).

    The -6 dB point is at center ± (bw_fraction/2 × center_mhz).

    Parameters
    ----------
    freq_mhz : np.ndarray
    power_db : np.ndarray
    center_mhz : float
        Detector centre frequency in MHz.
    bw_fraction : float
        Fractional bandwidth at rolloff_db (e.g. 0.60 → ±30% of centre).
    rolloff_db : float
        dB level at the bandwidth edge (typically -6.0).

    Returns
    -------
    filtered_db : np.ndarray   — detector-weighted power in dB
    response_db : np.ndarray   — detector transfer function in dB
    """
    # Gaussian sigma derived from -6 dB point
    half_bw = (bw_fraction / 2.0) * center_mhz  # half-bandwidth in MHz
    # For Gaussian: G(f) = exp(-(f-fc)² / (2σ²))
    # At f = fc + half_bw: G = exp(-half_bw²/(2σ²)) = 10^(rolloff_db/20)
    # → σ² = -half_bw² / (2 * ln(10^(rolloff_db/20)))
    ln_val = (rolloff_db / 20.0) * np.log(10.0)  # negative number
    sigma2 = -(half_bw ** 2) / (2.0 * ln_val)
    sigma  = np.sqrt(sigma2)

    response_linear = np.exp(-((freq_mhz - center_mhz) ** 2) / (2.0 * sigma2))
    response_db     = 20.0 * np.log10(np.maximum(response_linear, 1e-30))

    filtered_db = power_db + response_db
    return filtered_db, response_db


# ═══════════════════════════════════════════════════════════════════════
#  4. Spectral Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_spectral_metrics(freq_mhz, power_db,
                             band_low=2.0, band_high=8.0):
    """Compute Hysi-style spectral metrics.

    Parameters
    ----------
    freq_mhz : np.ndarray
    power_db : np.ndarray
    band_low, band_high : float
        Frequency band for mean power computation (MHz).

    Returns
    -------
    metrics : dict
        peak_freq_mhz      — frequency of spectral peak (excluding DC)
        peak_power_db      — dB value at peak
        mean_power_band_db — mean dB in [band_low, band_high] MHz
        band_low_mhz       — band_low used
        band_high_mhz      — band_high used
        rms_amplitude      — RMS of linear amplitude in band
    """
    # Exclude DC (index 0)
    ndc_mask = freq_mhz > 0
    f_ndc = freq_mhz[ndc_mask]
    p_ndc = power_db[ndc_mask]

    if len(f_ndc) == 0:
        return {
            "peak_freq_mhz": 0.0,
            "peak_power_db": float("nan"),
            "mean_power_band_db": float("nan"),
            "band_low_mhz": band_low,
            "band_high_mhz": band_high,
            "rms_amplitude": float("nan"),
        }

    peak_idx      = np.argmax(p_ndc)
    peak_freq     = float(f_ndc[peak_idx])
    peak_power    = float(p_ndc[peak_idx])

    band_mask     = (f_ndc >= band_low) & (f_ndc <= band_high)
    if np.any(band_mask):
        mean_band_db = float(np.mean(p_ndc[band_mask]))
        # RMS of linear amplitude in band
        lin_amp      = 10.0 ** (p_ndc[band_mask] / 20.0)
        rms_amp      = float(np.sqrt(np.mean(lin_amp ** 2)))
    else:
        mean_band_db = float("nan")
        rms_amp      = float("nan")

    return {
        "peak_freq_mhz":      peak_freq,
        "peak_power_db":      peak_power,
        "mean_power_band_db": mean_band_db,
        "band_low_mhz":       band_low,
        "band_high_mhz":      band_high,
        "rms_amplitude":      rms_amp,
    }


def compute_relative_metrics(off_metrics, diff_metrics, force_metrics=None):
    """Compute DIFF-vs-OFF and FORCE-vs-OFF relative metrics.

    Parameters
    ----------
    off_metrics : dict   — from compute_spectral_metrics for OFF signal
    diff_metrics : dict  — from compute_spectral_metrics for DIFF signal
    force_metrics : dict or None

    Returns
    -------
    dict
    """
    result = {
        "DIFF_minus_OFF_peak_dB": (
            diff_metrics["peak_power_db"] - off_metrics["peak_power_db"]
        ),
        "DIFF_minus_OFF_band_dB": (
            diff_metrics["mean_power_band_db"] - off_metrics["mean_power_band_db"]
        ),
    }
    if force_metrics is not None:
        result["FORCE_minus_OFF_peak_dB"] = (
            force_metrics["peak_power_db"] - off_metrics["peak_power_db"]
        )
        result["FORCE_minus_OFF_band_dB"] = (
            force_metrics["mean_power_band_db"] - off_metrics["mean_power_band_db"]
        )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  5. Full Single-Combo Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_sweep_pipeline(data, phi_local,
                       off_case, on_case, force_case, force_phi=20.0,
                       t_start_ns=30.0, t_end_ns=200.0,
                       detector_center_mhz=5.0,
                       detector_bw_fraction=0.60,
                       band_low=2.0, band_high=8.0):
    """Run the full Hysi-style spectral pipeline for one Phi_local value.

    Parameters
    ----------
    data : dict         — from parse_wide_comsol_csv
    phi_local : float
    off_case, on_case, force_case : float   — pair_case values
    force_phi : float   — Phi_local for FORCE signal
    t_start_ns, t_end_ns : float   — time gate
    detector_center_mhz : float
    detector_bw_fraction : float
    band_low, band_high : float   — spectral analysis band (MHz)

    Returns
    -------
    result : dict  — all arrays and metrics, keyed by signal name
    missing : list of str  — any signals not found
    """
    # 1. Build signals
    signals, missing = build_signal_set(
        data, phi_local, off_case, on_case, force_case, force_phi
    )
    if signals is None:
        return None, missing

    t_ns  = signals["t_ns"]
    names = ["OFF", "ON", "DIFF"] + (["FORCE"] if signals["FORCE"] is not None else [])

    result = {
        "phi_local":  phi_local,
        "signals_raw": signals,
        "gate":       {"t_start": t_start_ns, "t_end": t_end_ns},
        "spectra":    {},
        "metrics":    {},
    }

    # Shared reference power for dB (use max OFF power for consistent scale)
    ref_power = None

    for name in names:
        sig = signals[name]
        if sig is None:
            continue

        # 2. Time gate
        t_g, sig_g, _ = apply_time_gate(t_ns, sig, t_start_ns, t_end_ns)
        if len(t_g) < 4:
            warnings.warn(f"[{name}] Too few points after time gate: {len(t_g)}")
            continue

        # 3. FFT
        freq_mhz, power, amplitude = compute_hamming_fft(t_g, sig_g)

        # 4. dB — establish reference from OFF signal
        if name == "OFF":
            power_db, ref_power = power_to_db(power, ref=None)
        else:
            power_db, _ = power_to_db(power, ref=ref_power)

        # 5. Detector response
        filtered_db, response_db = apply_detector_response(
            freq_mhz, power_db,
            center_mhz=detector_center_mhz,
            bw_fraction=detector_bw_fraction,
        )

        # 6. Metrics
        metrics_raw = compute_spectral_metrics(
            freq_mhz, power_db, band_low, band_high
        )
        metrics_flt = compute_spectral_metrics(
            freq_mhz, filtered_db, band_low, band_high
        )

        result["spectra"][name] = {
            "t_gated":     t_g,
            "sig_gated":   sig_g,
            "freq_mhz":    freq_mhz,
            "power":       power,
            "amplitude":   amplitude,
            "power_db":    power_db,
            "filtered_db": filtered_db,
            "response_db": response_db,
        }
        result["metrics"][name] = {
            "raw":      metrics_raw,
            "filtered": metrics_flt,
        }

    # 7. Relative metrics
    if "OFF" in result["metrics"] and "DIFF" in result["metrics"]:
        result["relative"] = compute_relative_metrics(
            result["metrics"]["OFF"]["raw"],
            result["metrics"]["DIFF"]["raw"],
            result["metrics"].get("FORCE", {}).get("raw"),
        )
    else:
        result["relative"] = {}

    return result, missing


# ═══════════════════════════════════════════════════════════════════════
#  6. Plotting Helpers
# ═══════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Signal colour map
_SIG_COLORS = {
    "OFF":   "#5b8dee",   # blue
    "ON":    "#e05c5c",   # red
    "DIFF":  "#2ecc71",   # green
    "FORCE": "#f39c12",   # orange
}


def plot_time_domain_sweep(signals, phi_local, t_start_ns=30.0, t_end_ns=200.0,
                           title_prefix=""):
    """Plot raw time-domain waveforms for OFF/ON/DIFF/FORCE.

    Returns matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#131320")

    t_ns = signals["t_ns"]

    # Top: OFF, ON, FORCE
    for name in ["OFF", "ON", "FORCE"]:
        sig = signals.get(name)
        if sig is None:
            continue
        axes[0].plot(t_ns, sig * 1e3,  # Pa → mPa
                     label=name, color=_SIG_COLORS[name],
                     linewidth=1.4, alpha=0.9)

    # Bottom: DIFF
    diff = signals.get("DIFF")
    if diff is not None:
        axes[1].plot(t_ns, diff * 1e3,
                     label="DIFF (ON − OFF)",
                     color=_SIG_COLORS["DIFF"],
                     linewidth=1.4)

    # Gate shading on both panels
    for ax in axes:
        ax.axvspan(t_start_ns, t_end_ns, alpha=0.08,
                   color="#ffffff", label="Time gate")
        ax.axvline(t_start_ns, color="#aaaaaa", linestyle="--",
                   linewidth=0.8, alpha=0.6)
        ax.axvline(t_end_ns,   color="#aaaaaa", linestyle="--",
                   linewidth=0.8, alpha=0.6)

    # Styling
    for ax in axes:
        ax.set_ylabel("Pressure (mPa)", color="#cccccc", fontsize=11)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466")
        ax.legend(fontsize=10, facecolor="#1a1a2e",
                  labelcolor="#cccccc", framealpha=0.8)

    axes[1].set_xlabel("Time (ns)", color="#cccccc", fontsize=11)
    axes[0].set_title(
        f"{title_prefix}  |  Φ_local = {phi_local}  |  Time-Domain Waveforms",
        color="#e0e0ff", fontsize=13, pad=10
    )
    fig.tight_layout()
    return fig


def plot_power_spectra_db(result_spectra, phi_local,
                          freq_max=15.0, title_prefix="",
                          band_low=2.0, band_high=8.0):
    """Plot power spectra in dB for OFF / DIFF / FORCE.

    Returns matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#131320")

    names_left  = ["OFF", "ON", "DIFF", "FORCE"]
    names_right = ["OFF", "ON", "DIFF", "FORCE"]

    for name in names_left:
        sp = result_spectra.get(name)
        if sp is None:
            continue
        freq = sp["freq_mhz"]
        mask = freq <= freq_max
        axes[0].plot(freq[mask], sp["power_db"][mask],
                     label=name, color=_SIG_COLORS[name],
                     linewidth=1.5, alpha=0.9)

    for name in names_right:
        sp = result_spectra.get(name)
        if sp is None:
            continue
        freq = sp["freq_mhz"]
        mask = freq <= freq_max
        axes[1].plot(freq[mask], sp["filtered_db"][mask],
                     label=f"{name} (det.)",
                     color=_SIG_COLORS[name],
                     linewidth=1.5, alpha=0.9)

    # Band shading
    for ax in axes:
        ax.axvspan(band_low, band_high, alpha=0.07,
                   color="#8888ff", label=f"{band_low}–{band_high} MHz band")
        ax.set_xlabel("Frequency (MHz)", color="#cccccc", fontsize=11)
        ax.set_ylabel("Power (dB, rel.)", color="#cccccc", fontsize=11)
        ax.set_xlim(0, freq_max)
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333355")
        ax.grid(True, alpha=0.15, color="#444466")
        ax.legend(fontsize=9, facecolor="#1a1a2e",
                  labelcolor="#cccccc", framealpha=0.8)

    axes[0].set_title("Raw Power Spectrum (dB)", color="#e0e0ff", fontsize=12)
    axes[1].set_title("Detector-Filtered Spectrum (dB)", color="#e0e0ff", fontsize=12)

    fig.suptitle(
        f"{title_prefix}  |  Φ_local = {phi_local}  |  Power Spectra",
        color="#e0e0ff", fontsize=13, y=1.01
    )
    fig.tight_layout()
    return fig


def plot_detector_response_curve(freq_mhz, response_db,
                                 center_mhz=5.0, freq_max=15.0):
    """Plot detector transfer function.

    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#131320")

    mask = freq_mhz <= freq_max
    ax.plot(freq_mhz[mask], response_db[mask],
            color="#a78bfa", linewidth=2.0, label="Detector response (dB)")
    ax.axhline(-6.0, color="#f39c12", linestyle="--",
               linewidth=1.2, alpha=0.7, label="-6 dB")
    ax.axvline(center_mhz, color="#5b8dee", linestyle=":",
               linewidth=1.2, alpha=0.7, label=f"f_c = {center_mhz} MHz")

    ax.set_xlabel("Frequency (MHz)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Response (dB)", color="#cccccc", fontsize=11)
    ax.set_title("Detector Frequency Response", color="#e0e0ff", fontsize=12)
    ax.set_xlim(0, freq_max)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#333355")
    ax.grid(True, alpha=0.15, color="#444466")
    ax.legend(fontsize=10, facecolor="#1a1a2e",
              labelcolor="#cccccc", framealpha=0.8)
    fig.tight_layout()
    return fig
