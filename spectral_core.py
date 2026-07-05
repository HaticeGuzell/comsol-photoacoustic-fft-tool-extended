#!/usr/bin/env python3
"""Configurable FFT and detector-response processing."""

from __future__ import annotations

import warnings

import numpy as np


WINDOWS = ("Rectangular", "Hann", "Hamming", "Tukey", "Blackman")
DETREND_MODES = ("None", "Mean subtraction", "Linear detrend", "Baseline interval")


def _window(name, length, tukey_alpha=0.25):
    key = str(name).lower()
    if key == "rectangular":
        return np.ones(length)
    if key == "hann":
        return np.hanning(length)
    if key == "hamming":
        return np.hamming(length)
    if key == "tukey":
        if tukey_alpha <= 0:
            return np.ones(length)
        if tukey_alpha >= 1:
            return np.hanning(length)
        x = np.linspace(0.0, 1.0, length)
        result = np.ones(length)
        first = x < tukey_alpha / 2.0
        last = x >= 1.0 - tukey_alpha / 2.0
        result[first] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * x[first] / tukey_alpha - 1.0))
        )
        result[last] = 0.5 * (
            1.0 + np.cos(
                np.pi * (2.0 * x[last] / tukey_alpha - 2.0 / tukey_alpha + 1.0)
            )
        )
        return result
    if key == "blackman":
        return np.blackman(length)
    raise ValueError(f"Unknown FFT window: {name}")


def _detrend(time_ns, pressure, mode, baseline_end_ns=None):
    key = str(mode).lower()
    if key == "none":
        return pressure.copy()
    if key == "mean subtraction":
        return pressure - np.mean(pressure)
    if key == "linear detrend":
        slope, intercept = np.polyfit(time_ns, pressure, 1)
        return pressure - (slope * time_ns + intercept)
    if key == "baseline interval":
        if baseline_end_ns is None:
            raise ValueError("Baseline interval detrending requires baseline_end_ns.")
        mask = time_ns <= baseline_end_ns
        if not np.any(mask):
            raise ValueError("Baseline interval contains no samples.")
        return pressure - np.mean(pressure[mask])
    raise ValueError(f"Unknown detrend mode: {mode}")


def resolve_nfft(length, mode="Signal length", custom_nfft=None):
    mode = str(mode).lower()
    if mode == "signal length":
        return length
    if mode == "next power of 2":
        return 1 << int(np.ceil(np.log2(max(length, 1))))
    if mode == "custom":
        if custom_nfft is None or int(custom_nfft) < length:
            raise ValueError("Custom NFFT must be at least the gated signal length.")
        return int(custom_nfft)
    raise ValueError(f"Unknown NFFT mode: {mode}")


def detector_response(freq_mhz, config):
    """Return amplitude transfer function for a detector configuration."""
    mode = str(config.get("mode", "None / raw")).lower()
    freq = np.asarray(freq_mhz, dtype=float)
    if mode.startswith("none"):
        return np.ones_like(freq)
    if "gaussian" in mode or mode.startswith("hysi"):
        center = float(config.get("center_mhz", 5.0))
        fraction = float(config.get("bw_fraction", 0.60))
        half_bw = max(fraction * center / 2.0, 1e-12)
        sigma_sq = -(half_bw ** 2) / (2.0 * np.log(10 ** (-6.0 / 20.0)))
        return np.exp(-((freq - center) ** 2) / (2.0 * sigma_sq))
    if "bandpass" in mode:
        low = float(config.get("low_mhz", 2.0))
        high = float(config.get("high_mhz", 8.0))
        return ((freq >= low) & (freq <= high)).astype(float)
    raise ValueError(f"Unknown detector mode: {config.get('mode')}")


def compute_spectrum(trace, config):
    """Compute an unnormalized raw and detector-filtered power spectrum."""
    time_ns = np.asarray(trace["time_ns"], dtype=float)
    pressure = np.asarray(trace["pressure_pa"], dtype=float)
    gate_start = config.get("gate_start_ns")
    gate_end = config.get("gate_end_ns")
    if gate_start is not None or gate_end is not None:
        low = -np.inf if gate_start is None else float(gate_start)
        high = np.inf if gate_end is None else float(gate_end)
        mask = (time_ns >= low) & (time_ns <= high)
        time_ns, pressure = time_ns[mask], pressure[mask]
    if len(time_ns) < 4:
        raise ValueError("At least four samples are required after time gating.")

    order = np.argsort(time_ns)
    time_ns, pressure = time_ns[order], pressure[order]
    dt = np.diff(time_ns)
    dt_mean = float(np.mean(dt))
    if dt_mean <= 0:
        raise ValueError("Time values must be strictly increasing.")
    variation = float(np.max(np.abs(dt - dt_mean)) / dt_mean)
    if variation > 0.01:
        warnings.warn(
            f"Non-uniform time step ({variation:.2%}); interpolating to a uniform grid."
        )
        uniform = np.linspace(time_ns[0], time_ns[-1], len(time_ns))
        pressure = np.interp(uniform, time_ns, pressure)
        time_ns = uniform
        dt_mean = float(np.mean(np.diff(time_ns)))

    processed = _detrend(
        time_ns,
        pressure,
        config.get("detrend", "Mean subtraction"),
        config.get("baseline_end_ns"),
    )
    window = _window(
        config.get("window", "Hamming"),
        len(processed),
        float(config.get("tukey_alpha", 0.25)),
    )
    nfft = resolve_nfft(
        len(processed),
        config.get("nfft_mode", "Signal length"),
        config.get("custom_nfft"),
    )
    fft_values = np.fft.rfft(processed * window, n=nfft)
    coherent_energy = max(float(np.sum(window ** 2)), np.finfo(float).tiny)
    power = np.abs(fft_values) ** 2 / coherent_energy
    amplitude = np.abs(fft_values) / max(float(np.sum(window)), np.finfo(float).tiny)
    freq_mhz = np.fft.rfftfreq(nfft, d=dt_mean * 1e-9) / 1e6
    response = detector_response(freq_mhz, config.get("detector", {}))
    power_filtered = power * response ** 2
    return {
        "freq_mhz": freq_mhz,
        "power": power,
        "power_flt": power_filtered,
        "amplitude": amplitude,
        "response": response,
        "t_gated": time_ns,
        "sig_gated": processed,
        "n_samples": len(processed),
        "nfft": nfft,
        "dt_ns": dt_mean,
        "true_resolution_mhz": 1e3 / (len(processed) * dt_mean),
        "display_bin_spacing_mhz": (
            float(freq_mhz[1] - freq_mhz[0]) if len(freq_mhz) > 1 else np.nan
        ),
        "zero_padded": nfft > len(processed),
    }


def analyze_traces(traces, config):
    results, errors = [], []
    for trace in traces:
        try:
            results.append(compute_spectrum(trace, config))
        except Exception as exc:
            results.append(None)
            errors.append(f"{trace.get('role', 'trace')}: {exc}")
    return results, errors


def power_to_db(power, ref=None, floor_db=-120.0):
    values = np.asarray(power, dtype=float)
    reference = float(np.max(values)) if ref is None else float(ref)
    reference = max(reference, np.finfo(float).tiny)
    db = 10.0 * np.log10(np.maximum(values / reference, np.finfo(float).tiny))
    return np.maximum(db, floor_db), reference


def ratio_db(power_a, power_b, floor_db=-120.0):
    """Power ratio from physical (unnormalized) power arrays."""
    numerator = np.asarray(power_a, dtype=float)
    denominator = np.asarray(power_b, dtype=float)
    ratio = np.maximum(numerator, np.finfo(float).tiny) / np.maximum(
        denominator, np.finfo(float).tiny
    )
    return np.maximum(10.0 * np.log10(ratio), floor_db)


def interpolate_spectrum(result, target_freq, key="power"):
    return np.interp(
        target_freq,
        result["freq_mhz"],
        result[key],
        left=np.nan,
        right=np.nan,
    )
