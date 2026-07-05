#!/usr/bin/env python3
"""Time-domain, spectral, and ratio metrics for generic traces."""

from __future__ import annotations

import numpy as np

from spectral_core import ratio_db


def time_metrics(trace):
    t = np.asarray(trace["time_ns"], dtype=float)
    p = np.asarray(trace["pressure_pa"], dtype=float)
    if not len(p):
        return {}
    peak_index = int(np.argmax(np.abs(p)))
    return {
        "time_peak_positive_pa": float(np.max(p)),
        "time_peak_negative_pa": float(np.min(p)),
        "time_peak_absolute_pa": float(np.max(np.abs(p))),
        "time_peak_to_peak_pa": float(np.ptp(p)),
        "time_rms_pa": float(np.sqrt(np.mean(p ** 2))),
        "time_peak_arrival_ns": float(t[peak_index]),
    }


def spectral_metrics(freq_mhz, power, band_low=2.0, band_high=8.0,
                     midband_mhz=None):
    f = np.asarray(freq_mhz, dtype=float)
    p = np.asarray(power, dtype=float)
    positive = f > 0
    band = positive & (f >= band_low) & (f <= band_high) & np.isfinite(p)
    if not np.any(positive):
        return {}
    peak_candidates = np.where(positive, p, -np.inf)
    peak_index = int(np.argmax(peak_candidates))
    peak_power = max(float(p[peak_index]), np.finfo(float).tiny)
    half = peak_power * 10 ** (-6.0 / 10.0)
    above = positive & (p >= half)
    bandwidth = (
        float(f[above][-1] - f[above][0]) if np.any(above) else np.nan
    )
    metrics = {
        "peak_frequency_mhz": float(f[peak_index]),
        "peak_power": peak_power,
        "peak_power_self_db": 0.0,
        "bandwidth_6db_mhz": bandwidth,
    }
    if np.any(band):
        band_f, band_p = f[band], p[band]
        band_db = 10.0 * np.log10(
            np.maximum(band_p / peak_power, np.finfo(float).tiny)
        )
        slope, intercept = (
            np.polyfit(band_f, band_db, 1) if len(band_f) >= 2 else (np.nan, np.nan)
        )
        mid = (
            float(midband_mhz)
            if midband_mhz is not None
            else float((band_low + band_high) / 2.0)
        )
        # NumPy compatibility: in newer NumPy versions np.trapz may be removed.
        # Do not use getattr(np, "trapezoid", np.trapz), because the default
        # argument is evaluated immediately and can raise AttributeError.
        if hasattr(np, "trapezoid"):
            integrated_power = np.trapezoid(band_p, band_f)
        else:
            # Small local fallback equivalent to trapezoidal integration.
            integrated_power = np.sum((band_p[1:] + band_p[:-1]) * np.diff(band_f) / 2.0)
        metrics.update({
            "mean_spectral_power": float(np.mean(band_p)),
            "mean_spectral_power_db": float(
                10.0 * np.log10(
                    max(float(np.mean(band_p)) / peak_power, np.finfo(float).tiny)
                )
            ),
            "integrated_power": float(integrated_power),
            "spectral_slope_db_per_mhz": float(slope),
            "spectral_intercept_db": float(intercept),
            "midband_frequency_mhz": mid,
            "midband_value_db": float(slope * mid + intercept),
        })
    return metrics


def trace_metrics(trace, spectrum, band_low=2.0, band_high=8.0,
                  midband_mhz=None):
    row = {}
    row.update(time_metrics(trace))
    for prefix, key in (("raw", "power"), ("filtered", "power_flt")):
        values = spectral_metrics(
            spectrum["freq_mhz"],
            spectrum[key],
            band_low,
            band_high,
            midband_mhz,
        )
        row.update({f"{prefix}_{name}": value for name, value in values.items()})
    row.update({
        "fft_n_samples": spectrum["n_samples"],
        "fft_nfft": spectrum["nfft"],
        "fft_true_resolution_mhz": spectrum["true_resolution_mhz"],
        "fft_display_bin_spacing_mhz": spectrum["display_bin_spacing_mhz"],
    })
    return row


def build_metrics_table(traces, spectra, band_low=2.0, band_high=8.0,
                        midband_mhz=None):
    rows = []
    for index, (trace, spectrum) in enumerate(zip(traces, spectra)):
        if spectrum is None:
            continue
        row = {
            "trace_id": index,
            "role": trace.get("role", ""),
            "source_file": trace.get("source_file", ""),
        }
        row.update({
            key: value for key, value in trace.get("params", {}).items()
            if key != "role"
        })
        row.update(trace_metrics(trace, spectrum, band_low, band_high, midband_mhz))
        rows.append(row)
    return rows


def ratio_metrics(numerator, denominator, band_low=2.0, band_high=8.0,
                  filtered=False):
    key = "power_flt" if filtered else "power"
    freq = denominator["freq_mhz"]
    p_num = np.interp(freq, numerator["freq_mhz"], numerator[key])
    p_den = denominator[key]
    values = ratio_db(p_num, p_den)
    band = (freq >= band_low) & (freq <= band_high) & np.isfinite(values)
    if not np.any(band):
        return {}
    best = int(np.nanargmax(np.where(band, values, np.nan)))
    midband_frequency = float((band_low + band_high) / 2.0)
    midband_value = float(np.interp(midband_frequency, freq, values))
    return {
        "ratio_domain": "filtered" if filtered else "raw",
        "mean_ratio_db": float(np.nanmean(values[band])),
        "midband_ratio_db": midband_value,
        "best_ratio_db": float(values[best]),
        "best_ratio_frequency_mhz": float(freq[best]),
        "percent_visibility_peak": float(
            100.0 * np.sqrt(
                max(float(np.nanmax(p_num[band])), 0.0)
                / max(float(np.nanmax(p_den[band])), np.finfo(float).tiny)
            )
        ),
    }


def matched_role_pairs(traces, numerator_roles, denominator_role, group_keys):
    """Return uniquely matched trace-index pairs using role and group values."""
    denominator_index = {}
    for trace_index, trace in enumerate(traces):
        if trace.get("role") != denominator_role:
            continue
        params = trace.get("params", {})
        key = tuple((name, params.get(name)) for name in group_keys)
        denominator_index.setdefault(key, []).append(trace_index)

    pairs = []
    for numerator_role in numerator_roles:
        numerator_index = {}
        for trace_index, trace in enumerate(traces):
            if trace.get("role") != numerator_role:
                continue
            params = trace.get("params", {})
            key = tuple((name, params.get(name)) for name in group_keys)
            numerator_index.setdefault(key, []).append(trace_index)
        for key, numerator_matches in numerator_index.items():
            denominator_matches = denominator_index.get(key, [])
            if len(numerator_matches) == 1 and len(denominator_matches) == 1:
                pairs.append((numerator_matches[0], denominator_matches[0], key))
    return pairs


def difference_validation(trace_a, trace_b):
    """Compare two time traces, interpolating B onto A over their overlap."""
    ta = np.asarray(trace_a["time_ns"], dtype=float)
    tb = np.asarray(trace_b["time_ns"], dtype=float)
    pa = np.asarray(trace_a["pressure_pa"], dtype=float)
    pb = np.asarray(trace_b["pressure_pa"], dtype=float)
    low, high = max(np.min(ta), np.min(tb)), min(np.max(ta), np.max(tb))
    mask = (ta >= low) & (ta <= high)
    if not np.any(mask):
        return {}
    left = pa[mask]
    right = np.interp(ta[mask], tb, pb)
    error = left - right
    reference = max(float(np.max(np.abs(right))), np.finfo(float).tiny)
    correlation = (
        float(np.corrcoef(left, right)[0, 1])
        if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0
        else np.nan
    )
    return {
        "validation_max_error_pa": float(np.max(np.abs(error))),
        "validation_rms_error_pa": float(np.sqrt(np.mean(error ** 2))),
        "validation_normalized_max_error_percent": float(
            100.0 * np.max(np.abs(error)) / reference
        ),
        "validation_correlation": correlation,
    }
