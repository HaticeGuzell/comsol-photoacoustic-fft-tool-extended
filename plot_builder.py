#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_builder.py — Overlay + Ratio Plot Builder
==============================================

Provides flexible, publication-quality plot generation for:
  - Overlay time-domain waveforms (any combination of traces)
  - Overlay power spectra in dB
  - Ratio spectra (A/B from unnormalized power, REDLINE 6)
  - Detector response curve
  - Layout options: single axis, separate subplots, small multiples

All plots support two themes: 'dark' (app) and 'light' (publication).
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from spectral_core import ratio_db, power_to_db, interpolate_spectrum


# ───────────────────────────────────────────────────────────────────────
#  Colour palettes
# ───────────────────────────────────────────────────────────────────────

_DARK_COLORS  = ["#5b8dee", "#e05c5c", "#2ecc71", "#f39c12", "#a78bfa",
                 "#38bdf8", "#fb7185", "#34d399", "#fbbf24", "#c084fc"]
_LIGHT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

_ROLE_COLORS = {
    "OFF":           0,
    "ON":            1,
    "DIFF":          2,
    "FORCE":         3,
    "Dispersed_OFF": 0,
    "Aggregated_OFF":1,
    "Aggregated_ON": 2,
    "Delta_structure":3,
    "Delta_force":   4,
    "Delta_total":   5,
}


def _get_colors(theme):
    return _DARK_COLORS if theme == "dark" else _LIGHT_COLORS


def _role_color(role_name, theme):
    palette = _get_colors(theme)
    idx = _ROLE_COLORS.get(role_name, hash(role_name) % len(palette))
    return palette[idx % len(palette)]


def format_phi(value):
    """Format fluence values for publication legends."""
    try:
        return f"{float(value):g} mJ/cm²"
    except (TypeError, ValueError):
        return str(value)


def format_detector(value):
    return str(value).capitalize()


def format_role(role):
    mapping = {
        "OFF": "OFF",
        "ON": "ON",
        "FORCE": "FORCE-only",
        "DIFF": "ON − OFF",
    }
    return mapping.get(role, str(role))


def _apply_theme(fig, axes, theme, grid=True):
    """Apply dark or light theme to figure and axes."""
    if theme == "dark":
        fig.patch.set_facecolor("#0f0f1a")
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor("#131320")
            ax.tick_params(colors="#aaaaaa")
            ax.spines[:].set_color("#333355")
            ax.grid(grid, alpha=0.15, color="#444466")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#e0e0ff")
    else:
        fig.patch.set_facecolor("white")
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor("white")
            ax.tick_params(colors="black")
            for sp in ax.spines.values():
                sp.set_color("#333333")
            ax.grid(grid, alpha=0.3, color="#cccccc")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")


def _legend_kwargs(theme, style=None):
    style = style or {}
    position = style.get("legend_position", "Best")
    if position == "None":
        return None
    if theme == "dark":
        kwargs = {
            "facecolor": "#1a1a2e",
            "labelcolor": "#cccccc",
            "framealpha": 0.85,
        }
    else:
        kwargs = {
            "facecolor": "white",
            "labelcolor": "black",
            "framealpha": 0.9,
            "edgecolor": "#cccccc",
        }
    location_map = {
        "Best": "best",
        "Upper right": "upper right",
        "Upper left": "upper left",
        "Lower right": "lower right",
        "Lower left": "lower left",
        "Center right": "center right",
    }
    if position == "Outside right":
        kwargs.update({"loc": "center left", "bbox_to_anchor": (1.02, 0.5)})
    else:
        kwargs["loc"] = location_map.get(position, "best")
    kwargs["fontsize"] = float(style.get("font_size", 11)) * 0.82
    legend_title = str(style.get("legend_title", "")).strip()
    if legend_title:
        kwargs["title"] = legend_title
    return kwargs


def _style_fig_size(style, default):
    style = style or {}
    value = style.get("fig_size")
    if value and len(value) == 2:
        return tuple(float(item) for item in value)
    return default


def _apply_axis_style(ax, style, default_xlabel, default_ylabel):
    style = style or {}
    font_size = float(style.get("font_size", 11))
    ax.set_xlabel(style.get("xlabel") or default_xlabel, fontsize=font_size)
    ax.set_ylabel(style.get("ylabel") or default_ylabel, fontsize=font_size)
    ax.tick_params(labelsize=max(font_size - 1, 1))
    xlim = style.get("xlim")
    ylim = style.get("ylim")
    if xlim and any(value is not None for value in xlim):
        current = ax.get_xlim()
        ax.set_xlim(
            current[0] if xlim[0] is None else xlim[0],
            current[1] if xlim[1] is None else xlim[1],
        )
    if ylim and any(value is not None for value in ylim):
        current = ax.get_ylim()
        ax.set_ylim(
            current[0] if ylim[0] is None else ylim[0],
            current[1] if ylim[1] is None else ylim[1],
        )


def _draw_legend(ax, theme, style):
    kwargs = _legend_kwargs(theme, style)
    if kwargs is not None:
        ax.legend(**kwargs)


# ───────────────────────────────────────────────────────────────────────
#  Curve descriptor
# ───────────────────────────────────────────────────────────────────────

def _trace_label(trace, label_fields=None):
    """Build a display label from trace params."""
    params = trace.get("params", {})
    role   = trace.get("role") or params.get("role", "")
    if label_fields:
        parts = [role] if role else []
        for f in label_fields:
            if f in params:
                parts.append(f"{f}={params[f]}")
        return " | ".join(parts) if parts else str(params)
    # Auto: role + detector + key sweep var
    parts = [role] if role else []
    for key in ["detector", "Phi_local", "N_RBC", "agg_state"]:
        if key in params:
            parts.append(f"{params[key]}")
    return " | ".join(parts) if parts else str(params)


# ───────────────────────────────────────────────────────────────────────
#  1.  Time-domain overlay
# ───────────────────────────────────────────────────────────────────────

def plot_time_overlay(
    trace_list,
    t_start_ns=None, t_end_ns=None,
    y_unit="Pa",
    layout="overlay",
    title="Time-Domain Waveforms",
    theme="dark",
    label_fields=None,
    labels=None,
    fig_size=(13, 6),
    style=None,
):
    """Plot time-domain waveforms for any selection of traces.

    Parameters
    ----------
    trace_list : list of Trace dicts
    t_start_ns, t_end_ns : float or None  — gate markers
    y_unit : 'Pa' or 'mPa'
    layout : 'overlay' (single axis) or 'subplots'
    theme : 'dark' or 'light'
    label_fields : list of str or None

    Returns matplotlib Figure.
    """
    style = style or {}
    fig_size = _style_fig_size(style, fig_size)
    n = len(trace_list)
    if n == 0:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, "No traces selected", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    colors = _get_colors(theme)

    if layout == "subplots" and n > 1:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(fig_size[0], fig_size[1] * nrows / max(1, nrows - 0.5)))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    else:
        fig, ax_single = plt.subplots(figsize=fig_size)
        axes_flat = [ax_single] * n

    scale = 1e3 if y_unit == "mPa" else 1.0

    for i, tr in enumerate(trace_list):
        ax  = axes_flat[i] if layout == "subplots" else axes_flat[0]
        col = _role_color(tr.get("role", ""), theme) if tr.get("role") else colors[i % len(colors)]
        lbl = (
            labels[i]
            if labels is not None and i < len(labels)
            else _trace_label(tr, label_fields)
        )
        t   = tr["time_ns"]
        p   = tr["pressure_pa"] * scale
        ax.plot(
            t, p, label=lbl, color=col,
            linewidth=float(style.get("line_width", 2.0)), alpha=0.9,
        )

        if layout == "subplots":
            ax.set_title(lbl, fontsize=9)

    # Gate markers
    axes_to_mark = (list(set(axes_flat))
                    if layout == "overlay" else axes_flat[:n])
    for ax in axes_to_mark:
        if t_start_ns is not None:
            ax.axvline(t_start_ns, color="#aaaaaa", linestyle="--",
                       linewidth=0.8, alpha=0.5, label="_nolegend_")
        if t_end_ns is not None:
            ax.axvline(t_end_ns, color="#aaaaaa", linestyle=":",
                       linewidth=0.8, alpha=0.5, label="_nolegend_")
        if t_start_ns is not None and t_end_ns is not None:
            ax.axvspan(t_start_ns, t_end_ns, alpha=0.05,
                       color="#ffffff" if theme == "dark" else "#000000")
        _apply_axis_style(ax, style, "Time (ns)", f"Pressure ({y_unit})")
        if layout == "overlay":
            _draw_legend(ax, theme, style)

    # Title
    all_axes = (list(set(axes_flat)) if layout == "overlay" else axes_flat[:n])
    if layout == "overlay":
        all_axes[0].set_title(
            style.get("title") or title,
            fontsize=float(style.get("font_size", 11)) + 1,
        )
    else:
        fig.suptitle(
            style.get("title") or title,
            fontsize=float(style.get("font_size", 11)) + 1,
        )

    _apply_theme(fig, all_axes, theme, grid=style.get("grid", True))
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────
#  2.  Power spectrum overlay (dB)
# ───────────────────────────────────────────────────────────────────────

def plot_spectrum_overlay(
    spectral_results_list,       # list of SpectralResult dicts
    labels=None,                 # list of str or None
    use_filtered=False,          # True = detector-filtered, False = raw
    freq_max=15.0,
    band_low=2.0, band_high=8.0,
    db_ref_mode="self",          # 'self', 'global_max', 'absolute', or float
    y_label="Power (dB, rel.)",
    title="Power Spectrum",
    layout="overlay",
    theme="dark",
    fig_size=(13, 6),
    style=None,
):
    """Overlay power spectra from multiple SpectralResult dicts.

    REDLINE 6: ratio_db computed before this function is called.
    dB here is visual only.
    """
    style = style or {}
    fig_size = _style_fig_size(style, fig_size)
    n = len(spectral_results_list)
    if n == 0:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, "No spectra to plot", ha="center",
                va="center", transform=ax.transAxes)
        return fig

    colors = _get_colors(theme)

    # Determine global reference power for consistent scale
    if db_ref_mode == "global_max":
        all_powers = []
        for sr in spectral_results_list:
            p = sr["power_flt"] if use_filtered else sr["power"]
            all_powers.append(float(np.max(p)))
        global_ref = max(all_powers) if all_powers else 1.0
    elif isinstance(db_ref_mode, (int, float)):
        global_ref = float(db_ref_mode)
    elif db_ref_mode == "absolute":
        global_ref = 1.0
    else:
        global_ref = None

    if layout == "subplots" and n > 1:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(
            fig_size[0], fig_size[1] * max(1, nrows) * 0.7))
        axes_flat = np.array(axes).flatten()
    else:
        fig, ax_single = plt.subplots(figsize=fig_size)
        axes_flat = [ax_single] * n

    for i, sr in enumerate(spectral_results_list):
        ax     = axes_flat[i] if layout == "subplots" else axes_flat[0]
        col    = colors[i % len(colors)]
        lbl    = labels[i] if labels and i < len(labels) else f"Trace {i+1}"
        freq   = sr["freq_mhz"]
        power  = sr["power_flt"] if use_filtered else sr["power"]
        mask   = freq <= freq_max

        ref = global_ref if global_ref is not None else None
        p_db, _ = power_to_db(power, ref=ref)

        ax.plot(
            freq[mask], p_db[mask], label=lbl, color=col,
            linewidth=float(style.get("line_width", 2.0)), alpha=0.9,
        )

        if layout == "subplots":
            ax.set_title(lbl, fontsize=9)

    # Band shading + labels
    all_axes = (list(set(axes_flat)) if layout == "overlay" else axes_flat[:n])
    for ax in all_axes:
        if style.get("show_band_shading", True):
            band_label = "_nolegend_"
            if style.get("show_band_legend", False):
                band_label = (
                    style.get("band_label")
                    or f"{band_low:g}–{band_high:g} MHz analysis band"
                )
            ax.axvspan(
                band_low, band_high, alpha=0.06,
                color="#8888ff" if theme == "dark" else "#4444aa",
                label=band_label,
            )
        ax.set_xlim(0, freq_max)
        _apply_axis_style(ax, style, "Frequency (MHz)", y_label)
        if layout == "overlay":
            _draw_legend(ax, theme, style)
        else:
            _draw_legend(ax, theme, style)

    if layout == "overlay":
        all_axes[0].set_title(
            style.get("title") or title,
            fontsize=float(style.get("font_size", 11)) + 1,
        )
    else:
        fig.suptitle(
            style.get("title") or title,
            fontsize=float(style.get("font_size", 11)) + 1,
        )

    _apply_theme(fig, all_axes, theme, grid=style.get("grid", True))
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────
#  3.  Ratio plot (REDLINE 6: from unnormalized power)
# ───────────────────────────────────────────────────────────────────────

def plot_ratio(
    sr_numerator_list,    # list of SpectralResult
    sr_denominator,       # single SpectralResult (reference)
    labels=None,
    use_filtered=False,
    freq_max=15.0,
    band_low=2.0, band_high=8.0,
    ratio_type="power_db",    # 'power_db', 'amplitude', 'percent'
    title="Ratio Spectrum",
    theme="dark",
    fig_size=(12, 5),
    style=None,
):
    """Plot ratio spectra A/B from unnormalized power.

    REDLINE 6: ratio computed from unnormalized power, not visual dB.
    """
    style = style or {}
    fig_size = _style_fig_size(style, fig_size)
    n = len(sr_numerator_list)
    if n == 0 or sr_denominator is None:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    colors = _get_colors(theme)
    fig, ax = plt.subplots(figsize=fig_size)

    p_key   = "power_flt" if use_filtered else "power"
    p_ref   = sr_denominator[p_key]
    freq    = sr_denominator["freq_mhz"]
    mask    = freq <= freq_max

    for i, sr_a in enumerate(sr_numerator_list):
        col = colors[i % len(colors)]
        lbl = labels[i] if labels and i < len(labels) else f"Numerator {i+1}"
        p_a = interpolate_spectrum(sr_a, freq, p_key)

        if ratio_type == "power_db":
            ratio_vals = ratio_db(p_a, p_ref)
            y_label = "Power Ratio (dB)"
            ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
        elif ratio_type == "amplitude":
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_vals = np.where(p_ref > 0,
                                      np.sqrt(p_a / np.where(p_ref > 0, p_ref, 1.0)),
                                      0.0)
            y_label = "Amplitude Ratio"
        elif ratio_type == "percent":
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_vals = np.where(p_ref > 0,
                                      100.0 * np.sqrt(p_a / np.where(p_ref > 0, p_ref, 1.0)),
                                      0.0)
            y_label = "Visibility (%)"
        else:
            ratio_vals = ratio_db(p_a, p_ref)
            y_label = "Power Ratio (dB)"

        ax.plot(
            freq[mask], ratio_vals[mask], label=lbl, color=col,
            linewidth=float(style.get("line_width", 2.0)), alpha=0.9,
        )

    if style.get("show_band_shading", True):
        band_label = "_nolegend_"
        if style.get("show_band_legend", False):
            band_label = (
                style.get("band_label")
                or f"{band_low:g}–{band_high:g} MHz analysis band"
            )
        ax.axvspan(
            band_low, band_high, alpha=0.06,
            color="#8888ff" if theme == "dark" else "#4444aa",
            label=band_label,
        )
    ax.set_xlim(0, freq_max)
    _apply_axis_style(ax, style, "Frequency (MHz)", y_label)
    ax.set_title(
        style.get("title") or title,
        fontsize=float(style.get("font_size", 11)) + 1,
    )
    _draw_legend(ax, theme, style)
    _apply_theme(fig, [ax], theme, grid=style.get("grid", True))
    fig.tight_layout()
    return fig


def plot_ratio_pairs(
    spectrum_pairs,
    labels=None,
    use_filtered=False,
    freq_max=15.0,
    band_low=2.0,
    band_high=8.0,
    ratio_type="power_db",
    title="Ratio Spectrum",
    theme="dark",
    fig_size=(12, 5),
    style=None,
):
    """Overlay ratios where every numerator has its own matched denominator."""
    style = style or {}
    fig_size = _style_fig_size(style, fig_size)
    if not spectrum_pairs:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, "No matched ratio pairs", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    colors = _get_colors(theme)
    fig, ax = plt.subplots(figsize=fig_size)
    power_key = "power_flt" if use_filtered else "power"
    y_label = "Power Ratio (dB)"

    for index, (numerator, denominator) in enumerate(spectrum_pairs):
        freq = denominator["freq_mhz"]
        p_num = interpolate_spectrum(numerator, freq, power_key)
        p_den = denominator[power_key]
        if ratio_type == "amplitude":
            ratio_values = np.sqrt(
                np.maximum(p_num, 0.0)
                / np.maximum(p_den, np.finfo(float).tiny)
            )
            y_label = "Amplitude Ratio"
        elif ratio_type == "percent":
            ratio_values = 100.0 * np.sqrt(
                np.maximum(p_num, 0.0)
                / np.maximum(p_den, np.finfo(float).tiny)
            )
            y_label = "Visibility (%)"
        else:
            ratio_values = ratio_db(p_num, p_den)
            y_label = "Power Ratio (dB)"
        mask = freq <= freq_max
        label = labels[index] if labels and index < len(labels) else f"Pair {index + 1}"
        ax.plot(
            freq[mask],
            ratio_values[mask],
            label=label,
            color=colors[index % len(colors)],
            linewidth=float(style.get("line_width", 2.0)),
            alpha=0.9,
        )

    if ratio_type == "power_db":
        ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
    if style.get("show_band_shading", True):
        band_label = "_nolegend_"
        if style.get("show_band_legend", False):
            band_label = (
                style.get("band_label")
                or f"{band_low:g}–{band_high:g} MHz analysis band"
            )
        ax.axvspan(
            band_low, band_high, alpha=0.06,
            color="#8888ff" if theme == "dark" else "#4444aa",
            label=band_label,
        )
    ax.set_xlim(0, freq_max)
    _apply_axis_style(ax, style, "Frequency (MHz)", y_label)
    ax.set_title(
        style.get("title") or title,
        fontsize=float(style.get("font_size", 11)) + 1,
    )
    _draw_legend(ax, theme, style)
    _apply_theme(fig, [ax], theme, grid=style.get("grid", True))
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────
#  4.  Detector response curve
# ───────────────────────────────────────────────────────────────────────

def plot_detector_response(
    freq_mhz, resp_db,
    center_mhz=5.0, freq_max=15.0,
    theme="dark", fig_size=(10, 4),
):
    """Plot detector transfer function."""
    fig, ax = plt.subplots(figsize=fig_size)
    mask = freq_mhz <= freq_max
    col  = "#a78bfa" if theme == "dark" else "#6d28d9"
    ax.plot(freq_mhz[mask], resp_db[mask], color=col,
            linewidth=2.0, label="Detector response (dB)")
    ax.axhline(-6.0, color="#f39c12", linestyle="--",
               linewidth=1.2, alpha=0.7, label="-6 dB")
    ax.axvline(center_mhz, color="#5b8dee", linestyle=":",
               linewidth=1.2, alpha=0.7, label=f"f_c = {center_mhz} MHz")
    ax.set_xlabel("Frequency (MHz)", fontsize=11)
    ax.set_ylabel("Response (dB)", fontsize=11)
    ax.set_title("Detector Frequency Response", fontsize=12)
    ax.set_xlim(0, freq_max)
    ax.legend(**_legend_kwargs(theme))
    _apply_theme(fig, [ax], theme)
    fig.tight_layout()
    return fig


# ───────────────────────────────────────────────────────────────────────
#  5.  Combined 2-panel: time + spectrum for a single trace
# ───────────────────────────────────────────────────────────────────────

def plot_trace_summary(
    trace, spectral_result,
    t_start_ns=30.0, t_end_ns=200.0,
    freq_max=15.0, band_low=2.0, band_high=8.0,
    theme="dark", fig_size=(14, 6),
    title="",
):
    """Two-panel summary: time domain (left) + spectrum (right)."""
    fig = plt.figure(figsize=fig_size)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.3)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])

    col = "#5b8dee" if theme == "dark" else "#1f77b4"
    role = trace.get("role", "") or trace.get("params", {}).get("role", "")

    # Time domain
    t = trace["time_ns"]
    p = trace["pressure_pa"] * 1e3
    ax_t.plot(t, p, color=col, linewidth=1.3)
    ax_t.axvline(t_start_ns, color="#aaaaaa", linestyle="--",
                 linewidth=0.8, alpha=0.5)
    ax_t.axvline(t_end_ns, color="#aaaaaa", linestyle=":",
                 linewidth=0.8, alpha=0.5)
    ax_t.axvspan(t_start_ns, t_end_ns, alpha=0.05,
                 color="#ffffff" if theme == "dark" else "#000000")
    ax_t.set_xlabel("Time (ns)", fontsize=11)
    ax_t.set_ylabel("Pressure (mPa)", fontsize=11)
    ax_t.set_title("Time Domain", fontsize=11)

    # Spectrum
    if spectral_result is not None:
        freq = spectral_result["freq_mhz"]
        mask = freq <= freq_max
        p_db, _ = power_to_db(spectral_result["power"])
        ax_f.plot(freq[mask], p_db[mask], color=col,
                  linewidth=1.4, label="Raw")
        pf_db, _ = power_to_db(spectral_result["power_flt"])
        ax_f.plot(freq[mask], pf_db[mask], color="#2ecc71",
                  linewidth=1.4, linestyle="--", label="Filtered", alpha=0.8)
        ax_f.axvspan(band_low, band_high, alpha=0.06,
                     color="#8888ff" if theme == "dark" else "#4444aa")
        ax_f.set_xlabel("Frequency (MHz)", fontsize=11)
        ax_f.set_ylabel("Power (dB, self-norm.)", fontsize=11)
        ax_f.set_xlim(0, freq_max)
        ax_f.set_title("Power Spectrum", fontsize=11)
        ax_f.legend(**_legend_kwargs(theme))

    label_str = title or _trace_label(trace)
    fig.suptitle(label_str, fontsize=13)
    _apply_theme(fig, [ax_t, ax_f], theme)
    fig.tight_layout()
    return fig
