#!/usr/bin/env python3
"""Guided, workflow-based interface for common PA research analyses."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re

import numpy as np
import pandas as pd

from export_core import export_analysis_package
from generic_parser import parse_csv
from metrics_core import build_metrics_table, matched_role_pairs, ratio_metrics
from signal_math import combine_traces, materialize_broadcast_roles
from spectral_core import analyze_traces


WORKFLOWS = [
    "Pair validation",
    "Fluence sensitivity",
    "Aggregation contrast",
    "Custom ratio builder",
]

FFT_PRESETS = {
    "Thesis Tau10, R180": {
        "gate_start_ns": 20.0,
        "gate_end_ns": 180.0,
        "window": "Hamming",
        "detrend": "Mean subtraction",
        "nfft_mode": "Custom",
        "custom_nfft": 16384,
        "band_low": 2.0,
        "band_high": 8.0,
        "freq_max": 20.0,
    },
    "3D Tau30": {
        "gate_start_ns": 30.0,
        "gate_end_ns": 200.0,
        "window": "Hamming",
        "detrend": "Mean subtraction",
        "nfft_mode": "Custom",
        "custom_nfft": 16384,
        "band_low": 2.0,
        "band_high": 8.0,
        "freq_max": 20.0,
    },
    "Pair validation default": {
        "gate_start_ns": 30.0,
        "gate_end_ns": 200.0,
        "window": "Hamming",
        "detrend": "Mean subtraction",
        "nfft_mode": "Custom",
        "custom_nfft": 16384,
        "band_low": 2.0,
        "band_high": 8.0,
        "freq_max": 20.0,
    },
}


def _geometry_from_filename(filename):
    lower = filename.lower()
    if "dispersedbalanced" in lower or "dispersed_balanced" in lower:
        return "DispersedBalanced"
    if "aggregated" in lower:
        return "Aggregated"
    if "dispersed" in lower:
        return "Dispersed"
    return "Unknown"


def _filename_role(filename):
    normalized = re.sub(r"[^a-z0-9]+", "_", filename.lower())
    rules = [
        ("total_recon", "TOTAL_RECON"),
        ("structure", "STRUCTURE"),
        ("a_force", "A_FORCE"),
        ("a_off", "A_OFF"),
        ("a_on", "A_ON"),
        ("d_off", "D_OFF"),
        ("force_only", "A_FORCE"),
    ]
    for token, role in rules:
        if token in normalized:
            return role
    return None


def _classify_trace(trace):
    clone = copy.deepcopy(trace)
    filename = clone.get("source_file", "")
    geometry = _geometry_from_filename(filename)
    params = clone.setdefault("params", {})
    pair_case = params.get("pair_case")
    role = _filename_role(filename)
    if role is None and pair_case is not None:
        try:
            role = {3: "A_OFF", 4: "A_ON", 5: "A_FORCE"}.get(
                int(float(pair_case))
            )
        except (TypeError, ValueError):
            role = None
    if role is None and geometry in {"Dispersed", "DispersedBalanced"}:
        role = "D_OFF"
    role = role or "Unclassified"
    clone["role"] = role
    params["role"] = role
    params["geometry"] = geometry
    return clone


def _parse_uploads(uploads):
    traces, errors = [], []
    for upload in uploads:
        try:
            parsed, _, _ = parse_csv(
                io.BytesIO(upload.getvalue()),
                filename=upload.name,
                input_format="auto",
                time_unit="auto",
            )
            traces.extend(_classify_trace(trace) for trace in parsed)
        except Exception as exc:
            errors.append((upload.name, str(exc)))
    return traces, errors


def _file_detection_table(traces, uploads):
    rows = []
    for upload in uploads:
        file_traces = [
            trace for trace in traces if trace.get("source_file") == upload.name
        ]
        roles = sorted({trace.get("role", "") for trace in file_traces})
        detectors = sorted({
            trace.get("params", {}).get("detector", "Unknown")
            for trace in file_traces
        })
        geometries = sorted({
            trace.get("params", {}).get("geometry", "Unknown")
            for trace in file_traces
        })
        pair_cases = sorted({
            trace.get("params", {}).get("pair_case")
            for trace in file_traces
            if trace.get("params", {}).get("pair_case") is not None
        }, key=str)
        rows.append({
            "File name": upload.name,
            "Detector": ", ".join(map(str, detectors)) or "Unknown",
            "Geometry": ", ".join(map(str, geometries)) or "Unknown",
            "Role": ", ".join(roles) or "Unclassified",
            "pair_case": ", ".join(map(str, pair_cases)) or "none",
            "Status": "✅ OK" if file_traces and "Unclassified" not in roles else "⚠️ Check",
        })
    return pd.DataFrame(rows)


def _required_roles(workflow):
    return {
        "Pair validation": ["A_OFF", "A_ON", "A_FORCE"],
        "Fluence sensitivity": ["A_OFF", "A_FORCE"],
        "Aggregation contrast": ["A_OFF", "A_FORCE", "D_OFF"],
        "Custom ratio builder": [],
    }[workflow]


def _verification(traces, workflow):
    detectors = sorted({
        trace.get("params", {}).get("detector")
        for trace in traces
        if trace.get("params", {}).get("detector")
    })
    required = _required_roles(workflow)
    rows, missing = [], []
    if not detectors and required:
        return pd.DataFrame(columns=["Detector", *required]), [
            "No detector labels were identified"
        ]
    for detector in detectors:
        available = {
            trace.get("role") for trace in traces
            if trace.get("params", {}).get("detector") == detector
        }
        row = {"Detector": detector}
        for role in required:
            present = role in available
            row[role] = "✅" if present else "❌"
            if not present:
                missing.append(f"{detector}: {role}")
        rows.append(row)
    return pd.DataFrame(rows), missing


def _sanity_checks(traces, uploads):
    messages = []
    for upload in uploads:
        lower = upload.name.lower()
        text = upload.getvalue().decode("utf-8-sig", errors="ignore")
        if "top" in lower and re.search(r"\(\s*-120(?:\.0+)?\s*,\s*0(?:\.0+)?\s*\)", text):
            messages.append(
                ("warning", f"{upload.name}: filename says Top, but the header "
                 "contains point (-120, 0). Check the COMSOL export point.")
            )
    for detector in {
        trace.get("params", {}).get("detector") for trace in traces
    }:
        role_map = {
            trace.get("role"): trace for trace in traces
            if trace.get("params", {}).get("detector") == detector
        }
        if "A_FORCE" in role_map and "A_OFF" in role_map:
            force_peak = float(np.max(np.abs(role_map["A_FORCE"]["pressure_pa"])))
            off_peak = float(np.max(np.abs(role_map["A_OFF"]["pressure_pa"])))
            if force_peak > off_peak:
                messages.append(
                    ("warning", f"{detector}: A_FORCE is larger than A_OFF. "
                     "Check role mapping and physical scaling.")
                )
    return messages


def _fft_settings(st, preset_name):
    base = copy.deepcopy(FFT_PRESETS.get(preset_name, FFT_PRESETS["Thesis Tau10, R180"]))
    if preset_name == "Custom":
        with st.expander("Advanced FFT settings", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                base["gate_start_ns"] = st.number_input("Gate start (ns)", value=20.0)
                base["gate_end_ns"] = st.number_input("Gate end (ns)", value=180.0)
            with c2:
                base["window"] = st.selectbox(
                    "Window", ["Rectangular", "Hann", "Hamming", "Tukey", "Blackman"],
                    index=2,
                )
                base["custom_nfft"] = int(st.number_input(
                    "NFFT", value=16384, min_value=4, step=128
                ))
            with c3:
                base["band_low"] = st.number_input("Metric band low (MHz)", value=2.0)
                base["band_high"] = st.number_input("Metric band high (MHz)", value=8.0)
                base["freq_max"] = st.number_input("Plot max (MHz)", value=20.0)
            base["detrend"] = "Mean subtraction"
            base["nfft_mode"] = "Custom"
    response = st.radio(
        "Detector response",
        ["Off — raw spectrum", "On — approximate 5 MHz response"],
        horizontal=True,
    )
    base["detector"] = (
        {"mode": "None / raw"}
        if response.startswith("Off")
        else {
            "mode": "Hysi 5 MHz, 60% Gaussian",
            "center_mhz": 5.0,
            "bw_fraction": 0.60,
        }
    )
    return base


def _prepare_analysis(traces, workflow):
    collections = {}
    for trace in traces:
        collections.setdefault(trace.get("role"), []).append(trace)
    if workflow == "Fluence sensitivity":
        collections = materialize_broadcast_roles(
            collections,
            ["detector", "Phi_local"],
            {"A_FORCE": ["Phi_local"]},
        )
    base = [trace for items in collections.values() for trace in items]
    derived = []
    diagnostics = []
    detectors = sorted({
        trace.get("params", {}).get("detector") for trace in base
        if trace.get("params", {}).get("detector")
    })
    if workflow in {"Aggregation contrast", "Pair validation"}:
        for detector in detectors:
            role_map = {}
            for trace in base:
                if trace.get("params", {}).get("detector") == detector:
                    role_map.setdefault(trace.get("role"), []).append(trace)
            if workflow == "Aggregation contrast":
                needed = ("A_OFF", "A_FORCE", "D_OFF")
                if all(len(role_map.get(role, [])) == 1 for role in needed):
                    a_off = role_map["A_OFF"][0]
                    force = role_map["A_FORCE"][0]
                    d_off = role_map["D_OFF"][0]
                    structure = combine_traces(a_off, d_off, "-", "STRUCTURE")
                    a_plus_force = combine_traces(
                        a_off, force, "+", "A_OFF_PLUS_FORCE"
                    )
                    total = combine_traces(
                        a_plus_force, d_off, "-", "TOTAL_RECON"
                    )
                    derived.extend([structure, total])
                else:
                    diagnostics.append(f"{detector}: ambiguous aggregation roles.")
            else:
                needed = ("A_OFF", "A_ON")
                if all(len(role_map.get(role, [])) == 1 for role in needed):
                    derived.append(
                        combine_traces(
                            role_map["A_ON"][0],
                            role_map["A_OFF"][0],
                            "-",
                            "DIFF",
                        )
                    )
    return base + derived, diagnostics


def _recipe_ratio_definitions(workflow, recipe):
    if workflow == "Pair validation":
        definitions = [
            ("DIFF", "A_OFF", "DIFF / A_OFF"),
            ("A_FORCE", "A_OFF", "A_FORCE / A_OFF"),
        ]
    elif workflow == "Fluence sensitivity":
        definitions = [("A_FORCE", "A_OFF", "A_FORCE / A_OFF")]
    else:
        definitions = [
            ("STRUCTURE", "D_OFF", "STRUCTURE / D_OFF"),
            ("A_FORCE", "A_OFF", "A_FORCE / A_OFF"),
            ("TOTAL_RECON", "D_OFF", "TOTAL_RECON / D_OFF"),
        ]
    if recipe == "Aggregation contrast":
        return [item for item in definitions if item[0] == "STRUCTURE"]
    if recipe == "Force visibility":
        return [item for item in definitions if item[0] in {"A_FORCE", "DIFF"}]
    if recipe == "Total reconstructed contrast":
        return [item for item in definitions if item[0] == "TOTAL_RECON"]
    return definitions


def _build_ratio_rows(traces, spectra, definitions, group_keys, band_low, band_high):
    rows, plot_pairs, labels = [], [], []
    for numerator_role, denominator_role, ratio_name in definitions:
        pairs = matched_role_pairs(
            traces, [numerator_role], denominator_role, group_keys
        )
        for numerator_index, denominator_index, key in pairs:
            if spectra[numerator_index] is None or spectra[denominator_index] is None:
                continue
            values = ratio_metrics(
                spectra[numerator_index],
                spectra[denominator_index],
                band_low,
                band_high,
                False,
            )
            params = dict(key)
            detector = params.get("detector", "Unknown")
            rows.append({
                "Detector": detector,
                "Ratio": ratio_name,
                **{name: value for name, value in key if name != "detector"},
                "Band mean ratio (dB)": values.get("mean_ratio_db"),
                "Midband ratio (dB)": values.get("midband_ratio_db"),
                "Best ratio (dB)": values.get("best_ratio_db"),
                "Best frequency (MHz)": values.get("best_ratio_frequency_mhz"),
            })
            plot_pairs.append((spectra[numerator_index], spectra[denominator_index]))
            labels.append(f"{detector}: {ratio_name}")
    return rows, plot_pairs, labels


def _interpret_ratios(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "No matched ratio metrics are available."
    messages = []
    for detector in frame["Detector"].drop_duplicates():
        subset = frame[frame["Detector"] == detector]
        structure = subset[subset["Ratio"] == "STRUCTURE / D_OFF"]
        force = subset[subset["Ratio"] == "A_FORCE / A_OFF"]
        if not structure.empty and not force.empty:
            s_value = float(structure.iloc[0]["Band mean ratio (dB)"])
            f_value = float(force.iloc[0]["Band mean ratio (dB)"])
            if f_value < s_value - 40:
                messages.append(
                    f"{detector}: structural contrast dominates. The force-only "
                    f"ratio is {s_value - f_value:.1f} dB weaker in the analysis band."
                )
            else:
                messages.append(
                    f"{detector}: structural and force contributions should both "
                    "be reviewed; their separation is below 40 dB."
                )
    return " ".join(messages) or "Review the matched band-ratio table below."


def render_guided_workbench(st, fig_to_png, workflow, fft_preset):
    """Render the six-step beginner workflow."""
    import matplotlib.pyplot as plt
    from plot_builder import plot_ratio_pairs, plot_time_overlay

    st.markdown("## Guided analysis")
    st.caption(
        "Follow the six steps below. Advanced mapping and custom ratio controls "
        "remain available in Advanced mode."
    )

    st.markdown("### Step 1 — Upload COMSOL CSV files")
    st.caption(
        "Upload detector CSV files exported from COMSOL. Both AllCases files "
        "and single-case files are supported."
    )
    with st.expander("ℹ️ Which files should I upload?"):
        st.markdown(
            """
For aggregation contrast, upload:

- Aggregated Right/Left/Top AllCases files
- Dispersed or DispersedBalanced Right/Left/Top OFF files

Examples:

`2D_4RBC_Tau10_R180_Aggregated_Right_AllCases.csv`  
`2D_4RBC_Tau10_R180_DispersedBalanced_OFF_Right.csv`
"""
        )
    uploads = st.file_uploader(
        "CSV files", type=["csv"], accept_multiple_files=True,
        key="guided_uploads",
    )
    if not uploads:
        st.info("Next: upload the COMSOL detector files for your selected workflow.")
        return

    traces, parse_errors = _parse_uploads(uploads)
    st.markdown("### Step 2 — Auto-detect signals")
    with st.expander("ℹ️ How does the app detect detector and case labels?"):
        st.markdown(
            """
Detector and geometry are inferred from filenames. For Aggregated AllCases
files, `pair_case=3`, `4`, and `5` map to A_OFF, A_ON, and A_FORCE.
Dispersed OFF files map to D_OFF. You can override parsing in Advanced mode.
"""
        )
    st.dataframe(
        _file_detection_table(traces, uploads),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Advanced: manual signal-label override"):
        st.warning("Only use this table if automatic signal labeling is incorrect.")
        if st.button("Reset to auto-detect", key="guided_reset_overrides"):
            st.session_state.pop("guided_override_table", None)
            rerun = getattr(st, "rerun", None) or st.experimental_rerun
            rerun()
        override_rows = [
            {
                "File": trace.get("source_file", ""),
                "Detector": trace.get("params", {}).get("detector", "Unknown"),
                "Geometry": trace.get("params", {}).get("geometry", "Unknown"),
                "Role": trace.get("role", "Unclassified"),
                "pair_case": trace.get("params", {}).get("pair_case"),
            }
            for trace in traces
        ]
        edited_overrides = st.data_editor(
            pd.DataFrame(override_rows),
            use_container_width=True,
            hide_index=True,
            key="guided_override_table",
            column_config={
                "File": st.column_config.TextColumn("File", disabled=True),
                "Detector": st.column_config.TextColumn("Detector"),
                "Geometry": st.column_config.SelectboxColumn(
                    "Geometry",
                    options=["Aggregated", "DispersedBalanced", "Dispersed", "Unknown"],
                ),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=[
                        "A_OFF", "A_ON", "A_FORCE", "D_OFF", "STRUCTURE",
                        "TOTAL_RECON", "Unclassified",
                    ],
                ),
                "pair_case": st.column_config.NumberColumn(
                    "pair_case", disabled=True
                ),
            },
        )
        for trace, (_, edited) in zip(traces, edited_overrides.iterrows()):
            trace["role"] = str(edited["Role"])
            trace["params"]["role"] = str(edited["Role"])
            trace["params"]["detector"] = str(edited["Detector"])
            trace["params"]["geometry"] = str(edited["Geometry"])
        st.caption("Edits are applied immediately to the verification dashboard.")
    if parse_errors:
        with st.expander("Advanced: files requiring manual mapping"):
            st.warning(
                "Auto-detection could not parse the files below. Switch the "
                "sidebar Mode to Advanced to open manual column mapping."
            )
            for filename, error in parse_errors:
                st.write(f"- **{filename}** — {error}")

    st.markdown("### Step 3 — Verify imported data")
    with st.expander("ℹ️ What do A_OFF, A_FORCE, and D_OFF mean?"):
        st.markdown(
            """
- **A_OFF**: aggregated geometry without mechanical force (`pair_case=3`)
- **A_FORCE**: force-only contribution (`pair_case=5`)
- **D_OFF**: dispersed geometry without force
"""
        )
    verification, missing = _verification(traces, workflow)
    st.dataframe(verification, use_container_width=True, hide_index=True)
    for level, message in _sanity_checks(traces, uploads):
        getattr(st, level)(message)
    if missing:
        st.error("Some required signals are missing.")
        st.write("Missing: " + ", ".join(missing))
        st.info("Next: upload the missing files or switch to Advanced mode.")
    else:
        st.success("All required signals were found. You can run the analysis.")

    st.markdown("### Step 4 — Choose analysis recipe")
    if workflow == "Aggregation contrast":
        recipe_options = [
            "Aggregation contrast",
            "Force visibility",
            "Total reconstructed contrast",
            "All standard ratios",
            "Custom ratio",
        ]
    elif workflow == "Pair validation":
        recipe_options = [
            "Pair validation",
            "Force visibility",
            "All standard ratios",
            "Custom ratio",
        ]
    else:
        recipe_options = [
            "Force visibility",
            "All standard ratios",
            "Custom ratio",
        ]
    recipe = st.radio(
        "Analysis recipe",
        recipe_options,
        index=recipe_options.index("All standard ratios"),
        horizontal=True,
    )
    with st.expander("ℹ️ What is STRUCTURE, FORCE, and TOTAL_RECON?"):
        st.markdown(
            """
This analysis uses:

`STRUCTURE = A_OFF − D_OFF`  
`FORCE = A_FORCE`  
`TOTAL_RECON = A_OFF + A_FORCE − D_OFF`

Standard ratios:

`STRUCTURE / D_OFF`  
`A_FORCE / A_OFF`  
`TOTAL_RECON / D_OFF`
"""
        )
    if recipe == "Custom ratio" or workflow == "Custom ratio builder":
        st.info(
            "Custom numerator/denominator analysis is available in Advanced mode."
        )

    st.markdown("### Step 5 — FFT settings and run")
    st.caption(f"FFT preset: {fft_preset}")
    with st.expander("ℹ️ What do gate, window, NFFT, and metric band mean?"):
        st.markdown(
            """
- **Gate** selects the time interval used for the FFT.
- **Window** reduces edge discontinuities.
- **NFFT** controls zero-padding and visual frequency interpolation.
- **Metric band** defines the frequency range summarized in tables.
"""
        )
    settings = _fft_settings(st, fft_preset)
    with st.expander("ℹ️ What is detector-weighted spectrum?"):
        st.markdown(
            """
Raw spectrum shows the simulated pressure spectrum directly.
Detector-weighted spectrum estimates how a finite-bandwidth 5 MHz transducer
would emphasize or suppress frequency components.
"""
        )
    signature = hashlib.sha256(json.dumps({
        "files": [(upload.name, upload.size) for upload in uploads],
        "workflow": workflow,
        "recipe": recipe,
        "settings": settings,
    }, sort_keys=True, default=str).encode()).hexdigest()
    can_run = (
        not missing
        and not parse_errors
        and recipe != "Custom ratio"
        and workflow != "Custom ratio builder"
        and bool(traces)
    )
    if st.button(
        "Run analysis", type="primary", use_container_width=True,
        disabled=not can_run,
    ):
        analysis_traces, diagnostics = _prepare_analysis(traces, workflow)
        spectra, errors = analyze_traces(analysis_traces, settings)
        metric_rows = build_metrics_table(
            analysis_traces,
            spectra,
            settings["band_low"],
            settings["band_high"],
        )
        group_keys = (
            ["detector", "Phi_local"]
            if workflow == "Fluence sensitivity"
            else ["detector"]
        )
        definitions = _recipe_ratio_definitions(workflow, recipe)
        ratio_rows, ratio_pairs, ratio_labels = _build_ratio_rows(
            analysis_traces,
            spectra,
            definitions,
            group_keys,
            settings["band_low"],
            settings["band_high"],
        )
        st.session_state.guided_result = {
            "signature": signature,
            "traces": analysis_traces,
            "spectra": spectra,
            "metrics": metric_rows,
            "ratios": ratio_rows,
            "ratio_pairs": ratio_pairs,
            "ratio_labels": ratio_labels,
            "settings": settings,
            "workflow": workflow,
            "recipe": recipe,
            "warnings": diagnostics + errors,
        }

    result = st.session_state.get("guided_result")
    if not result:
        if can_run:
            st.info("Next: click Run analysis.")
        return
    if result["signature"] != signature:
        st.warning("Inputs or settings changed. Run the analysis again.")

    st.markdown("### Step 6 — Results and exports")
    with st.expander("ℹ️ How should I interpret dB ratios?"):
        st.markdown(
            """
A 20 dB decrease in power ratio represents a 100× smaller power contribution.
Matched ratios compare traces with the same detector and sweep parameters.
"""
        )
    summary_tab, time_tab, ratio_tab, download_tab = st.tabs(
        ["Summary", "Time domain", "Spectral ratios", "Downloads"]
    )
    ratio_frame = pd.DataFrame(result["ratios"])
    metrics_frame = pd.DataFrame(result["metrics"])
    with summary_tab:
        if ratio_frame.empty:
            st.warning("No matched ratio metrics were produced.")
        else:
            st.dataframe(ratio_frame, use_container_width=True, hide_index=True)
            interpretation = _interpret_ratios(result["ratios"])
            st.success(interpretation)
        for warning in result["warnings"]:
            st.warning(warning)

    display_traces = [
        trace for trace in result["traces"]
        if trace.get("role") in {
            "A_OFF", "A_FORCE", "D_OFF", "DIFF", "STRUCTURE", "TOTAL_RECON"
        }
    ][:12]
    time_figure = plot_time_overlay(
        display_traces,
        settings.get("gate_start_ns"),
        settings.get("gate_end_ns"),
        y_unit="mPa",
        labels=[
            f"{trace.get('params', {}).get('detector')}: {trace.get('role')}"
            for trace in display_traces
        ],
        theme="light",
        style={
            "title": "Time-domain signal comparison",
            "legend_title": "Signal",
            "legend_position": "Outside right",
            "fig_size": [8, 4.5],
            "font_size": 11,
            "line_width": 2,
            "grid": True,
        },
    )
    ratio_figure = None
    if result["ratio_pairs"]:
        ratio_figure = plot_ratio_pairs(
            result["ratio_pairs"],
            labels=result["ratio_labels"],
            freq_max=settings["freq_max"],
            band_low=settings["band_low"],
            band_high=settings["band_high"],
            theme="light",
            style={
                "title": "Matched spectral ratios",
                "xlabel": "Frequency (MHz)",
                "ylabel": "Power ratio (dB)",
                "legend_title": "Ratio",
                "legend_position": "Outside right",
                "fig_size": [8, 4.5],
                "font_size": 11,
                "line_width": 2,
                "grid": True,
                "show_band_shading": True,
                "show_band_legend": False,
            },
        )

    with time_tab:
        st.pyplot(time_figure)
    with ratio_tab:
        if ratio_figure is not None:
            st.pyplot(ratio_figure)
        else:
            st.info("No matched spectral ratio plot is available.")

    time_png = fig_to_png(time_figure, 300)
    ratio_png = fig_to_png(ratio_figure, 300) if ratio_figure is not None else None
    plt.close(time_figure)
    if ratio_figure is not None:
        plt.close(ratio_figure)
    plots = {"Guided_time_domain.png": time_png}
    if ratio_png is not None:
        plots["Guided_spectral_ratios.png"] = ratio_png
    config = {
        "mode": "Guided",
        "workflow": workflow,
        "recipe": recipe,
        "fft": settings,
        "detected_files": _file_detection_table(traces, uploads).to_dict("records"),
    }
    package = export_analysis_package(
        result["traces"],
        result["spectra"],
        result["metrics"],
        config,
        plots=plots,
        report_notes=_interpret_ratios(result["ratios"]),
        extra_files={
            "time_domain_peak_metrics.csv": metrics_frame.to_csv(index=False),
            "spectral_ratio_metrics_2_8MHz.csv": ratio_frame.to_csv(index=False),
        },
    )
    with download_tab:
        st.download_button(
            "Download metrics only",
            ratio_frame.to_csv(index=False).encode(),
            "spectral_ratio_metrics.csv",
            "text/csv",
        )
        figures_buffer = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(figures_buffer, "w") as archive:
            for name, payload in plots.items():
                archive.writestr(name, payload)
        st.download_button(
            "Download figures only",
            figures_buffer.getvalue(),
            "guided_figures.zip",
            "application/zip",
        )
        st.download_button(
            "Download full results package",
            package,
            "pa_guided_analysis_results.zip",
            "application/zip",
            type="primary",
            use_container_width=True,
        )
