#!/usr/bin/env python3
"""Streamlit UI for the general PA spectral analysis workbench."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_core import export_analysis_package
from generic_parser import (
    extract_pair_case,
    infer_detector_from_filename,
    parameter_summary,
    parse_csv,
    parse_manual_mapped_csv,
    read_tabular_csv,
    suggest_pressure_columns,
    suggest_time_column,
    traces_to_long_dataframe,
)
from metrics_core import (
    build_metrics_table,
    difference_validation,
    matched_role_pairs,
    ratio_metrics,
)
from plot_builder import (
    format_detector,
    format_phi,
    format_role,
    plot_ratio,
    plot_ratio_pairs,
    plot_spectrum_overlay,
    plot_time_overlay,
)
from presets import PRESETS, get_preset
from signal_math import (
    assign_roles,
    derive_signals,
    flatten_collections,
    materialize_broadcast_roles,
)
from spectral_core import DETREND_MODES, WINDOWS, analyze_traces


def _label(trace, index=None):
    params = trace.get("params", {})
    parts = [trace.get("role") or params.get("role") or "Signal"]
    for key in ("detector", "Phi_local", "N_RBC", "agg_state", "force_on", "tau"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    parts.append(trace.get("source_file", ""))
    if index is not None:
        parts.append(f"#{index}")
    return " | ".join(str(part) for part in parts if str(part))


def _rows(frame):
    return [
        {key: ("" if pd.isna(value) else value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _parse_broadcast(role_rows):
    result = {}
    for row in role_rows:
        role = str(row.get("name", "")).strip()
        keys = [
            item.strip()
            for item in str(row.get("broadcast_keys", "")).split(",")
            if item.strip()
        ]
        if role and keys:
            result[role] = keys
    return result


def _signature(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _theme_value(label):
    return "dark" if label == "Dark app" else "light"


def _roles_and_parameter_keys(traces):
    roles = sorted({
        trace.get("role") for trace in traces if trace.get("role")
    })
    keys = sorted({
        key
        for trace in traces
        for key in trace.get("params", {})
        if key != "role"
    })
    return roles, keys


def _default_numerator_roles(roles):
    if "FORCE" in roles:
        return ["FORCE"]
    if "DIFF" in roles:
        return ["DIFF"]
    return roles[:1]


def _ratio_pair_label(traces, numerator_index, denominator_index, key):
    numerator_role = traces[numerator_index].get("role", "Numerator")
    denominator_role = traces[denominator_index].get("role", "Denominator")
    group_text = " | ".join(
        f"{name}={value}" for name, value in key
    )
    label = f"{numerator_role}/{denominator_role}"
    return f"{label} | {group_text}" if group_text else label


def _pair_filter_controls(st, traces, pairs, match_keys, key_prefix):
    """Render optional subset filters and return the remaining matched pairs."""
    if not pairs or not match_keys:
        return pairs
    st.caption("Filter matched pairs")
    selections = {}
    columns = st.columns(min(len(match_keys), 3))
    for position, parameter in enumerate(match_keys):
        values = sorted(
            {
                traces[numerator_index].get("params", {}).get(parameter)
                for numerator_index, _, _ in pairs
            },
            key=lambda value: str(value),
        )
        with columns[position % len(columns)]:
            selections[parameter] = st.multiselect(
                parameter,
                values,
                default=values,
                key=f"{key_prefix}_{parameter}",
            )
    return [
        pair for pair in pairs
        if all(
            traces[pair[0]].get("params", {}).get(parameter)
            in selections[parameter]
            for parameter in match_keys
        )
    ]


def _short_scientific_labels(traces, curve_indices, denominator_role=None):
    if not curve_indices:
        return []
    selected = [traces[index] for index in curve_indices]
    roles = {trace.get("role") for trace in selected}
    detectors = {
        trace.get("params", {}).get("detector") for trace in selected
        if trace.get("params", {}).get("detector") is not None
    }
    fluences = {
        trace.get("params", {}).get("Phi_local") for trace in selected
        if trace.get("params", {}).get("Phi_local") is not None
    }
    labels = []
    for trace in selected:
        params = trace.get("params", {})
        parts = []
        if len(roles) > 1:
            role_text = format_role(trace.get("role", "Signal"))
            if denominator_role:
                role_text = f"{role_text}/{format_role(denominator_role)}"
            parts.append(role_text)
        if len(detectors) > 1:
            parts.append(format_detector(params.get("detector")))
        if len(fluences) > 1:
            parts.append(format_phi(params.get("Phi_local")))
        if not parts:
            role_text = format_role(trace.get("role", "Signal"))
            if denominator_role:
                role_text = f"{role_text}/{format_role(denominator_role)}"
            parts.append(role_text)
            if params.get("detector") is not None and len(selected) == 1:
                parts.append(format_detector(params["detector"]))
            if params.get("Phi_local") is not None and len(selected) == 1:
                parts.append(format_phi(params["Phi_local"]))
        labels.append(", ".join(parts))
    return labels


def _optional_float(text):
    value = str(text).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_png_filename(value, fallback):
    filename = os.path.basename(str(value).strip()) or fallback
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    if not filename.lower().endswith(".png"):
        filename += ".png"
    return filename


def _publication_controls(
    st,
    plot_type,
    original_labels,
    short_labels,
    default_title,
    default_ylabel,
    default_filename,
):
    key_prefix = re.sub(r"[^a-z0-9]+", "_", plot_type.lower()).strip("_")
    with st.expander("Publication labels and style", expanded=False):
        label_mode = st.radio(
            "Legend label mode",
            ["Full auto label", "Short scientific label", "Custom labels"],
            index=1,
            horizontal=True,
            key=f"{key_prefix}_legend_mode",
        )
        display_labels = list(original_labels)
        if label_mode == "Short scientific label":
            display_labels = list(short_labels)
        elif label_mode == "Custom labels":
            editor = st.data_editor(
                pd.DataFrame({
                    "Original label": original_labels,
                    "Custom label": short_labels,
                }),
                hide_index=True,
                use_container_width=True,
                key=f"{key_prefix}_legend_editor",
                column_config={
                    "Original label": st.column_config.TextColumn(
                        "Original label", disabled=True
                    ),
                    "Custom label": st.column_config.TextColumn(
                        "Custom label", required=True
                    ),
                },
            )
            display_labels = [
                str(value) if str(value).strip() else original_labels[index]
                for index, value in enumerate(editor["Custom label"].tolist())
            ]

        title_col, axis_col = st.columns(2)
        with title_col:
            figure_title = st.text_input(
                "Figure title", default_title, key=f"{key_prefix}_title"
            )
            legend_title = st.text_input(
                "Legend title",
                "Case" if plot_type == "Ratio spectrum" else "Signal",
                key=f"{key_prefix}_legend_title",
            )
            caption_note = st.text_area(
                "Caption note", "", key=f"{key_prefix}_caption"
            )
        with axis_col:
            xlabel = st.text_input(
                "X-axis label",
                "Time (ns)" if plot_type == "Time-domain pressure"
                else "Frequency (MHz)",
                key=f"{key_prefix}_xlabel",
            )
            ylabel = st.text_input(
                "Y-axis label", default_ylabel, key=f"{key_prefix}_ylabel"
            )
            legend_position = st.selectbox(
                "Legend position",
                [
                    "Best", "Upper right", "Upper left", "Lower right",
                    "Lower left", "Center right", "Outside right", "None",
                ],
                key=f"{key_prefix}_legend_position",
            )

        st.caption("Axis limits — leave blank for automatic limits")
        limit_columns = st.columns(4)
        limit_values = []
        for column, label, suffix in zip(
            limit_columns,
            ["X min", "X max", "Y min", "Y max"],
            ["xmin", "xmax", "ymin", "ymax"],
        ):
            with column:
                limit_values.append(
                    _optional_float(
                        st.text_input(label, "", key=f"{key_prefix}_{suffix}")
                    )
                )

        size_1, size_2, size_3, size_4 = st.columns(4)
        with size_1:
            figure_width = st.number_input(
                "Figure width (in)", 3.0, 20.0, 8.0, 0.5,
                key=f"{key_prefix}_width",
            )
        with size_2:
            figure_height = st.number_input(
                "Figure height (in)", 2.0, 16.0, 4.5, 0.5,
                key=f"{key_prefix}_height",
            )
        with size_3:
            dpi = int(st.number_input(
                "DPI", 72, 600, 300, 25, key=f"{key_prefix}_dpi"
            ))
        with size_4:
            font_size = st.number_input(
                "Font size", 6.0, 24.0, 11.0, 1.0,
                key=f"{key_prefix}_font_size",
            )

        style_1, style_2, style_3 = st.columns(3)
        with style_1:
            line_width = st.number_input(
                "Line width", 0.5, 6.0, 2.0, 0.25,
                key=f"{key_prefix}_line_width",
            )
        with style_2:
            grid = st.checkbox("Show grid", True, key=f"{key_prefix}_grid")
        with style_3:
            show_band_shading = st.checkbox(
                "Show analysis band shading",
                plot_type != "Time-domain pressure",
                key=f"{key_prefix}_band_shading",
            )
        band_1, band_2 = st.columns(2)
        with band_1:
            show_band_legend = st.checkbox(
                "Show band in legend",
                False,
                disabled=not show_band_shading,
                key=f"{key_prefix}_band_legend",
            )
        with band_2:
            band_label = st.text_input(
                "Band label",
                "2–8 MHz analysis band",
                disabled=not show_band_shading,
                key=f"{key_prefix}_band_label",
            )

        output_filename = _safe_png_filename(
            st.text_input(
                "Output figure filename",
                default_filename,
                key=f"{key_prefix}_filename",
            ),
            default_filename,
        )

    style = {
        "title": figure_title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "legend_title": legend_title,
        "legend_position": legend_position,
        "xlim": [limit_values[0], limit_values[1]],
        "ylim": [limit_values[2], limit_values[3]],
        "fig_size": [figure_width, figure_height],
        "dpi": dpi,
        "font_size": font_size,
        "line_width": line_width,
        "grid": grid,
        "show_band_shading": show_band_shading,
        "show_band_legend": show_band_legend,
        "band_label": band_label,
        "caption_note": caption_note,
        "legend_label_mode": label_mode,
        "legend_labels": dict(zip(original_labels, display_labels)),
        "output_filename": output_filename,
    }
    return style, display_labels


def _manual_mapping_panel(st, upload, frame, default_time_unit="auto", expanded=True):
    """Render one file's mapping controls and return generic parsed traces."""
    key_prefix = "manual_" + hashlib.sha1(
        upload.name.encode("utf-8")
    ).hexdigest()[:10]
    columns = list(frame.columns)
    with st.expander(f"Manual column mapping — {upload.name}", expanded=expanded):
        st.dataframe(frame.head(10), use_container_width=True, hide_index=True)
        st.caption("Detected columns: " + " · ".join(map(str, columns)))
        suggested_time = suggest_time_column(columns)
        time_index = columns.index(suggested_time) if suggested_time in columns else 0
        map_1, map_2 = st.columns(2)
        with map_1:
            time_column = st.selectbox(
                "Time column",
                columns,
                index=time_index,
                key=f"{key_prefix}_time_column",
            )
        unit_options = [
            "Auto-detect from column",
            "seconds",
            "nanoseconds",
            "microseconds",
            "milliseconds",
        ]
        unit_lookup = {
            "auto": "Auto-detect from column",
            "s": "seconds",
            "ns": "nanoseconds",
            "us": "microseconds",
            "ms": "milliseconds",
        }
        with map_2:
            time_unit_label = st.selectbox(
                "Time unit",
                unit_options,
                index=unit_options.index(
                    unit_lookup.get(default_time_unit, "Auto-detect from column")
                ),
                key=f"{key_prefix}_time_unit",
            )

        pressure_suggestions = suggest_pressure_columns(columns, time_column)
        pressure_mode = st.radio(
            "Pressure column mode",
            ["Single pressure column", "Multiple pressure columns"],
            index=1 if len(pressure_suggestions) > 1 else 0,
            horizontal=True,
            key=f"{key_prefix}_pressure_mode",
        )
        pressure_options = [column for column in columns if column != time_column]
        if not pressure_options:
            st.warning("No candidate pressure columns are available.")
            return None
        if pressure_mode == "Single pressure column":
            default_pressure = (
                pressure_suggestions[0]
                if pressure_suggestions else pressure_options[0]
            )
            pressure_columns = [st.selectbox(
                "Pressure column",
                pressure_options,
                index=pressure_options.index(default_pressure),
                key=f"{key_prefix}_pressure_single",
            )]
        else:
            pressure_columns = st.multiselect(
                "Pressure columns",
                pressure_options,
                default=pressure_suggestions,
                key=f"{key_prefix}_pressure_multiple",
            )

        detector_1, detector_2 = st.columns(2)
        with detector_1:
            detector_source_label = st.radio(
                "Detector source",
                ["Infer from filename", "Manual detector name"],
                horizontal=True,
                key=f"{key_prefix}_detector_source",
            )
        inferred_detector = infer_detector_from_filename(upload.name)
        with detector_2:
            manual_detector = st.text_input(
                "Detector name",
                inferred_detector,
                disabled=detector_source_label == "Infer from filename",
                key=f"{key_prefix}_detector_name",
            )
            if detector_source_label == "Infer from filename":
                st.caption(f"Inferred detector: {inferred_detector}")

        parameter_label = st.radio(
            "Parameter extraction",
            [
                "Infer pair_case from column headers",
                "Use selected parameter column",
                "Assign pair_case manually",
            ],
            horizontal=True,
            key=f"{key_prefix}_parameter_mode",
        )
        parameter_mode = {
            "Infer pair_case from column headers": "header",
            "Use selected parameter column": "column",
            "Assign pair_case manually": "manual",
        }[parameter_label]
        parameter_column = None
        manual_pair_cases = {}
        if parameter_mode == "column":
            parameter_options = [
                column for column in columns
                if column != time_column and column not in pressure_columns
            ]
            if parameter_options:
                parameter_column = st.selectbox(
                    "pair_case parameter column",
                    parameter_options,
                    key=f"{key_prefix}_parameter_column",
                )
            else:
                st.warning("No unused column is available as a parameter column.")
        else:
            missing_headers = [
                column for column in pressure_columns
                if extract_pair_case(column) is None
            ]
            if parameter_mode == "header":
                found = {
                    str(column): extract_pair_case(column)
                    for column in pressure_columns
                    if extract_pair_case(column) is not None
                }
                if found:
                    st.caption(
                        "Header pair_case values: "
                        + ", ".join(f"{name} → {value}" for name, value in found.items())
                    )
                assignment_columns = missing_headers
            else:
                assignment_columns = pressure_columns
            if assignment_columns:
                st.caption("Assign pair_case for pressure columns")
                for index, column in enumerate(assignment_columns):
                    manual_pair_cases[column] = st.text_input(
                        str(column),
                        value=str(3 + index),
                        key=f"{key_prefix}_pair_case_{index}",
                    )

        unit_value = {
            "Auto-detect from column": "auto",
            "seconds": "seconds",
            "nanoseconds": "nanoseconds",
            "microseconds": "microseconds",
            "milliseconds": "milliseconds",
        }[time_unit_label]
        try:
            parsed, metadata, preview = parse_manual_mapped_csv(
                io.BytesIO(upload.getvalue()),
                filename=upload.name,
                time_column=time_column,
                pressure_columns=pressure_columns,
                time_unit=unit_value,
                detector_source=(
                    "filename"
                    if detector_source_label == "Infer from filename"
                    else "manual"
                ),
                detector_name=manual_detector,
                parameter_mode=parameter_mode,
                parameter_column=parameter_column,
                manual_pair_cases=manual_pair_cases,
            )
            for warning in metadata.get("warnings", []):
                st.warning(warning)
            st.success(f"Parsed {len(parsed)} traces from this mapping.")
            if parsed:
                st.dataframe(
                    parameter_summary(parsed),
                    use_container_width=True,
                    hide_index=True,
                )
            return parsed, metadata, preview
        except Exception as exc:
            st.warning(f"Mapping is incomplete: {exc}")
            return None


def _current_summary(st, parsed_count, collections, group_keys, fft_config):
    roles = [name for name, traces in collections.items() if traces]
    with st.sidebar:
        st.markdown("### Current analysis")
        st.caption(
            f"Loaded traces: **{parsed_count}**  \n"
            f"Roles/derived: **{', '.join(roles) or 'none'}**  \n"
            f"Group by: **{', '.join(group_keys) or 'none'}**  \n"
            f"Gate: **{fft_config.get('gate_start_ns')}–"
            f"{fft_config.get('gate_end_ns')} ns**  \n"
            f"Window: **{fft_config.get('window')}**  \n"
            f"Detector: **{fft_config.get('detector', {}).get('mode')}**"
        )


def render_workbench(st, fig_to_png):
    """Render Mode 2 and own its workbench state."""
    with st.sidebar:
        st.markdown("## Workflow")
        workflow_label = st.radio(
            "What do you want to do?",
            [
                "Pair validation: ON − OFF ≈ FORCE",
                "Fluence sensitivity",
                "Aggregation contrast: Aggregated vs Dispersed",
                "General custom FFT / ratio analysis",
            ],
            index=2,
            key="workbench_workflow",
        )
        st.markdown("## Mode")
        experience_mode = st.radio(
            "Interface complexity",
            ["Guided mode", "Advanced mode"],
            index=0,
            key="workbench_experience_mode",
        )
        st.markdown("## FFT preset")
        guided_fft_preset = st.radio(
            "Recommended settings",
            [
                "Thesis Tau10, R180",
                "3D Tau30",
                "Pair validation default",
                "Custom",
            ],
            index=0,
            key="guided_fft_preset",
        )

    workflow = {
        "Pair validation: ON − OFF ≈ FORCE": "Pair validation",
        "Fluence sensitivity": "Fluence sensitivity",
        "Aggregation contrast: Aggregated vs Dispersed": "Aggregation contrast",
        "General custom FFT / ratio analysis": "Custom ratio builder",
    }[workflow_label]
    if experience_mode == "Guided mode":
        from guided_ui import render_guided_workbench
        render_guided_workbench(
            st,
            fig_to_png,
            workflow=workflow,
            fft_preset=guided_fft_preset,
        )
        return

    st.markdown("### General PA Spectral Analysis Workbench")
    st.caption(
        "Load arbitrary COMSOL waveform collections, define signal roles, "
        "build matched differences, compare curves, and export a reproducible package."
    )
    st.info(
        "Advanced mode exposes manual mapping, custom roles, FFT controls, "
        "matched ratios, and publication figure settings."
    )

    preset_name = st.selectbox("Project preset", list(PRESETS), index=0)
    preset = get_preset(preset_name)
    load_tab, roles_tab, derived_tab, fft_tab, metrics_tab, plot_tab, export_tab = st.tabs(
        [
            "1 · Load & Inspect",
            "2 · Signal Roles",
            "3 · Derived Signals",
            "4 · FFT & Detector",
            "5 · Metrics",
            "6 · Plot Builder",
            "7 · Export",
        ]
    )

    with load_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            import_mode = st.radio(
                "CSV import mode",
                ["Auto-detect", "Manual column mapping"],
                index=0,
                key="csv_import_mode",
            )
        with c2:
            input_format = st.selectbox(
                "Input format",
                ["Auto-detect", "COMSOL wide format", "Long/tidy CSV format",
                 "Single signal format"],
                disabled=import_mode == "Manual column mapping",
            )
        with c3:
            time_unit_label = st.selectbox(
                "Time unit", ["Auto (header; otherwise ns)", "ns", "us", "ms", "s"]
            )
        uploads = st.file_uploader(
            "Upload one or more waveform CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Detector is inferred from filenames such as Right_AllCases.csv "
                 "when absent from the data.",
        )
        if not uploads:
            st.info("Upload at least one CSV file to begin.")
            st.stop()

        format_map = {
            "Auto-detect": "auto",
            "COMSOL wide format": "wide",
            "Long/tidy CSV format": "long",
            "Single signal format": "single",
        }
        time_unit = "auto" if time_unit_label.startswith("Auto") else time_unit_label
        traces, parse_reports, previews = [], [], {}
        for upload in uploads:
            try:
                tabular_frame = read_tabular_csv(io.BytesIO(upload.getvalue()))
            except Exception as exc:
                st.error(f"{upload.name}: preview could not be read: {exc}")
                continue

            parsed_result = None
            if import_mode == "Auto-detect":
                with st.expander(f"Preview — {upload.name}", expanded=False):
                    st.dataframe(
                        tabular_frame.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Detected columns: "
                        + " · ".join(map(str, tabular_frame.columns))
                    )
                try:
                    parsed_result = parse_csv(
                        io.BytesIO(upload.getvalue()),
                        filename=upload.name,
                        input_format=format_map[input_format],
                        time_unit=time_unit,
                    )
                except Exception as exc:
                    st.warning(
                        f"{upload.name}: Auto-detect could not identify a complete "
                        "time/pressure mapping. Please use the manual mapping below. "
                        f"Details: {exc}"
                    )
                    parsed_result = _manual_mapping_panel(
                        st,
                        upload,
                        tabular_frame,
                        default_time_unit=time_unit,
                        expanded=True,
                    )
            else:
                parsed_result = _manual_mapping_panel(
                    st,
                    upload,
                    tabular_frame,
                    default_time_unit=time_unit,
                    expanded=True,
                )

            if parsed_result is None:
                continue
            try:
                parsed, metadata, preview = parsed_result
                traces.extend(parsed)
                parse_reports.append({
                    "file": upload.name,
                    "format": metadata["format"],
                    "traces": metadata["trace_count"],
                    "parameters": ", ".join(metadata.get("parameter_keys", [])),
                    "ignored_columns": len(metadata.get("ignored_columns", [])),
                })
                previews[upload.name] = (preview, metadata)
            except Exception as exc:
                st.error(f"{upload.name}: {exc}")
        if not traces:
            st.info(
                "No traces are parsed yet. Complete the manual mapping fields above."
            )
            st.stop()
        st.success(f"Loaded {len(traces)} traces from {len(parse_reports)} files.")
        st.dataframe(pd.DataFrame(parse_reports), use_container_width=True, hide_index=True)
        detectors = sorted({
            trace.get("params", {}).get("detector")
            for trace in traces
            if trace.get("params", {}).get("detector") is not None
        }, key=str)
        pair_cases = sorted({
            trace.get("params", {}).get("pair_case")
            for trace in traces
            if trace.get("params", {}).get("pair_case") is not None
        }, key=str)
        st.caption(
            f"Total traces parsed: {len(traces)} · "
            f"Detectors: {', '.join(map(str, detectors)) or 'none'} · "
            f"pair_case values: {', '.join(map(str, pair_cases)) or 'none'}"
        )
        st.markdown("#### Trace catalogue")
        st.dataframe(parameter_summary(traces), use_container_width=True, hide_index=True)
        normalized_frame = traces_to_long_dataframe(traces)
        st.download_button(
            "Export normalized long CSV",
            normalized_frame.to_csv(index=False).encode("utf-8"),
            "normalized_waveforms_long.csv",
            "text/csv",
        )
        with st.expander("Raw previews and ignored columns"):
            for filename, (preview, metadata) in previews.items():
                st.markdown(f"**{filename}**")
                ignored = metadata.get("ignored_columns", [])
                if ignored:
                    st.caption("Ignored: " + ", ".join(map(str, ignored[:20])))
                st.dataframe(preview, use_container_width=True, hide_index=True)

    parameter_keys = sorted({key for trace in traces for key in trace.get("params", {})})

    with roles_tab:
        st.markdown("#### Signal Role Builder")
        st.caption(
            "Each filter returns a trace collection. Supported: ==, !=, <, <=, >, "
            ">=, in [...], and AND. OR, parentheses, regex, and Python eval are disabled."
        )
        role_frame = st.data_editor(
            pd.DataFrame(preset["roles"]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"role_editor_{preset_name}",
            column_config={
                "name": st.column_config.TextColumn("Role name", required=True),
                "filter": st.column_config.TextColumn(
                    "Safe filter", help='Example: detector == "Right" AND pair_case in [3, 4]'
                ),
                "broadcast_keys": st.column_config.TextColumn(
                    "Broadcast across keys",
                    help="Comma-separated sweep keys missing from this role, e.g. Phi_local.",
                ),
            },
        )
        role_rows = _rows(role_frame)
        default_groups = [
            key for key in preset.get("group_keys", []) if key in parameter_keys
        ]
        group_keys = st.multiselect(
            "Match/group derived signals by",
            parameter_keys,
            default=default_groups or (["detector"] if "detector" in parameter_keys else []),
            help="ON and OFF are paired only when these parameter values match.",
        )
        try:
            collections = assign_roles(traces, role_rows)
            broadcast = _parse_broadcast(role_rows)
            collections = materialize_broadcast_roles(collections, group_keys, broadcast)
            st.dataframe(
                pd.DataFrame([
                    {
                        "role": role,
                        "matched traces": len(items),
                        "broadcast keys": ", ".join(broadcast.get(role, [])),
                    }
                    for role, items in collections.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Role definition error: {exc}")
            st.stop()

    with derived_tab:
        st.markdown("#### Derived Signal Builder")
        st.caption(
            "Formulas are evaluated per matched group. Slight time-grid differences "
            "are interpolated over their common interval."
        )
        derived_frame = st.data_editor(
            pd.DataFrame(preset.get("derived", []), columns=["name", "formula"]),
            num_rows="dynamic",
            use_container_width=True,
            key=f"derived_editor_{preset_name}",
            column_config={
                "name": st.column_config.TextColumn("Derived role", required=True),
                "formula": st.column_config.TextColumn(
                    "Formula", help="Examples: ON - OFF; Aggregated_ON - Aggregated_OFF"
                ),
            },
        )
        derived_rows = _rows(derived_frame)
        try:
            collections, diagnostics = derive_signals(collections, derived_rows, group_keys)
            generated = [
                {
                    "derived role": row.get("name", ""),
                    "generated traces": len(
                        collections.get(str(row.get("name", "")).strip(), [])
                    ),
                }
                for row in derived_rows if str(row.get("name", "")).strip()
            ]
            if generated:
                st.dataframe(pd.DataFrame(generated), use_container_width=True, hide_index=True)
            for diagnostic in diagnostics[:30]:
                st.warning(diagnostic)
        except Exception as exc:
            st.error(f"Derived signal error: {exc}")
            st.stop()

    analysis_traces = flatten_collections(collections)

    with fft_tab:
        st.markdown("#### FFT settings")
        defaults = preset.get("fft", {})
        a, b, c = st.columns(3)
        with a:
            gate_mode = st.radio("Time gate", ["Manual", "Full signal"], horizontal=True)
            gate_start = st.number_input(
                "Gate start (ns)", value=float(defaults.get("gate_start_ns", 30.0)),
                disabled=gate_mode == "Full signal",
            )
            gate_end = st.number_input(
                "Gate end (ns)", value=float(defaults.get("gate_end_ns", 200.0)),
                disabled=gate_mode == "Full signal",
            )
            window = st.selectbox(
                "Window", WINDOWS, index=WINDOWS.index(defaults.get("window", "Hamming"))
            )
            tukey_alpha = st.slider(
                "Tukey alpha", 0.0, 1.0, 0.25, 0.05, disabled=window != "Tukey"
            )
        with b:
            detrend = st.selectbox(
                "Detrend", DETREND_MODES,
                index=DETREND_MODES.index(defaults.get("detrend", "Mean subtraction")),
            )
            baseline_end = st.number_input(
                "Baseline interval end (ns)", value=10.0,
                disabled=detrend != "Baseline interval",
            )
            nfft_options = ["Signal length", "Next power of 2", "Custom"]
            nfft_mode = st.selectbox(
                "NFFT", nfft_options,
                index=nfft_options.index(defaults.get("nfft_mode", "Signal length")),
            )
            custom_nfft = int(st.number_input(
                "Custom NFFT",
                value=int(defaults.get("custom_nfft", 2048)),
                min_value=4,
                step=128,
                disabled=nfft_mode != "Custom",
            ))
            if nfft_mode != "Signal length":
                st.info(
                    "Zero-padding improves visual interpolation, not true frequency resolution."
                )
        with c:
            detector_defaults = preset.get("detector", {})
            detector_options = [
                "None / raw",
                "Hysi 5 MHz, 60% Gaussian",
                "Custom Gaussian",
                "Ideal bandpass",
            ]
            detector_mode = st.selectbox(
                "Detector mode", detector_options,
                index=detector_options.index(
                    detector_defaults.get("mode", "None / raw")
                ),
            )
            center_default = (
                5.0 if detector_mode == "Hysi 5 MHz, 60% Gaussian"
                else float(detector_defaults.get("center_mhz", 5.0))
            )
            center_mhz = st.number_input(
                "Detector centre (MHz)", value=center_default, min_value=0.01,
                disabled="Gaussian" not in detector_mode,
            )
            bw_default = (
                0.60 if detector_mode == "Hysi 5 MHz, 60% Gaussian"
                else float(detector_defaults.get("bw_fraction", 0.60))
            )
            bw_fraction = st.slider(
                "Fractional bandwidth (-6 dB)", 0.05, 2.0, bw_default, 0.05,
                disabled="Gaussian" not in detector_mode,
            )
            det_low = st.number_input(
                "Bandpass low (MHz)", value=2.0, min_value=0.0,
                disabled=detector_mode != "Ideal bandpass",
            )
            det_high = st.number_input(
                "Bandpass high (MHz)", value=8.0, min_value=0.01,
                disabled=detector_mode != "Ideal bandpass",
            )

        d, e, f = st.columns(3)
        with d:
            band_low = st.number_input("Metric band low (MHz)", value=2.0, min_value=0.0)
        with e:
            band_high = st.number_input("Metric band high (MHz)", value=8.0, min_value=0.01)
        with f:
            freq_max = st.number_input("Plot frequency max (MHz)", value=15.0, min_value=0.01)

        fft_config = {
            "gate_start_ns": None if gate_mode == "Full signal" else gate_start,
            "gate_end_ns": None if gate_mode == "Full signal" else gate_end,
            "window": window,
            "tukey_alpha": tukey_alpha,
            "detrend": detrend,
            "baseline_end_ns": baseline_end,
            "nfft_mode": nfft_mode,
            "custom_nfft": custom_nfft,
            "detector": {
                "mode": detector_mode,
                "center_mhz": center_mhz,
                "bw_fraction": bw_fraction,
                "low_mhz": det_low,
                "high_mhz": det_high,
            },
        }
        config = {
            "csv_import_mode": import_mode,
            "input_format": input_format,
            "time_unit": time_unit_label,
            "manual_mappings": [
                {
                    "source_file": trace.get("source_file", ""),
                    **trace.get("mapping", {}),
                }
                for trace in traces if trace.get("mapping")
            ],
            "preset": preset_name,
            "roles": role_rows,
            "derived": derived_rows,
            "group_keys": group_keys,
            "broadcast": broadcast,
            "fft": fft_config,
            "metric_band_mhz": [band_low, band_high],
            "plot_frequency_max_mhz": freq_max,
        }
        run_signature = _signature({
            **config, "files": [(item.name, item.size) for item in uploads]
        })
        if st.button(
            "Run spectral analysis",
            type="primary",
            use_container_width=True,
            disabled=not analysis_traces,
        ):
            with st.spinner(f"Analyzing {len(analysis_traces)} traces…"):
                spectra, errors = analyze_traces(analysis_traces, fft_config)
                metric_rows = build_metrics_table(
                    analysis_traces, spectra, band_low, band_high
                )
                st.session_state.workbench_result = {
                    "signature": run_signature,
                    "traces": analysis_traces,
                    "spectra": spectra,
                    "metrics": metric_rows,
                    "errors": errors,
                    "config": config,
                }
            st.success(
                f"Analysis complete: {sum(item is not None for item in spectra)} spectra."
            )
        if not analysis_traces:
            st.warning("No traces match the current role definitions.")

    _current_summary(st, len(traces), collections, group_keys, fft_config)
    result = st.session_state.get("workbench_result")
    result_current = bool(result and result.get("signature") == run_signature)
    if result and not result_current:
        st.warning("Settings changed after the last run. Run spectral analysis again.")

    with metrics_tab:
        if not result:
            st.info("Run the spectral analysis to populate metrics.")
        else:
            metrics_frame = pd.DataFrame(result["metrics"])
            st.markdown("#### Time, raw-spectrum, and detector-filtered metrics")
            st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
            st.download_button(
                "Download metrics CSV",
                metrics_frame.to_csv(index=False).encode("utf-8"),
                "metrics_summary.csv",
                "text/csv",
            )
            if result["errors"]:
                with st.expander("Analysis warnings"):
                    for error in result["errors"]:
                        st.warning(error)
            valid_indices = [
                index for index, spectrum in enumerate(result["spectra"])
                if spectrum is not None
            ]
            labels = {
                _label(result["traces"][index], index): index for index in valid_indices
            }
            validation_pairs = matched_role_pairs(
                result["traces"],
                ["DIFF"],
                "FORCE",
                result["config"].get("group_keys", []),
            )
            if validation_pairs:
                st.markdown("#### DIFF–FORCE validation")
                validation_rows = []
                for diff_index, force_index, key in validation_pairs:
                    row = {
                        name: value for name, value in key
                    }
                    row.update(
                        difference_validation(
                            result["traces"][diff_index],
                            result["traces"][force_index],
                        )
                    )
                    validation_rows.append(row)
                st.dataframe(
                    pd.DataFrame(validation_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            if len(labels) >= 2:
                st.markdown("#### Matched ratio metrics")
                st.caption(
                    "Each numerator is divided by the denominator with identical "
                    "selected match-key values. Metrics use unnormalized power."
                )
                label_list = list(labels)
                metric_roles, metric_parameter_keys = _roles_and_parameter_keys(
                    result["traces"]
                )
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    metric_numerator_roles = st.multiselect(
                        "Numerator role(s)",
                        metric_roles,
                        default=_default_numerator_roles(metric_roles),
                        key="metric_matched_numerator_roles",
                    )
                with mc2:
                    metric_denominator_role = st.selectbox(
                        "Denominator role",
                        metric_roles,
                        index=(
                            metric_roles.index("OFF")
                            if "OFF" in metric_roles else 0
                        ),
                        key="metric_matched_denominator_role",
                    )
                with mc3:
                    metric_match_keys = st.multiselect(
                        "Match keys",
                        metric_parameter_keys,
                        default=[
                            key for key in result["config"].get("group_keys", [])
                            if key in metric_parameter_keys
                        ],
                        key="metric_matched_keys",
                    )
                with mc4:
                    metric_ratio_filtered = st.checkbox(
                        "Use filtered power",
                        value=False,
                        key="metric_matched_filtered",
                    )
                metric_pairs = matched_role_pairs(
                    result["traces"],
                    metric_numerator_roles,
                    metric_denominator_role,
                    metric_match_keys,
                )
                metric_pairs = _pair_filter_controls(
                    st,
                    result["traces"],
                    metric_pairs,
                    metric_match_keys,
                    "metric_pair_filter",
                )
                metric_ratio_rows = []
                for numerator_index, denominator_index, key in metric_pairs:
                    numerator_spectrum = result["spectra"][numerator_index]
                    denominator_spectrum = result["spectra"][denominator_index]
                    if numerator_spectrum is None or denominator_spectrum is None:
                        continue
                    ratio_result = ratio_metrics(
                        numerator_spectrum,
                        denominator_spectrum,
                        band_low,
                        band_high,
                        metric_ratio_filtered,
                    )
                    metric_ratio_rows.append({
                        "numerator_role": result["traces"][numerator_index].get("role"),
                        "denominator_role": result["traces"][denominator_index].get("role"),
                        **dict(key),
                        "ratio_type": "power_db",
                        "use_filtered": metric_ratio_filtered,
                        "mean_ratio_db_in_band": ratio_result.get("mean_ratio_db"),
                        "midband_ratio_db": ratio_result.get("midband_ratio_db"),
                        "best_frequency_mhz": ratio_result.get(
                            "best_ratio_frequency_mhz"
                        ),
                        "best_ratio_db": ratio_result.get("best_ratio_db"),
                    })
                if metric_ratio_rows:
                    st.dataframe(
                        pd.DataFrame(metric_ratio_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning(
                        "No unique matched ratio pairs were found for these roles "
                        "and match keys."
                    )

                with st.expander("Single-pair ratio metrics"):
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        ratio_num = st.selectbox(
                            "Numerator trace",
                            label_list,
                            index=min(1, len(label_list) - 1),
                            key="metric_ratio_num",
                        )
                    with rc2:
                        ratio_den = st.selectbox(
                            "Denominator trace",
                            label_list,
                            index=0,
                            key="metric_ratio_den",
                        )
                    with rc3:
                        ratio_filtered = st.checkbox(
                            "Filtered ratio",
                            value=False,
                            key="metric_ratio_filtered",
                        )
                    rm = ratio_metrics(
                        result["spectra"][labels[ratio_num]],
                        result["spectra"][labels[ratio_den]],
                        band_low,
                        band_high,
                        ratio_filtered,
                    )
                    validation = difference_validation(
                        result["traces"][labels[ratio_num]],
                        result["traces"][labels[ratio_den]],
                    )
                    st.dataframe(
                        pd.DataFrame([{**rm, **validation}]),
                        use_container_width=True,
                        hide_index=True,
                    )

    with plot_tab:
        if not result:
            st.info("Run the spectral analysis to open the plot builder.")
        else:
            valid_indices = [
                index for index, spectrum in enumerate(result["spectra"])
                if spectrum is not None
            ]
            labels = {
                _label(result["traces"][index], index): index for index in valid_indices
            }
            label_list = list(labels)
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                plot_type = st.selectbox(
                    "Plot domain",
                    ["Time-domain pressure", "Raw power spectrum",
                     "Detector-filtered spectrum", "Ratio spectrum"],
                )
            with pc2:
                layout_label = st.selectbox(
                    "Layout", ["Overlay in one axis", "Separate subplots"],
                    disabled=plot_type == "Ratio spectrum",
                )
            with pc3:
                theme_label = st.selectbox(
                    "Theme", ["Dark app", "Publication light", "Minimal thesis"]
                )
            theme = _theme_value(theme_label)
            layout = "overlay" if layout_label.startswith("Overlay") else "subplots"
            figure = None
            plot_style = None

            if plot_type != "Ratio spectrum":
                selected = st.multiselect(
                    "Available curves",
                    label_list,
                    default=label_list[:min(4, len(label_list))],
                )
                chosen_indices = [labels[item] for item in selected]
                if plot_type == "Time-domain pressure":
                    y_unit = st.radio(
                        "Pressure unit", ["Pa", "mPa"], horizontal=True
                    )
                    original_labels = selected
                    short_labels = _short_scientific_labels(
                        result["traces"], chosen_indices
                    )
                    plot_style, display_labels = _publication_controls(
                        st,
                        plot_type,
                        original_labels,
                        short_labels,
                        "Time-domain force validation",
                        f"Pressure ({y_unit})",
                        "Fig_TimeDomain_Validation.png",
                    )
                    figure = plot_time_overlay(
                        [result["traces"][index] for index in chosen_indices],
                        fft_config.get("gate_start_ns"),
                        fft_config.get("gate_end_ns"),
                        y_unit=y_unit,
                        layout=layout,
                        theme=theme,
                        labels=display_labels,
                        style=plot_style,
                    )
                else:
                    normalization = st.selectbox(
                        "dB reference",
                        [
                            "Each curve self-normalized",
                            "Global selected-curve max",
                            "Selected reference trace max",
                            "No normalization / absolute FFT power",
                        ],
                    )
                    db_ref = "self"
                    if normalization.startswith("Global"):
                        db_ref = "global_max"
                    elif normalization.startswith("No normalization"):
                        db_ref = "absolute"
                    elif normalization.startswith("Selected reference") and label_list:
                        reference_label = st.selectbox("Reference trace", label_list)
                        reference_index = labels[reference_label]
                        power_key = (
                            "power_flt"
                            if plot_type == "Detector-filtered spectrum" else "power"
                        )
                        db_ref = float(
                            np.max(result["spectra"][reference_index][power_key])
                        )
                    original_labels = selected
                    short_labels = _short_scientific_labels(
                        result["traces"], chosen_indices
                    )
                    is_filtered_plot = plot_type == "Detector-filtered spectrum"
                    plot_style, display_labels = _publication_controls(
                        st,
                        plot_type,
                        original_labels,
                        short_labels,
                        "Detector-filtered spectrum"
                        if is_filtered_plot else "Raw power spectrum",
                        "Filtered power (dB, rel.)"
                        if is_filtered_plot else "Power (dB, rel.)",
                        "Fig_FilteredSpectrum.png"
                        if is_filtered_plot else "Fig_RawSpectrum.png",
                    )
                    figure = plot_spectrum_overlay(
                        [result["spectra"][index] for index in chosen_indices],
                        labels=display_labels,
                        use_filtered=is_filtered_plot,
                        freq_max=freq_max,
                        band_low=band_low,
                        band_high=band_high,
                        db_ref_mode=db_ref,
                        title=plot_type,
                        layout=layout,
                        theme=theme,
                        style=plot_style,
                    )
            elif label_list:
                ratio_mode = st.radio(
                    "Ratio mode",
                    [
                        "Matched denominator by role",
                        "Single selected denominator",
                    ],
                    index=0,
                    horizontal=True,
                )
                ratio_settings_1, ratio_settings_2 = st.columns(2)
                with ratio_settings_1:
                    ratio_type_label = st.selectbox(
                        "Ratio type",
                        ["Power ratio dB", "Amplitude ratio", "Percent visibility"],
                    )
                with ratio_settings_2:
                    filtered = st.checkbox(
                        "Use detector-filtered power",
                        value=False,
                        key="plot_ratio_filtered",
                    )
                ratio_type = {
                    "Power ratio dB": "power_db",
                    "Amplitude ratio": "amplitude",
                    "Percent visibility": "percent",
                }[ratio_type_label]
                if ratio_mode == "Matched denominator by role":
                    plot_roles, plot_parameter_keys = _roles_and_parameter_keys(
                        result["traces"]
                    )
                    rp1, rp2, rp3 = st.columns(3)
                    with rp1:
                        numerator_roles = st.multiselect(
                            "Numerator role(s)",
                            plot_roles,
                            default=_default_numerator_roles(plot_roles),
                            key="plot_matched_numerator_roles",
                        )
                    with rp2:
                        denominator_role = st.selectbox(
                            "Denominator/reference role",
                            plot_roles,
                            index=(
                                plot_roles.index("OFF")
                                if "OFF" in plot_roles else 0
                            ),
                            key="plot_matched_denominator_role",
                        )
                    with rp3:
                        match_keys = st.multiselect(
                            "Match keys",
                            plot_parameter_keys,
                            default=[
                                key for key in result["config"].get("group_keys", [])
                                if key in plot_parameter_keys
                            ],
                            key="plot_matched_keys",
                        )
                    ratio_pairs = matched_role_pairs(
                        result["traces"],
                        numerator_roles,
                        denominator_role,
                        match_keys,
                    )
                    ratio_pairs = _pair_filter_controls(
                        st,
                        result["traces"],
                        ratio_pairs,
                        match_keys,
                        "plot_pair_filter",
                    )
                    valid_pairs = [
                        pair for pair in ratio_pairs
                        if result["spectra"][pair[0]] is not None
                        and result["spectra"][pair[1]] is not None
                    ]
                    st.caption(f"Matched pairs: {len(valid_pairs)}")
                    if valid_pairs:
                        original_labels = [
                            _ratio_pair_label(
                                result["traces"],
                                numerator_index,
                                denominator_index,
                                key,
                            )
                            for numerator_index, denominator_index, key in valid_pairs
                        ]
                        short_labels = _short_scientific_labels(
                            result["traces"],
                            [pair[0] for pair in valid_pairs],
                            denominator_role,
                        )
                        ratio_ylabel = {
                            "power_db": "Power ratio (dB)",
                            "amplitude": "Amplitude ratio",
                            "percent": "Visibility (%)",
                        }[ratio_type]
                        numerator_text = "_".join(numerator_roles) or "Numerator"
                        plot_style, display_labels = _publication_controls(
                            st,
                            plot_type,
                            original_labels,
                            short_labels,
                            (
                                f"Matched {', '.join(numerator_roles)}/"
                                f"{denominator_role} spectral ratio"
                            ),
                            ratio_ylabel,
                            (
                                f"Fig_Ratio_{numerator_text}_"
                                f"{denominator_role}.png"
                            ),
                        )
                        figure = plot_ratio_pairs(
                            [
                                (
                                    result["spectra"][numerator_index],
                                    result["spectra"][denominator_index],
                                )
                                for numerator_index, denominator_index, _ in valid_pairs
                            ],
                            labels=display_labels,
                            use_filtered=filtered,
                            freq_max=freq_max,
                            band_low=band_low,
                            band_high=band_high,
                            ratio_type=ratio_type,
                            title=(
                                f"{', '.join(numerator_roles)} / "
                                f"{denominator_role} — matched ratios"
                            ),
                            theme=theme,
                            style=plot_style,
                        )
                    else:
                        st.warning(
                            "No unique matched pairs were found. Check the roles, "
                            "match keys, and filters."
                        )
                else:
                    st.warning(
                        "Single denominator mode divides all selected numerator "
                        "traces by the same reference trace. For sweep comparisons, "
                        "use matched denominator mode."
                    )
                    denominator = st.selectbox(
                        "Denominator/reference trace",
                        label_list,
                        key="plot_single_denominator",
                    )
                    numerators = st.multiselect(
                        "Numerator curves",
                        [item for item in label_list if item != denominator],
                        default=[
                            item for item in label_list
                            if item != denominator
                        ][:4],
                        key="plot_single_numerators",
                    )
                    numerator_indices = [labels[item] for item in numerators]
                    denominator_index = labels[denominator]
                    denominator_role = result["traces"][denominator_index].get(
                        "role", "Reference"
                    )
                    original_labels = [
                        f"{result['traces'][index].get('role')}/"
                        f"{denominator_role} | {item}"
                        for index, item in zip(numerator_indices, numerators)
                    ]
                    short_labels = _short_scientific_labels(
                        result["traces"],
                        numerator_indices,
                        denominator_role,
                    )
                    ratio_ylabel = {
                        "power_db": "Power ratio (dB)",
                        "amplitude": "Amplitude ratio",
                        "percent": "Visibility (%)",
                    }[ratio_type]
                    plot_style, display_labels = _publication_controls(
                        st,
                        plot_type,
                        original_labels,
                        short_labels,
                        "Spectral ratio to selected reference",
                        ratio_ylabel,
                        "Fig_Ratio_SelectedReference.png",
                    )
                    figure = plot_ratio(
                        [result["spectra"][index] for index in numerator_indices],
                        result["spectra"][denominator_index],
                        labels=display_labels,
                        use_filtered=filtered,
                        freq_max=freq_max,
                        band_low=band_low,
                        band_high=band_high,
                        ratio_type=ratio_type,
                        theme=theme,
                        style=plot_style,
                    )

            if figure is not None:
                st.pyplot(figure)
                plot_style = plot_style or {}
                plot_style["theme"] = theme_label
                plot_style["plot_type"] = plot_type
                caption_note = str(plot_style.get("caption_note", "")).strip()
                if caption_note:
                    st.caption(caption_note)
                dpi = int(plot_style.get("dpi", 300))
                png = fig_to_png(figure, dpi)
                output_filename = plot_style.get(
                    "output_filename", "plot_overlay_selected.png"
                )
                st.session_state.workbench_plot = {
                    "png": png,
                    "name": output_filename,
                    "figure_customization": plot_style,
                }
                st.download_button(
                    "Download current plot PNG",
                    png,
                    output_filename,
                    "image/png",
                )
                plt.close(figure)

    with export_tab:
        if not result:
            st.info("Run the spectral analysis before exporting.")
        else:
            st.markdown("#### Full reproducibility package")
            st.caption(
                "Includes metrics, long-form spectra, long-form time traces, "
                "analysis_config.json, report.md, and the latest selected plot."
            )
            plot_item = st.session_state.get("workbench_plot")
            plots = {plot_item["name"]: plot_item["png"]} if plot_item else {}
            figure_customization = (
                plot_item.get("figure_customization", {}) if plot_item else {}
            )
            export_roles = sorted({
                trace.get("role", "") for trace in result["traces"]
                if trace.get("role")
            })
            default_denominator_index = (
                export_roles.index("OFF") if "OFF" in export_roles else 0
            )
            ratio_denominator_role = st.selectbox(
                "Export ratio denominator role",
                export_roles,
                index=default_denominator_index,
            )
            preferred_numerators = [
                role for role in ("DIFF", "FORCE")
                if role in export_roles and role != ratio_denominator_role
            ]
            ratio_numerator_roles = st.multiselect(
                "Export ratio numerator roles",
                [
                    role for role in export_roles
                    if role != ratio_denominator_role
                ],
                default=preferred_numerators,
            )
            _, export_parameter_keys = _roles_and_parameter_keys(result["traces"])
            export_match_keys = st.multiselect(
                "Export ratio match keys",
                export_parameter_keys,
                default=[
                    key for key in result["config"].get("group_keys", [])
                    if key in export_parameter_keys
                ],
            )
            export_indices = [
                index for index, spectrum in enumerate(result["spectra"])
                if spectrum is not None
            ][:12]
            export_labels = [
                _label(result["traces"][index], index) for index in export_indices
            ]
            valid_ratio_pairs = []
            if export_indices:
                time_figure = plot_time_overlay(
                    [result["traces"][index] for index in export_indices],
                    result["config"]["fft"].get("gate_start_ns"),
                    result["config"]["fft"].get("gate_end_ns"),
                    theme="light",
                )
                plots["plot_time_domain.png"] = fig_to_png(time_figure, 300)
                plt.close(time_figure)
                for filtered, filename in (
                    (False, "plot_raw_spectrum.png"),
                    (True, "plot_filtered_spectrum.png"),
                ):
                    spectrum_figure = plot_spectrum_overlay(
                        [result["spectra"][index] for index in export_indices],
                        labels=export_labels,
                        use_filtered=filtered,
                        freq_max=result["config"]["plot_frequency_max_mhz"],
                        band_low=result["config"]["metric_band_mhz"][0],
                        band_high=result["config"]["metric_band_mhz"][1],
                        db_ref_mode="global_max",
                        theme="light",
                    )
                    plots[filename] = fig_to_png(spectrum_figure, 300)
                    plt.close(spectrum_figure)
                ratio_pairs = matched_role_pairs(
                    result["traces"],
                    ratio_numerator_roles,
                    ratio_denominator_role,
                    export_match_keys,
                )
                valid_ratio_pairs = [
                    (numerator, denominator, key)
                    for numerator, denominator, key in ratio_pairs
                    if result["spectra"][numerator] is not None
                    and result["spectra"][denominator] is not None
                ]
                if valid_ratio_pairs:
                    ratio_figure = plot_ratio_pairs(
                        [
                            (
                                result["spectra"][numerator],
                                result["spectra"][denominator],
                            )
                            for numerator, denominator, _ in valid_ratio_pairs
                        ],
                        labels=[
                            _ratio_pair_label(
                                result["traces"],
                                numerator,
                                denominator,
                                key,
                            )
                            for numerator, denominator, key in valid_ratio_pairs
                        ],
                        freq_max=result["config"]["plot_frequency_max_mhz"],
                        band_low=result["config"]["metric_band_mhz"][0],
                        band_high=result["config"]["metric_band_mhz"][1],
                        title=(
                            f"{', '.join(ratio_numerator_roles)} / "
                            f"{ratio_denominator_role}"
                        ),
                        theme="light",
                    )
                    plots["plot_ratio_spectrum.png"] = fig_to_png(ratio_figure, 300)
                    plt.close(ratio_figure)
            if plot_item:
                # Preserve the exact customized figure if its user-selected name
                # collides with one of the automatically generated package plots.
                plots[plot_item["name"]] = plot_item["png"]
            package = export_analysis_package(
                result["traces"], result["spectra"], result["metrics"],
                {
                    **result["config"],
                    "export_ratio": {
                        "numerator_roles": ratio_numerator_roles,
                        "denominator_role": ratio_denominator_role,
                        "match_keys": export_match_keys,
                        "matched_pair_count": len(valid_ratio_pairs)
                        if export_indices else 0,
                    },
                    "figure_customization": figure_customization,
                },
                plots=plots,
                report_notes=(
                    "Ratio spectra were computed using matched denominator "
                    "traces. Each numerator trace was divided by the denominator "
                    "trace with the same group-key values."
                    + (
                        "\n\nSelected figure caption: "
                        + str(figure_customization.get("caption_note"))
                        if figure_customization.get("caption_note") else ""
                    )
                ),
            )
            st.download_button(
                "Download full analysis package (ZIP)",
                package,
                "pa_spectral_analysis_package.zip",
                "application/zip",
                type="primary",
                use_container_width=True,
            )
