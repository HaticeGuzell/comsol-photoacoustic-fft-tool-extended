#!/usr/bin/env python3
"""Generic CSV ingestion for photoacoustic waveform collections.

The public data model is deliberately small: every waveform is a dictionary
with ``time_ns``, ``pressure_pa`` and an arbitrary ``params`` dictionary.
"""

from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_KEY_VALUE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P<value>[^;,]+)"
)
_TIME = re.compile(
    r"(?:^|[@;\s])t(?:ime)?\s*=\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?P<unit>ns|us|µs|μs|ms|s)?",
    re.IGNORECASE,
)
_PAIR_CASE = re.compile(
    r"pair_case\s*(?:=|:|,)?\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)


def cast_scalar(value):
    """Convert numeric/boolean text while preserving ordinary strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if not isinstance(value, str):
        if isinstance(value, np.generic):
            return value.item()
        return value
    text = value.strip().strip("\"'")
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if _NUMBER.match(text):
        number = float(text)
        return int(number) if number.is_integer() else number
    return text


def infer_detector_from_filename(filename):
    """Infer a useful detector label when the CSV does not provide one."""
    stem = Path(filename or "Detector").stem
    stem = re.sub(r"(?i)(?:[_\-\s]*(?:all[_\-\s]*cases|sweep|export))+$", "", stem)
    known = re.search(r"(?i)(right|left|top|bottom|front|back|center|centre)", stem)
    if known:
        return known.group(1).title()
    if re.search(r"(?i)transverse", stem):
        return "Top"
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    return cleaned or "Detector"


def extract_pair_case(column_name):
    """Extract pair_case from common COMSOL header variants."""
    match = _PAIR_CASE.search(str(column_name))
    return cast_scalar(match.group(1)) if match else None


def parse_header_params(header):
    """Parse every ``key=value`` item from a COMSOL column header."""
    params = {}
    for match in _KEY_VALUE.finditer(str(header)):
        key = match.group("key")
        if key.lower() in {"t", "time"}:
            continue
        params[key] = cast_scalar(match.group("value"))
    return params


def _time_to_ns(value, unit):
    scale = {
        "ns": 1.0,
        "us": 1e3,
        "µs": 1e3,
        "μs": 1e3,
        "ms": 1e6,
        "s": 1e9,
    }
    return float(value) * scale[unit]


def _resolve_time_unit(header_unit, requested):
    if requested and requested != "auto":
        return requested
    return (header_unit or "ns").lower()


def infer_time_unit(column_name):
    """Infer a time unit from a column label, defaulting to nanoseconds."""
    name = str(column_name).lower().replace("μ", "µ")
    if re.search(r"\(\s*s\s*\)", name) or "second" in name:
        return "s"
    if "nanosecond" in name or re.search(r"(?:^|[_\s(])ns(?:$|[_\s)])", name):
        return "ns"
    if "microsecond" in name or "µs" in name or re.search(
        r"(?:^|[_\s(])us(?:$|[_\s)])", name
    ):
        return "us"
    if "millisecond" in name or re.search(
        r"(?:^|[_\s(])ms(?:$|[_\s)])", name
    ):
        return "ms"
    return "ns"


def _read_text(source):
    if hasattr(source, "read"):
        position = source.tell() if hasattr(source, "tell") else None
        payload = source.read()
        if position is not None and hasattr(source, "seek"):
            source.seek(position)
        if isinstance(payload, bytes):
            return payload.decode("utf-8-sig", errors="replace")
        return str(payload)
    if isinstance(source, bytes):
        return source.decode("utf-8-sig", errors="replace")
    with open(source, "r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read()


def detect_input_format(source):
    """Return ``wide``, ``long`` or ``single`` from CSV contents."""
    text = _read_text(source)
    lines = [line for line in text.splitlines() if line.strip()]
    if any(len(list(_TIME.finditer(line))) > 1 for line in lines[:30]):
        return "wide"
    content = "\n".join(line for line in lines if not line.lstrip().startswith("%"))
    try:
        columns = [str(c).lower() for c in pd.read_csv(io.StringIO(content), nrows=2).columns]
    except Exception:
        return "wide" if any(_TIME.search(line) for line in lines[:30]) else "single"
    has_time = any("time" in c or c.strip() == "t" for c in columns)
    has_pressure = any(
        token in c for c in columns for token in ("pressure", "signal", " p ", "p(")
    )
    return "long" if has_time and has_pressure and len(columns) > 2 else "single"


def _csv_rows(text):
    return list(csv.reader(io.StringIO(text), skipinitialspace=True))


def read_tabular_csv(source):
    """Read ordinary or COMSOL comment-header CSV into a DataFrame.

    COMSOL Point Evaluation exports commonly place the real header on the last
    ``%`` line, followed by numeric rows. This reader preserves that header.
    """
    text = _read_text(source)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The CSV is empty.")

    def parsed(line):
        return next(csv.reader([line], skipinitialspace=True))

    def numeric_row(row):
        if not row:
            return False
        numeric = 0
        for cell in row:
            try:
                float(str(cell).strip())
                numeric += 1
            except (TypeError, ValueError):
                pass
        return numeric >= max(1, len(row) // 2)

    comment_headers = []
    data_lines = []
    direct_header = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%"):
            candidate = stripped.lstrip("%").strip()
            row = parsed(candidate)
            if len(row) >= 2 and any(
                token in str(cell).lower()
                for cell in row
                for token in ("time", "pressure", "p (")
            ):
                comment_headers.append(row)
            continue
        row = parsed(line)
        if direct_header is None and not data_lines and not numeric_row(row):
            direct_header = row
        else:
            data_lines.append(row)

    header = direct_header or (comment_headers[-1] if comment_headers else None)
    if header is None:
        width = max((len(row) for row in data_lines), default=0)
        header = [f"column_{index}" for index in range(width)]
    if not data_lines:
        raise ValueError("No data rows were found in the CSV.")

    width = len(header)
    normalized_rows = [
        (row + [""] * width)[:width] for row in data_lines
    ]
    frame = pd.DataFrame(normalized_rows, columns=[str(item).strip() for item in header])
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        nonempty = frame[column].astype(str).str.strip().ne("")
        if nonempty.any() and converted[nonempty].notna().all():
            frame[column] = converted
    return frame


def suggest_time_column(columns):
    return next(
        (
            column for column in columns
            if "time" in str(column).lower()
            or str(column).strip().lower() in {"t", "% t"}
        ),
        columns[0] if len(columns) else None,
    )


def suggest_pressure_columns(columns, time_column=None):
    suggestions = [
        column for column in columns
        if column != time_column
        and (
            "pressure" in str(column).lower()
            or re.search(r"(?:^|\W)p\s*(?:\(|$)", str(column), re.IGNORECASE)
        )
    ]
    return suggestions


def parse_manual_mapped_csv(
    source,
    filename,
    time_column,
    pressure_columns,
    time_unit="auto",
    detector_source="filename",
    detector_name=None,
    parameter_mode="header",
    parameter_column=None,
    manual_pair_cases=None,
):
    """Parse row-based COMSOL exports using explicit user column mapping."""
    frame = read_tabular_csv(source)
    if time_column not in frame.columns:
        raise ValueError(f"Time column {time_column!r} was not found.")
    pressure_columns = [
        column for column in pressure_columns if column in frame.columns
    ]
    if not pressure_columns:
        raise ValueError("Select at least one pressure column.")

    unit_aliases = {
        "auto": "auto",
        "seconds": "s",
        "second": "s",
        "s": "s",
        "nanoseconds": "ns",
        "nanosecond": "ns",
        "ns": "ns",
        "microseconds": "us",
        "microsecond": "us",
        "us": "us",
        "µs": "us",
        "milliseconds": "ms",
        "millisecond": "ms",
        "ms": "ms",
    }
    requested_unit = unit_aliases.get(str(time_unit).lower(), str(time_unit).lower())
    resolved_unit = (
        infer_time_unit(time_column) if requested_unit == "auto" else requested_unit
    )
    detector = (
        infer_detector_from_filename(filename)
        if detector_source == "filename"
        else str(detector_name or "").strip()
    )
    if not detector:
        raise ValueError("A detector name is required.")

    time_values = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    time_ns_all = np.asarray(
        [_time_to_ns(value, resolved_unit) for value in time_values],
        dtype=float,
    )
    manual_pair_cases = manual_pair_cases or {}
    mode = str(parameter_mode).lower()
    traces = []
    warnings_found = []

    if mode == "column":
        if parameter_column not in frame.columns:
            raise ValueError("Select a valid parameter column.")
        parameter_values = frame[parameter_column]
        groups = [
            (cast_scalar(value), np.asarray(parameter_values == value))
            for value in parameter_values.dropna().unique()
        ]
    else:
        groups = [(None, np.ones(len(frame), dtype=bool))]

    for pressure_column in pressure_columns:
        pressure_all = pd.to_numeric(
            frame[pressure_column], errors="coerce"
        ).to_numpy(float)
        header_params = parse_header_params(pressure_column)
        header_pair_case = (
            header_params.get("pair_case")
            if "pair_case" in header_params
            else extract_pair_case(pressure_column)
        )
        manual_pair_case = cast_scalar(manual_pair_cases.get(pressure_column))

        for selected_parameter, group_mask in groups:
            params = {
                key: value for key, value in header_params.items()
                if key.lower() not in {"time", "t"}
            }
            params["detector"] = detector
            if mode == "column":
                params["pair_case"] = selected_parameter
            elif mode == "manual":
                if manual_pair_case not in (None, ""):
                    params["pair_case"] = manual_pair_case
            elif header_pair_case is not None:
                params["pair_case"] = header_pair_case
            elif manual_pair_case not in (None, ""):
                params["pair_case"] = manual_pair_case
            else:
                warnings_found.append(
                    f"No pair_case was assigned for {pressure_column!r}."
                )

            finite = (
                group_mask
                & np.isfinite(time_ns_all)
                & np.isfinite(pressure_all)
            )
            order = np.argsort(time_ns_all[finite])
            if not np.any(finite):
                warnings_found.append(
                    f"No numeric samples were found for {pressure_column!r}."
                )
                continue
            traces.append({
                "time_ns": time_ns_all[finite][order],
                "pressure_pa": pressure_all[finite][order],
                "params": params,
                "detector": detector,
                "source_file": os.path.basename(filename),
                "source_format": "manual",
                "mapping": {
                    "time_column": str(time_column),
                    "pressure_column": str(pressure_column),
                    "time_unit": resolved_unit,
                    "parameter_mode": mode,
                },
            })

    meta = {
        "format": "manual",
        "trace_count": len(traces),
        "ignored_columns": [
            str(column) for column in frame.columns
            if column not in {time_column, *pressure_columns}
        ],
        "parameter_keys": sorted({
            key for trace in traces for key in trace["params"]
        }),
        "time_column": str(time_column),
        "pressure_columns": [str(column) for column in pressure_columns],
        "time_unit": resolved_unit,
        "detector": detector,
        "warnings": list(dict.fromkeys(warnings_found)),
    }
    return traces, meta, frame.head(10)


def traces_to_long_dataframe(traces):
    """Normalize generic traces to a reusable tidy CSV table."""
    rows = []
    for trace in traces:
        common = {
            "detector": trace.get("params", {}).get(
                "detector", trace.get("detector", "")
            ),
            "source_file": trace.get("source_file", ""),
            **{
                key: value for key, value in trace.get("params", {}).items()
                if key not in {"detector", "role"}
            },
        }
        for time_ns, pressure_pa in zip(
            trace["time_ns"], trace["pressure_pa"]
        ):
            rows.append({
                "time_ns": time_ns,
                "pressure_pa": pressure_pa,
                **common,
            })
    preferred = ["time_ns", "pressure_pa", "detector", "pair_case", "source_file"]
    frame = pd.DataFrame(rows)
    ordered = [column for column in preferred if column in frame.columns]
    ordered += [column for column in frame.columns if column not in ordered]
    return frame[ordered] if ordered else frame


def parse_wide_comsol_csv(source, filename=None, time_unit="auto"):
    """Parse COMSOL wide exports with arbitrary sweep parameters."""
    text = _read_text(source)
    rows = _csv_rows(text)
    desc_idx = None
    for index, row in enumerate(rows):
        if sum(bool(_TIME.search(cell)) for cell in row) >= 1:
            desc_idx = index
            break
    if desc_idx is None:
        raise ValueError("No COMSOL description row containing t=... was found.")

    descriptions = [cell.strip() for cell in rows[desc_idx]]
    numeric_rows = []
    for row in rows[desc_idx + 1:]:
        if not row or not any(cell.strip() for cell in row):
            continue
        converted = []
        numeric_count = 0
        for cell in row:
            try:
                converted.append(float(cell))
                numeric_count += 1
            except (TypeError, ValueError):
                converted.append(np.nan)
        if numeric_count:
            converted.extend([np.nan] * (len(descriptions) - len(converted)))
            numeric_rows.append(converted[:len(descriptions)])
    if not numeric_rows:
        raise ValueError("No numeric data rows were found after the COMSOL header.")
    values = np.asarray(numeric_rows, dtype=float)

    detector = infer_detector_from_filename(filename or getattr(source, "name", "Detector.csv"))
    groups = {}
    ignored = []
    parsed_columns = 0
    for col_idx, description in enumerate(descriptions):
        time_match = _TIME.search(description)
        if not time_match:
            ignored.append(description or f"column_{col_idx}")
            continue
        unit = _resolve_time_unit(time_match.group("unit"), time_unit)
        time_ns = _time_to_ns(time_match.group("value"), unit)
        params = parse_header_params(description)
        params.setdefault("detector", detector)
        key = tuple(sorted((name, repr(value)) for name, value in params.items()))
        column = values[:, col_idx]
        if not np.any(np.isfinite(column)):
            ignored.append(description)
            continue
        pressure = float(np.nanmean(column))
        groups.setdefault(key, {"params": params, "samples": []})["samples"].append(
            (time_ns, pressure)
        )
        parsed_columns += 1

    traces = []
    source_name = filename or getattr(source, "name", "uploaded.csv")
    for group in groups.values():
        samples = sorted(group["samples"], key=lambda item: item[0])
        traces.append({
            "time_ns": np.asarray([item[0] for item in samples], dtype=float),
            "pressure_pa": np.asarray([item[1] for item in samples], dtype=float),
            "params": dict(group["params"]),
            "source_file": os.path.basename(source_name),
            "source_format": "wide",
        })
    if not traces:
        raise ValueError("No waveform columns could be parsed from the COMSOL export.")

    meta = {
        "format": "wide",
        "trace_count": len(traces),
        "parsed_columns": parsed_columns,
        "ignored_columns": ignored,
        "parameter_keys": sorted({key for trace in traces for key in trace["params"]}),
    }
    preview = pd.DataFrame(values[:5], columns=descriptions)
    return traces, meta, preview


def _find_column(columns, candidates, contains=()):
    lowered = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        low = str(col).lower()
        if any(token in low for token in contains):
            return col
    return None


def parse_long_csv(source, filename=None, time_unit="auto",
                   time_column=None, pressure_column=None):
    """Parse tidy/long CSV; every unique parameter combination becomes a trace."""
    frame = read_tabular_csv(source)
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    time_column = time_column or _find_column(
        frame.columns,
        ("time_ns", "time", "t", "time_us", "time_ms", "time_s"),
        ("time",),
    )
    if pressure_column is None:
        pressure_suggestions = suggest_pressure_columns(
            list(frame.columns), time_column
        )
        if len(pressure_suggestions) > 1:
            raise ValueError(
                "Multiple pressure columns were detected. "
                "Please use Manual column mapping."
            )
        pressure_column = (
            pressure_suggestions[0] if pressure_suggestions
            else _find_column(
                frame.columns,
                ("pressure_pa", "pressure", "signal", "p"),
                ("pressure", "photoacoustic"),
            )
        )
    if time_column is None or pressure_column is None:
        raise ValueError(
            "Long CSV requires identifiable time and pressure columns "
            "(for example time_ns and pressure_pa)."
        )

    time_name = str(time_column).lower()
    explicit = next((unit for unit in ("ns", "us", "ms", "s")
                     if time_name.endswith(f"_{unit}")), None)
    unit = _resolve_time_unit(explicit, time_unit)
    param_columns = [
        col for col in frame.columns
        if col not in {time_column, pressure_column}
    ]
    detector = infer_detector_from_filename(filename or getattr(source, "name", "Detector.csv"))
    grouping = param_columns
    grouped = [((), frame)] if not grouping else frame.groupby(
        grouping, dropna=False, sort=False
    )
    traces = []
    source_name = filename or getattr(source, "name", "uploaded.csv")
    for group_key, group in grouped:
        if grouping:
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            params = {
                str(column): cast_scalar(value)
                for column, value in zip(grouping, key_values)
            }
        else:
            params = {}
        params.setdefault("detector", detector)
        time_values = pd.to_numeric(group[time_column], errors="coerce").to_numpy(float)
        pressure_values = pd.to_numeric(group[pressure_column], errors="coerce").to_numpy(float)
        mask = np.isfinite(time_values) & np.isfinite(pressure_values)
        order = np.argsort(time_values[mask])
        traces.append({
            "time_ns": np.asarray(
                [_time_to_ns(value, unit) for value in time_values[mask][order]],
                dtype=float,
            ),
            "pressure_pa": pressure_values[mask][order],
            "params": params,
            "source_file": os.path.basename(source_name),
            "source_format": "long",
        })
    meta = {
        "format": "long",
        "trace_count": len(traces),
        "ignored_columns": [],
        "parameter_keys": sorted({key for trace in traces for key in trace["params"]}),
        "time_column": str(time_column),
        "pressure_column": str(pressure_column),
    }
    return traces, meta, frame.head()


def parse_csv(source, filename=None, input_format="auto", time_unit="auto"):
    """Parse wide, long/tidy, or single-signal CSV into generic traces."""
    fmt = input_format.lower().strip()
    aliases = {
        "auto-detect": "auto",
        "comsol wide format": "wide",
        "long/tidy csv format": "long",
        "single signal format": "single",
    }
    fmt = aliases.get(fmt, fmt)
    if fmt == "auto":
        fmt = detect_input_format(source)
    if fmt == "wide":
        return parse_wide_comsol_csv(source, filename=filename, time_unit=time_unit)
    if fmt in {"long", "single"}:
        traces, meta, preview = parse_long_csv(
            source, filename=filename, time_unit=time_unit
        )
        meta["format"] = fmt
        for trace in traces:
            trace["source_format"] = fmt
        return traces, meta, preview
    raise ValueError(f"Unsupported input format: {input_format}")


def parameter_summary(traces):
    """Return one row per trace for UI inspection and selection."""
    rows = []
    for index, trace in enumerate(traces):
        row = {
            "trace_id": index,
            "source_file": trace.get("source_file", ""),
            "samples": len(trace["time_ns"]),
            "t_min_ns": float(np.min(trace["time_ns"])) if len(trace["time_ns"]) else np.nan,
            "t_max_ns": float(np.max(trace["time_ns"])) if len(trace["time_ns"]) else np.nan,
        }
        row.update(trace.get("params", {}))
        rows.append(row)
    return pd.DataFrame(rows)
