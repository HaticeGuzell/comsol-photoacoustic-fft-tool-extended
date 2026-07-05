#!/usr/bin/env python3
"""Reproducible ZIP export for complete workbench analyses."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _trace_rows(traces):
    rows = []
    for trace_id, trace in enumerate(traces):
        common = {
            "trace_id": trace_id,
            "role": trace.get("role", ""),
            "source_file": trace.get("source_file", ""),
            **{key: value for key, value in trace.get("params", {}).items()
               if key != "role"},
        }
        for time_ns, pressure in zip(trace["time_ns"], trace["pressure_pa"]):
            rows.append({**common, "time_ns": time_ns, "pressure_pa": pressure})
    return rows


def _spectrum_rows(traces, spectra):
    rows = []
    for trace_id, (trace, spectrum) in enumerate(zip(traces, spectra)):
        if spectrum is None:
            continue
        common = {
            "trace_id": trace_id,
            "role": trace.get("role", ""),
            "source_file": trace.get("source_file", ""),
            **{key: value for key, value in trace.get("params", {}).items()
               if key != "role"},
        }
        for index, frequency in enumerate(spectrum["freq_mhz"]):
            rows.append({
                **common,
                "frequency_mhz": frequency,
                "raw_power": spectrum["power"][index],
                "filtered_power": spectrum["power_flt"][index],
                "detector_amplitude_response": spectrum["response"][index],
            })
    return rows


def export_analysis_package(traces, spectra, metrics_rows, config, plots=None,
                            report_notes=None, extra_files=None):
    """Return bytes for a self-contained analysis ZIP."""
    plots = plots or {}
    extra_files = extra_files or {}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "metrics_summary.csv",
            pd.DataFrame(metrics_rows).to_csv(index=False),
        )
        archive.writestr(
            "all_time_traces_long.csv",
            pd.DataFrame(_trace_rows(traces)).to_csv(index=False),
        )
        archive.writestr(
            "all_spectra_long.csv",
            pd.DataFrame(_spectrum_rows(traces, spectra)).to_csv(index=False),
        )
        archive.writestr(
            "analysis_config.json",
            json.dumps(config, indent=2, ensure_ascii=False, default=_json_default),
        )
        report = [
            "# PA Spectral Analysis Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Analyzed traces: {len(traces)}",
            f"Successful spectra: {sum(item is not None for item in spectra)}",
            "",
            "The configuration required to reproduce this run is stored in "
            "`analysis_config.json`.",
        ]
        if report_notes:
            report.extend(["", str(report_notes)])
        archive.writestr("report.md", "\n".join(report))
        for name, payload in plots.items():
            archive.writestr(f"plots/{name}", payload)
        for name, payload in extra_files.items():
            archive.writestr(name, payload)
    return output.getvalue()
