#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMSOL FFT Analysis Tool — CLI Entry Point
============================================

Usage:
    python cli.py single --csv <path> --point-name "Detector P3" [options]
    python cli.py batch  --config batch_config.json
"""

import os
import sys
import io
import argparse

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                  encoding="utf-8", errors="replace")

from fft_core import (analyze_single, analyze_batch,
                      export_metrics_csv, export_spectra_csv,
                      write_analysis_report)


def cmd_single(args):
    """Run analysis on a single CSV file."""
    print(f"\n{'#' * 70}")
    print(f"  COMSOL FFT ANALYSIS TOOL — SINGLE MODE")
    print(f"{'#' * 70}")

    metrics, spectra = analyze_single(
        csv_path=args.csv,
        point_name=args.point_name,
        output_dir=args.output_dir,
        param_column=args.param_column,
        time_column=args.time_column,
        signal_column=args.signal_column,
        param_name=args.param_name,
        baseline_end_ns=args.baseline_end,
        freq_max=args.freq_max,
        export_freq_max=args.export_freq_max,
        dpi=args.dpi,
    )

    # Export metrics and spectra
    os.makedirs(args.output_dir, exist_ok=True)
    export_metrics_csv(metrics,
                       os.path.join(args.output_dir, "fft_metrics.csv"))
    export_spectra_csv(spectra,
                       os.path.join(args.output_dir, "fft_spectra.csv"),
                       freq_max=args.export_freq_max)

    config_info = {
        "csv_path": args.csv,
        "point_name": args.point_name,
        "baseline_end_ns": args.baseline_end,
        "freq_max_mhz": args.freq_max,
        "export_freq_max_mhz": args.export_freq_max,
        "figure_dpi": args.dpi,
    }
    write_analysis_report(metrics, config_info,
                          os.path.join(args.output_dir,
                                       "analysis_report.txt"))

    # Print final metrics table
    print(f"\n\n{'=' * 70}")
    print(f"  FINAL METRICS TABLE")
    print(f"{'=' * 70}")
    print(f"{'Point':<20s} {'Param':<12s} {'f_dom (MHz)':>12s} "
          f"{'BW50 (MHz)':>12s} {'BW Range (MHz)':>24s}")
    print("-" * 70)
    for r in metrics:
        pv = r.get("param_value", "-")
        print(f"{r['point']:<20s} {str(pv):<12s} "
              f"{r['dominant_freq']:>12.3f} "
              f"{r['bandwidth']:>12.3f} "
              f"[{r['bw_f_low']:>8.3f} -- {r['bw_f_high']:>8.3f}]")

    print(f"\n{'=' * 70}")
    print(f"  ANALYSIS COMPLETE — outputs in {args.output_dir}")
    print(f"{'=' * 70}")


def cmd_batch(args):
    """Run analysis from a JSON batch configuration."""
    print(f"\n{'#' * 70}")
    print(f"  COMSOL FFT ANALYSIS TOOL — BATCH MODE")
    print(f"{'#' * 70}")

    all_metrics, all_spectra = analyze_batch(args.config)

    # Print final metrics table
    print(f"\n\n{'=' * 70}")
    print(f"  FINAL METRICS TABLE (ALL ANALYSES)")
    print(f"{'=' * 70}")
    print(f"{'Point':<20s} {'Param':<12s} {'f_dom (MHz)':>12s} "
          f"{'BW50 (MHz)':>12s} {'BW Range (MHz)':>24s}")
    print("-" * 70)
    for r in all_metrics:
        pv = r.get("param_value", "-")
        print(f"{r['point']:<20s} {str(pv):<12s} "
              f"{r['dominant_freq']:>12.3f} "
              f"{r['bandwidth']:>12.3f} "
              f"[{r['bw_f_low']:>8.3f} -- {r['bw_f_high']:>8.3f}]")

    print(f"\n{'=' * 70}")
    print(f"  BATCH ANALYSIS COMPLETE")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        prog="comsol_fft_tool",
        description="COMSOL FFT Analysis Tool — "
                    "Frequency-domain analysis of COMSOL pressure exports")

    subparsers = parser.add_subparsers(dest="command",
                                       help="Analysis mode")

    # ── single mode ──
    sp = subparsers.add_parser("single",
                               help="Analyze a single CSV file")
    sp.add_argument("--csv", required=True,
                    help="Path to the COMSOL CSV file")
    sp.add_argument("--point-name", required=True,
                    help="Display name for the measurement point")
    sp.add_argument("--output-dir", default="./outputs",
                    help="Output directory (default: ./outputs)")
    sp.add_argument("--param-column", default=None,
                    help="Parameter column name for sweep data")
    sp.add_argument("--param-name", default=None,
                    help="Display name for the parameter")
    sp.add_argument("--time-column", default=None,
                    help="Time column name (auto-detected by default)")
    sp.add_argument("--signal-column", default=None,
                    help="Signal column name (auto-detected by default)")
    sp.add_argument("--baseline-end", type=float, default=10.0,
                    help="Baseline interval end in ns (default: 10.0)")
    sp.add_argument("--freq-max", type=float, default=150.0,
                    help="Max frequency for plots in MHz (default: 150)")
    sp.add_argument("--export-freq-max", type=float, default=500.0,
                    help="Max frequency for CSV export in MHz "
                         "(default: 500)")
    sp.add_argument("--dpi", type=int, default=300,
                    help="Figure resolution (default: 300)")
    sp.set_defaults(func=cmd_single)

    # ── batch mode ──
    bp = subparsers.add_parser("batch",
                               help="Run batch analysis from JSON config")
    bp.add_argument("--config", required=True,
                    help="Path to the batch JSON configuration file")
    bp.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
