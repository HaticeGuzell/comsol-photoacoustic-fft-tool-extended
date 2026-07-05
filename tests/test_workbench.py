import io
import json
import unittest
import zipfile

import numpy as np

from export_core import export_analysis_package
from generic_parser import (
    extract_pair_case,
    parse_csv,
    parse_manual_mapped_csv,
    read_tabular_csv,
    traces_to_long_dataframe,
)
from metrics_core import (
    build_metrics_table,
    difference_validation,
    matched_role_pairs,
    ratio_metrics,
)
from signal_math import (
    assign_roles,
    compile_filter,
    derive_signals,
    materialize_broadcast_roles,
)
from spectral_core import compute_spectrum
from presets import get_preset
from guided_ui import (
    _classify_trace,
    _interpret_ratios,
    _prepare_analysis,
    _recipe_ratio_definitions,
    _verification,
)


def trace(role_case, phi, pressure, detector="Right"):
    time = np.arange(len(pressure), dtype=float)
    return {
        "time_ns": time,
        "pressure_pa": np.asarray(pressure, dtype=float),
        "params": {
            "pair_case": role_case,
            "Phi_local": phi,
            "detector": detector,
        },
        "source_file": f"{detector}_AllCases.csv",
        "source_format": "wide",
    }


class ParserTests(unittest.TestCase):
    def test_generic_wide_parser_and_filename_detector(self):
        csv_text = "\n".join([
            "X,p @ t=0; pair_case=3; Phi_local=20; tau=30,"
            "p @ t=1; pair_case=3; Phi_local=20; tau=30,"
            "p @ t=0; pair_case=4; Phi_local=20; tau=30,"
            "p @ t=1; pair_case=4; Phi_local=20; tau=30",
            "0,1,2,4,6",
        ])
        traces, meta, _ = parse_csv(
            io.BytesIO(csv_text.encode()), filename="Right_AllCases.csv",
            input_format="auto"
        )
        self.assertEqual(meta["trace_count"], 2)
        self.assertEqual(traces[0]["params"]["detector"], "Right")
        self.assertEqual(traces[0]["params"]["tau"], 30)
        self.assertTrue(np.array_equal(traces[0]["time_ns"], [0, 1]))

    def test_long_parser_groups_parameter_combinations(self):
        csv_text = "\n".join([
            "time_ns,pressure_pa,N_RBC,agg_state",
            "0,1,2,dispersed",
            "1,2,2,dispersed",
            "0,3,4,aggregated",
            "1,4,4,aggregated",
        ])
        traces, meta, _ = parse_csv(
            io.BytesIO(csv_text.encode()), filename="Top.csv", input_format="long"
        )
        self.assertEqual(meta["trace_count"], 2)
        self.assertEqual({item["params"]["N_RBC"] for item in traces}, {2, 4})
        self.assertEqual({item["params"]["detector"] for item in traces}, {"Top"})

    def test_manual_mapping_reads_comsol_comment_header_and_multiple_pressures(self):
        csv_text = "\n".join([
            "% Model,2D Bridge synthetic fixture",
            '% Time (s),"p (Pa) @ Point 1, pair_case=3",'
            '"p (Pa) @ Point 1, pair_case: 4",'
            '"p (Pa) @ Point 1, pair_case 5"',
            "0,1,2,1",
            "1e-9,2,4,2",
            "2e-9,3,6,3",
        ])
        frame = read_tabular_csv(io.BytesIO(csv_text.encode()))
        pressure_columns = list(frame.columns[1:])
        with self.assertRaisesRegex(ValueError, "Multiple pressure columns"):
            parse_csv(
                io.BytesIO(csv_text.encode()),
                filename="2D_Bridge_Right_AllCases.csv",
                input_format="auto",
            )
        traces, meta, _ = parse_manual_mapped_csv(
            io.BytesIO(csv_text.encode()),
            filename="2D_Bridge_Right_AllCases.csv",
            time_column=frame.columns[0],
            pressure_columns=pressure_columns,
            time_unit="auto",
            detector_source="filename",
            parameter_mode="header",
        )
        self.assertEqual(meta["trace_count"], 3)
        self.assertEqual(meta["time_unit"], "s")
        self.assertEqual(
            {item["params"]["pair_case"] for item in traces},
            {3, 4, 5},
        )
        self.assertEqual(
            {item["params"]["detector"] for item in traces},
            {"Right"},
        )
        self.assertTrue(np.allclose(traces[0]["time_ns"], [0, 1, 2]))

        normalized = traces_to_long_dataframe(traces)
        self.assertEqual(
            list(normalized.columns[:5]),
            ["time_ns", "pressure_pa", "detector", "pair_case", "source_file"],
        )
        self.assertEqual(len(normalized), 9)

    def test_pair_case_header_variants_and_transverse_detector(self):
        self.assertEqual(extract_pair_case("p, pair_case=3"), 3)
        self.assertEqual(extract_pair_case("p, pair_case = 4"), 4)
        self.assertEqual(extract_pair_case("p, pair_case, 5"), 5)
        self.assertEqual(extract_pair_case("p, pair_case: 6"), 6)

        csv_text = "\n".join([
            "% Time (ns),p (Pa)",
            "0,1",
            "1,2",
        ])
        frame = read_tabular_csv(io.BytesIO(csv_text.encode()))
        traces, _, _ = parse_manual_mapped_csv(
            io.BytesIO(csv_text.encode()),
            filename="Bridge_Transverse.csv",
            time_column=frame.columns[0],
            pressure_columns=[frame.columns[1]],
            detector_source="filename",
            parameter_mode="manual",
            manual_pair_cases={frame.columns[1]: 3},
        )
        self.assertEqual(traces[0]["params"]["detector"], "Top")


class SignalMathTests(unittest.TestCase):
    def setUp(self):
        self.traces = [
            trace(3, 20, [0, 1, 2, 3]),
            trace(4, 20, [1, 2, 3, 4]),
            trace(3, 2, [0, 2, 4, 6]),
            trace(4, 2, [1, 3, 5, 7]),
            trace(5, 20, [0.1, 0.2, 0.3, 0.4]),
        ]

    def test_safe_filter_numeric_tolerance_and_in(self):
        predicate = compile_filter("Phi_local == 2.0000000001 AND pair_case in [3, 4]")
        self.assertTrue(predicate(self.traces[2]["params"]))
        with self.assertRaises(ValueError):
            compile_filter("pair_case == 3 OR pair_case == 4")

    def test_grouped_derived_and_force_broadcast(self):
        roles = assign_roles(self.traces, [
            {"name": "OFF", "filter": "pair_case == 3"},
            {"name": "ON", "filter": "pair_case == 4"},
            {"name": "FORCE", "filter": "pair_case == 5"},
        ])
        roles = materialize_broadcast_roles(
            roles, ["detector", "Phi_local"], {"FORCE": ["Phi_local"]}
        )
        self.assertEqual(len(roles["FORCE"]), 2)
        roles, diagnostics = derive_signals(
            roles, [{"name": "DIFF", "formula": "ON - OFF"}],
            ["detector", "Phi_local"]
        )
        self.assertFalse(diagnostics)
        self.assertEqual(len(roles["DIFF"]), 2)
        for item in roles["DIFF"]:
            self.assertTrue(np.allclose(item["pressure_pa"], 1.0))

    def test_fluence_visibility_regression_and_diff_force_validation(self):
        force = np.array([0.0, 0.001272, 0.0, -0.001272])
        fluences = [20.0, 2.0, 0.2, 0.02]
        source = []
        for phi in fluences:
            off = np.array([0.0, phi, 0.0, -phi])
            source.extend([
                trace(3, phi, off),
                trace(4, phi, off + force),
            ])
        source.append(trace(5, 20.0, force))
        roles = assign_roles(source, [
            {"name": "OFF", "filter": "pair_case == 3"},
            {"name": "ON", "filter": "pair_case == 4"},
            {"name": "FORCE", "filter": "pair_case == 5"},
        ])
        roles = materialize_broadcast_roles(
            roles, ["detector", "Phi_local"], {"FORCE": ["Phi_local"]}
        )
        roles, diagnostics = derive_signals(
            roles, [{"name": "DIFF", "formula": "ON - OFF"}],
            ["detector", "Phi_local"]
        )
        self.assertFalse(diagnostics)
        expected = [0.00636, 0.0636, 0.636, 6.36]
        by_phi = {item["params"]["Phi_local"]: item for item in roles["FORCE"]}
        diff_by_phi = {item["params"]["Phi_local"]: item for item in roles["DIFF"]}
        off_by_phi = {item["params"]["Phi_local"]: item for item in roles["OFF"]}
        for phi, visibility in zip(fluences, expected):
            observed = (
                100.0
                * np.max(np.abs(by_phi[phi]["pressure_pa"]))
                / np.max(np.abs(off_by_phi[phi]["pressure_pa"]))
            )
            self.assertAlmostEqual(observed, visibility, places=10)
            self.assertLess(
                np.max(
                    np.abs(
                        diff_by_phi[phi]["pressure_pa"]
                        - by_phi[phi]["pressure_pa"]
                    )
                ),
                1e-9,
            )


class SpectralTests(unittest.TestCase):
    def test_fluence_preset_uses_requested_zero_padding(self):
        preset = get_preset("Current 3D Fluence Sweep")
        self.assertEqual(preset["fft"]["nfft_mode"], "Custom")
        self.assertEqual(preset["fft"]["custom_nfft"], 16384)

    def test_fft_nfft_detector_metrics_and_ratio(self):
        time_ns = np.arange(0.0, 1000.0, 1.0)
        base = np.sin(2 * np.pi * 5e6 * time_ns * 1e-9)
        tr_a = {
            "time_ns": time_ns,
            "pressure_pa": base,
            "params": {"detector": "Right"},
            "role": "A",
        }
        tr_b = {**tr_a, "pressure_pa": 2.0 * base, "role": "B"}
        config = {
            "window": "Hann",
            "detrend": "Mean subtraction",
            "nfft_mode": "Next power of 2",
            "gate_start_ns": None,
            "gate_end_ns": None,
            "detector": {"mode": "Hysi 5 MHz, 60% Gaussian",
                         "center_mhz": 5.0, "bw_fraction": 0.6},
        }
        spec_a = compute_spectrum(tr_a, config)
        spec_b = compute_spectrum(tr_b, config)
        peak = spec_a["freq_mhz"][np.argmax(spec_a["power"][1:]) + 1]
        self.assertAlmostEqual(peak, 5.0, delta=0.7)
        self.assertEqual(spec_a["nfft"], 1024)
        self.assertTrue(spec_a["zero_padded"])
        ratio = ratio_metrics(spec_b, spec_a, 4.0, 6.0)
        self.assertAlmostEqual(ratio["mean_ratio_db"], 10 * np.log10(4), places=5)
        self.assertIn("midband_ratio_db", ratio)
        validation = difference_validation(tr_b, tr_b)
        self.assertLess(validation["validation_max_error_pa"], 1e-9)

        metrics = build_metrics_table([tr_a], [spec_a], 4.0, 6.0)
        figure_customization = {
            "title": "Fluence dependence",
            "ylabel": "FORCE/OFF power ratio (dB)",
            "legend_labels": {"full label": "20 mJ/cm²"},
            "ylim": [-100.0, -20.0],
            "output_filename": "Fig_Fluence.png",
        }
        package = export_analysis_package(
            [tr_a],
            [spec_a],
            metrics,
            {
                "window": "Hann",
                "figure_customization": figure_customization,
            },
            plots={"Fig_Fluence.png": b"synthetic-png"},
            report_notes="Synthetic publication figure.",
            extra_files={
                "spectral_ratio_metrics_2_8MHz.csv": "ratio,value\nsynthetic,-20"
            },
        )
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {
                    "metrics_summary.csv",
                    "all_time_traces_long.csv",
                    "all_spectra_long.csv",
                    "analysis_config.json",
                    "report.md",
                    "plots/Fig_Fluence.png",
                    "spectral_ratio_metrics_2_8MHz.csv",
                },
            )
            config_exported = json.loads(
                archive.read("analysis_config.json").decode("utf-8")
            )
            self.assertEqual(
                config_exported["figure_customization"],
                figure_customization,
            )
            self.assertEqual(
                archive.read("plots/Fig_Fluence.png"),
                b"synthetic-png",
            )
            self.assertIn(
                "Synthetic publication figure.",
                archive.read("report.md").decode("utf-8"),
            )


class MatchedRatioTests(unittest.TestCase):
    @staticmethod
    def _role_trace(role, detector, phi):
        return {
            "time_ns": np.array([0.0, 1.0, 2.0, 3.0]),
            "pressure_pa": np.ones(4),
            "params": {
                "role": role,
                "detector": detector,
                "Phi_local": phi,
            },
            "role": role,
        }

    def test_fluence_overlay_uses_same_detector_and_phi_denominator(self):
        traces = []
        for phi in (20.0, 2.0, 0.2, 0.02):
            traces.extend([
                self._role_trace("OFF", "Right", phi),
                self._role_trace("FORCE", "Right", phi),
            ])
        pairs = matched_role_pairs(
            traces, ["FORCE"], "OFF", ["detector", "Phi_local"]
        )
        self.assertEqual(len(pairs), 4)
        for numerator, denominator, _ in pairs:
            self.assertEqual(
                traces[numerator]["params"]["detector"],
                traces[denominator]["params"]["detector"],
            )
            self.assertEqual(
                traces[numerator]["params"]["Phi_local"],
                traces[denominator]["params"]["Phi_local"],
            )

    def test_detector_comparison_returns_three_matched_pairs(self):
        traces = []
        for detector in ("Right", "Left", "Top"):
            traces.extend([
                self._role_trace("OFF", detector, 20.0),
                self._role_trace("FORCE", detector, 20.0),
            ])
        pairs = matched_role_pairs(
            traces, ["FORCE"], "OFF", ["detector", "Phi_local"]
        )
        self.assertEqual(len(pairs), 3)

    def test_fluence_power_ratio_increases_twenty_db_per_decade(self):
        frequency = np.array([0.0, 2.0, 5.0, 8.0])
        observed = []
        for phi in (20.0, 2.0, 0.2, 0.02):
            numerator = {
                "freq_mhz": frequency,
                "power": np.ones(4),
                "power_flt": np.ones(4),
            }
            off_power = 1e9 * (phi / 20.0) ** 2
            denominator = {
                "freq_mhz": frequency,
                "power": np.full(4, off_power),
                "power_flt": np.full(4, off_power),
            }
            observed.append(
                ratio_metrics(numerator, denominator, 2.0, 8.0)[
                    "mean_ratio_db"
                ]
            )
        self.assertTrue(np.allclose(observed, [-90.0, -70.0, -50.0, -30.0]))

    def test_diff_off_and_force_off_ratios_are_identical(self):
        traces = [
            self._role_trace("OFF", "Right", 20.0),
            self._role_trace("DIFF", "Right", 20.0),
            self._role_trace("FORCE", "Right", 20.0),
        ]
        pairs = matched_role_pairs(
            traces, ["DIFF", "FORCE"], "OFF", ["detector", "Phi_local"]
        )
        self.assertEqual(len(pairs), 2)
        frequency = np.array([0.0, 2.0, 5.0, 8.0])
        spectra = [
            {
                "freq_mhz": frequency,
                "power": np.full(4, 100.0 if trace["role"] == "OFF" else 1.0),
                "power_flt": np.full(
                    4, 100.0 if trace["role"] == "OFF" else 1.0
                ),
            }
            for trace in traces
        ]
        ratios = [
            ratio_metrics(spectra[numerator], spectra[denominator], 2.0, 8.0)[
                "mean_ratio_db"
            ]
            for numerator, denominator, _ in pairs
        ]
        self.assertAlmostEqual(ratios[0], ratios[1], places=12)


class GuidedWorkflowTests(unittest.TestCase):
    @staticmethod
    def _trace(filename, detector, pressure, pair_case=None):
        params = {"detector": detector}
        if pair_case is not None:
            params["pair_case"] = pair_case
        return {
            "time_ns": np.arange(len(pressure), dtype=float),
            "pressure_pa": np.asarray(pressure, dtype=float),
            "params": params,
            "source_file": filename,
            "source_format": "synthetic",
        }

    def test_filename_and_pair_case_detection(self):
        aggregated = _classify_trace(
            self._trace(
                "Aggregated_Right_AllCases.csv", "Right", [0, 1, 0], 3
            )
        )
        dispersed = _classify_trace(
            self._trace(
                "DispersedBalanced_OFF_Right.csv", "Right", [0, 0.5, 0]
            )
        )
        self.assertEqual(aggregated["role"], "A_OFF")
        self.assertEqual(aggregated["params"]["geometry"], "Aggregated")
        self.assertEqual(dispersed["role"], "D_OFF")
        self.assertEqual(
            dispersed["params"]["geometry"], "DispersedBalanced"
        )

    def test_aggregation_recipe_builds_expected_derived_signals(self):
        traces = [
            _classify_trace(self._trace(
                "Aggregated_Right_AllCases.csv", "Right", [0, 4, 0], 3
            )),
            _classify_trace(self._trace(
                "Aggregated_Right_AllCases.csv", "Right", [0, 6, 0], 4
            )),
            _classify_trace(self._trace(
                "Aggregated_Right_AllCases.csv", "Right", [0, 1, 0], 5
            )),
            _classify_trace(self._trace(
                "DispersedBalanced_OFF_Right.csv", "Right", [0, 2, 0]
            )),
        ]
        verification, missing = _verification(
            traces, "Aggregation contrast"
        )
        self.assertFalse(missing)
        self.assertEqual(verification.iloc[0]["A_OFF"], "✅")
        prepared, diagnostics = _prepare_analysis(
            traces, "Aggregation contrast"
        )
        self.assertFalse(diagnostics)
        by_role = {trace["role"]: trace for trace in prepared}
        self.assertTrue(np.allclose(
            by_role["STRUCTURE"]["pressure_pa"], [0, 2, 0]
        ))
        self.assertTrue(np.allclose(
            by_role["TOTAL_RECON"]["pressure_pa"], [0, 3, 0]
        ))
        definitions = _recipe_ratio_definitions(
            "Aggregation contrast", "All standard ratios"
        )
        self.assertEqual(
            [definition[2] for definition in definitions],
            [
                "STRUCTURE / D_OFF",
                "A_FORCE / A_OFF",
                "TOTAL_RECON / D_OFF",
            ],
        )

    def test_guided_interpretation_reports_structural_dominance(self):
        interpretation = _interpret_ratios([
            {
                "Detector": "Right",
                "Ratio": "STRUCTURE / D_OFF",
                "Band mean ratio (dB)": -1.0,
            },
            {
                "Detector": "Right",
                "Ratio": "A_FORCE / A_OFF",
                "Band mean ratio (dB)": -100.0,
            },
        ])
        self.assertIn("structural contrast dominates", interpretation)


if __name__ == "__main__":
    unittest.main()
