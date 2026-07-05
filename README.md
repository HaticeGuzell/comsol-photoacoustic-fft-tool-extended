# PA Spectral Analysis Workbench

[![MIT License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24%2B-FF4B4B.svg)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243.svg)](https://numpy.org)

A Python/Streamlit tool for FFT-based frequency-domain analysis of COMSOL-exported photoacoustic pressure signals. Supports both single-signal and full parametric sweep (wide all-cases) COMSOL exports.

---

## Analysis Modes

### Mode 1 — Single Signal FFT *(original)*
Classic single-file analysis: upload a COMSOL CSV, select columns, run Hann-windowed FFT, extract dominant frequency and BW50.

### Mode 2 — General PA Spectral Analysis Workbench

A parameter-agnostic research workflow for COMSOL wide exports, tidy/long CSVs,
and single signals:

1. Parse every `key=value` item in wide headers, or group tidy rows by arbitrary
   parameter columns.
2. Infer detector names from filenames such as `Right_AllCases.csv` when needed.
3. Define role collections with safe filters (`==`, `!=`, `<`, `<=`, `>`, `>=`,
   `in [...]`, and `AND`).
4. Match derived signals by user-selected keys such as `detector + Phi_local`.
5. Broadcast sweep-independent roles, for example one FORCE trace across fluence.
6. Configure gate, detrending, FFT window, NFFT, and detector response.
7. Extract time metrics plus raw and filtered Hysi-style spectral parameters.
8. Overlay arbitrary curves, create subplots, and calculate ratio spectra from
   unnormalized power.
9. Export a reproducible ZIP containing metrics, traces, spectra, configuration,
   report, and the latest selected plot.

Mode 2 opens in **Guided mode** by default. **Advanced mode** retains manual
mapping, custom roles, derived formulas, detector models, matched ratios, and
publication figure controls.

### Guided workflow

The sidebar first asks for a workflow:

- Pair validation: `ON − OFF ≈ FORCE`
- Fluence sensitivity
- Aggregation contrast: Aggregated vs Dispersed
- General custom FFT / ratio analysis

Guided mode then presents six steps:

1. Upload COMSOL files
2. Auto-detect detector, geometry, role, and `pair_case`
3. Verify required signals and review sanity checks
4. Choose a standard analysis recipe
5. Select a simplified FFT preset and run
6. Review Summary, Time domain, Spectral ratios, and Downloads

For aggregation contrast, the standard formulas are:

```text
STRUCTURE = A_OFF - D_OFF
FORCE = A_FORCE
TOTAL_RECON = A_OFF + A_FORCE - D_OFF
```

The corresponding standard ratios are:

```text
STRUCTURE / D_OFF
A_FORCE / A_OFF
TOTAL_RECON / D_OFF
```

Guided results use explicit labels such as `Right: STRUCTURE / D_OFF`, include
automatic interpretation of structural versus force-only contributions, and
block analysis until required signals are present. All help and information
expanders in the application are written in English.

---

## Repository Contents

```text
app.py                     # Streamlit web interface (Mode 1 + Mode 2)
fft_core.py                # Mode 1 core: parsing, FFT, plotting, export
sweep_core.py              # Backward-compatible legacy sweep engine
generic_parser.py          # Generic wide/long/single CSV parser
signal_math.py             # Safe role filters, matching, broadcast, derived signals
spectral_core.py           # Configurable FFT and detector-response processing
metrics_core.py            # Time, spectral, and ratio metrics
plot_builder.py            # Overlay, subplot, ratio, and publication plots
export_core.py             # Reproducible full-analysis ZIP export
presets.py                 # Fluence, multi-RBC, detector, and custom presets
workbench_ui.py            # Tabbed Mode 2 workbench
guided_ui.py               # Beginner-friendly six-step workflow
cli.py                     # Command-line interface (Mode 1)
batch_config_example.json  # Example batch config (Mode 1)
requirements.txt           # Python dependencies
.streamlit/config.toml     # Dark theme + upload size config
examples/                  # Synthetic example input data
LICENSE                    # MIT License
```

---

## Installation

```bash
git clone https://github.com/HaticeGuzell/comsol-photoacoustic-fft-tool-extended.git
cd comsol-photoacoustic-fft-tool-extended
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Use the **sidebar** to switch between Mode 1 and Mode 2.

---

## Mode 2 — Supported CSV Formats

Choose **Auto-detect**, **COMSOL wide**, **Long/tidy**, or **Single signal** in
the Load & Inspect tab.

### Manual column mapping

Load & Inspect provides two CSV import modes:

- **Auto-detect** for known wide, tidy, and single-signal formats
- **Manual column mapping** for COMSOL Point Evaluation and unfamiliar exports

If automatic parsing cannot identify a complete time/pressure mapping, the app
keeps the file preview visible and opens the manual mapping panel instead of
stopping the workflow. Each uploaded file can independently configure:

- Time column and source unit (`s`, `ms`, `µs`, or `ns`), converted internally
  to nanoseconds
- One or multiple pressure columns, with every pressure column becoming a
  separate trace
- Detector inferred from the filename or entered manually
- `pair_case` extracted from pressure headers, read from a parameter column, or
  assigned manually per pressure column

Header variants such as `pair_case=3`, `pair_case = 3`, `pair_case, 3`, and
`pair_case: 3` are supported. Filenames containing `Right`, `Left`, `Top`, or
`Transverse` are mapped to detector labels.

After mapping, the trace catalogue reports detector and `pair_case` values.
**Export normalized long CSV** creates a reusable table with:

```text
time_ns,pressure_pa,detector,pair_case,source_file
```

### COMSOL wide

COMSOL wide all-cases export format (one column per time-step × parameter combination):

```text
X,Y,Z,
p @ t=0; pair_case=3; Phi_local=20,
p @ t=0.5; pair_case=3; Phi_local=20,
p @ t=0; pair_case=4; Phi_local=20,
p @ t=0; pair_case=5; Phi_local=20,
...
0.0,0.0,0.0,1.23e-5,2.34e-5,...
```

All parameters are parsed; `pair_case` and `Phi_local` are not special-cased.

### Long/tidy

```text
time_ns,pressure_pa,N_RBC,agg_state,force_on,detector
0.0,...,8,dispersed,0,Right
0.5,...,8,dispersed,0,Right
```

Every unique combination of columns other than time and pressure becomes a
trace.

### Role and derived examples

```text
OFF             pair_case == 3
ON              pair_case == 4
FORCE           pair_case == 5        broadcast across Phi_local
Aggregated_ON   agg_state == "aggregated" AND force_on == 1

DIFF            ON - OFF
Delta_force     Aggregated_ON - Aggregated_OFF
```

Filters are parsed without Python `eval`.

### Matched ratio analysis

The default Ratio Spectrum mode is **Matched denominator by role**. It matches
each numerator trace to a unique denominator trace with the same selected
group-key values before calculating a ratio from unnormalized power.

For a fluence sweep:

```text
Numerator role(s): FORCE
Denominator role:  OFF
Match keys:        detector, Phi_local
```

This produces correctly paired curves such as:

```text
FORCE/OFF | detector=Right | Phi_local=20
FORCE/OFF | detector=Right | Phi_local=2
FORCE/OFF | detector=Right | Phi_local=0.2
FORCE/OFF | detector=Right | Phi_local=0.02
```

Filters for each match key can restrict the plot to one detector, selected
fluence values, an RBC count, an aggregation state, or another parsed
parameter. The same matched-pair logic is used by:

- Ratio Spectrum plots
- Matched ratio metrics, including band mean, midband value, best ratio, and
  best frequency
- `plots/plot_ratio_spectrum.png` in the reproducibility ZIP

The optional **Single selected denominator** mode remains available for special
analyses, but it divides every selected numerator by one reference trace and
should not be used for ordinary sweep comparisons.

### Publication-ready figure customization

Every Plot Builder domain includes a **Publication labels and style** panel.
The exported PNG uses the same settings shown in the app:

- Editable figure title, x/y-axis labels, legend title, and caption note
- Full automatic, short scientific, or manually edited legend labels
- Scientific fluence formatting such as `20 mJ/cm²`
- Editable x/y limits and legend position, including outside-right or hidden
- Figure width, height, DPI, font size, line width, grid, and analysis-band
  shading controls
- Optional analysis-band legend entry and custom band label
- User-defined PNG filename

Short scientific labels adapt to the selected comparison. A fluence overlay
uses labels such as `20 mJ/cm²`, a detector comparison uses `Right`, `Left`,
and `Top`, and a DIFF/FORCE comparison uses `ON − OFF` and `FORCE-only`.

The selected figure is included in the analysis ZIP under its chosen filename.
Its title, labels, limits, dimensions, DPI, theme, and legend mapping are saved
under `figure_customization` in `analysis_config.json`.

---

## Mode 1 — Classic CSV Format

```text
% Model, example
% tau_p (ns),Time (ns),Synthetic pressure (Pa)
5,0.0,1.23e-10
5,0.2,2.34e-10
10,0.0,9.87e-11
```

---

## FFT Methodology

**Mode 1 (Hann):**
- Baseline subtraction → Hann window → rfft → normalized amplitude → dominant freq → BW50

**Mode 2 (configurable workbench):**
- Optional gate and detrending
- Rectangular, Hann, Hamming, Tukey, or Blackman window
- Signal-length, next-power-of-two, or custom NFFT
- Raw, Gaussian, or ideal-bandpass detector model
- Visual dB reference independent of ratio calculations
- Peak, RMS, arrival, bandwidth, integrated power, slope, intercept, midband,
  and ratio metrics

Zero padding changes displayed bin spacing but not true frequency resolution;
the application reports both values.

## Tests

```bash
python -m unittest discover -s tests -v
```

The regression suite uses only synthetic arrays and in-memory synthetic CSV
text. It does not read private or experimental COMSOL datasets. Tests cover
generic parsing, filename detector inference, safe filters, grouped
differences, FORCE broadcast, configurable FFT, matched detector/fluence
ratios, the expected 20 dB-per-decade synthetic trend, DIFF/FORCE agreement,
manual Point Evaluation mapping, seconds-to-nanoseconds conversion, normalized
long CSV export, and ZIP contents.

---

## Notes

Do not upload private COMSOL files or unpublished research data. The `examples/` CSV is synthetic demonstration data only.

---

## License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2026 Hatice Guzel (HaticeGuzell)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
