#!/usr/bin/env python3
"""Project presets for common PA research workflows."""

from __future__ import annotations

import copy


PRESETS = {
    "Current 3D Fluence Sweep": {
        "roles": [
            {"name": "OFF", "filter": "pair_case == 3", "broadcast_keys": ""},
            {"name": "ON", "filter": "pair_case == 4", "broadcast_keys": ""},
            {"name": "FORCE", "filter": "pair_case == 5", "broadcast_keys": "Phi_local"},
        ],
        "derived": [{"name": "DIFF", "formula": "ON - OFF"}],
        "group_keys": ["detector", "Phi_local"],
        "fft": {
            "gate_start_ns": 30.0,
            "gate_end_ns": 200.0,
            "window": "Hamming",
            "detrend": "Mean subtraction",
            "nfft_mode": "Custom",
            "custom_nfft": 16384,
        },
        "detector": {
            "mode": "Hysi 5 MHz, 60% Gaussian",
            "center_mhz": 5.0,
            "bw_fraction": 0.60,
        },
    },
    "2D Multi-RBC Aggregation": {
        "roles": [
            {
                "name": "Dispersed_OFF",
                "filter": 'agg_state == "dispersed" AND force_on == 0',
                "broadcast_keys": "",
            },
            {
                "name": "Aggregated_OFF",
                "filter": 'agg_state == "aggregated" AND force_on == 0',
                "broadcast_keys": "",
            },
            {
                "name": "Aggregated_ON",
                "filter": 'agg_state == "aggregated" AND force_on == 1',
                "broadcast_keys": "",
            },
        ],
        "derived": [
            {"name": "Delta_structure", "formula": "Aggregated_OFF - Dispersed_OFF"},
            {"name": "Delta_force", "formula": "Aggregated_ON - Aggregated_OFF"},
            {"name": "Delta_total", "formula": "Aggregated_ON - Dispersed_OFF"},
        ],
        "group_keys": ["detector", "N_RBC"],
    },
    "Detector Comparison": {
        "roles": [
            {"name": "OFF", "filter": "pair_case == 3", "broadcast_keys": ""},
            {"name": "ON", "filter": "pair_case == 4", "broadcast_keys": ""},
        ],
        "derived": [{"name": "DIFF", "formula": "ON - OFF"}],
        "group_keys": ["detector", "Phi_local"],
    },
    "Custom": {
        "roles": [{"name": "Signal", "filter": "", "broadcast_keys": ""}],
        "derived": [],
        "group_keys": ["detector"],
    },
}


def get_preset(name):
    return copy.deepcopy(PRESETS[name])
