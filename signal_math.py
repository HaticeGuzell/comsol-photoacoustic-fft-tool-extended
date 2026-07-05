#!/usr/bin/env python3
"""Safe role filters and grouped signal arithmetic."""

from __future__ import annotations

import ast
import copy
import re
import warnings

import numpy as np


_CLAUSE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(==|!=|<=|>=|<|>)\s*(.+?)\s*$"
)
_IN_CLAUSE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(\[.*\])\s*$",
    re.IGNORECASE,
)
_FORMULA = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\u0394\u03b4]*)\s*([+-])\s*"
    r"([A-Za-z_][A-Za-z0-9_\u0394\u03b4]*)\s*$"
)


def _literal(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        raw = text.strip().strip("\"'")
        try:
            number = float(raw)
            return int(number) if number.is_integer() else number
        except ValueError:
            return raw


def _equal(left, right, tolerance=1e-9):
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    return str(left) == str(right)


def compile_filter(expression):
    """Compile the deliberately small filter language without ``eval``."""
    expression = (expression or "").strip()
    if not expression:
        return lambda params: True
    if re.search(r"\bOR\b|[()]", expression, re.IGNORECASE):
        raise ValueError("OR and parenthesized expressions are not supported.")
    clauses = re.split(r"\s+AND\s+", expression, flags=re.IGNORECASE)
    tests = []
    for clause in clauses:
        in_match = _IN_CLAUSE.match(clause)
        if in_match:
            key = in_match.group(1)
            value = _literal(in_match.group(2))
            if not isinstance(value, (list, tuple, set)):
                raise ValueError(f"'in' requires a list: {clause}")
            tests.append((key, "in", list(value)))
            continue
        match = _CLAUSE.match(clause)
        if not match:
            raise ValueError(f"Unsupported filter clause: {clause}")
        tests.append((match.group(1), match.group(2), _literal(match.group(3))))

    def predicate(params):
        for key, operator, expected in tests:
            if key not in params:
                return False
            actual = params[key]
            if operator == "in":
                passed = any(_equal(actual, item) for item in expected)
            elif operator == "==":
                passed = _equal(actual, expected)
            elif operator == "!=":
                passed = not _equal(actual, expected)
            else:
                try:
                    left, right = float(actual), float(expected)
                    passed = {
                        "<": left < right,
                        "<=": left <= right,
                        ">": left > right,
                        ">=": left >= right,
                    }[operator]
                except (TypeError, ValueError):
                    passed = False
            if not passed:
                return False
        return True

    return predicate


def assign_roles(traces, role_definitions):
    """Return role -> trace collection using safe filter expressions."""
    collections = {}
    for definition in role_definitions:
        name = str(definition.get("name", "")).strip()
        expression = str(definition.get("filter", "")).strip()
        if not name:
            continue
        predicate = compile_filter(expression)
        matches = []
        for trace in traces:
            if predicate(trace.get("params", {})):
                clone = copy.deepcopy(trace)
                clone["role"] = name
                clone.setdefault("params", {})["role"] = name
                matches.append(clone)
        collections[name] = matches
    return collections


def _group_key(trace, group_keys, excluded=()):
    params = trace.get("params", {})
    excluded = set(excluded)
    return tuple(
        (key, params.get(key))
        for key in group_keys
        if key not in excluded
    )


def materialize_broadcast_roles(collections, group_keys, broadcast_by_role):
    """Clone broadcast roles across groups observed in non-broadcast roles."""
    target_groups = {}
    for role, traces in collections.items():
        if broadcast_by_role.get(role):
            continue
        for trace in traces:
            target_groups[_group_key(trace, group_keys)] = trace.get("params", {})
    if not target_groups:
        return collections

    result = {role: list(traces) for role, traces in collections.items()}
    for role, missing_keys in broadcast_by_role.items():
        sources = collections.get(role, [])
        if not sources:
            continue
        expanded = []
        for target_key, target_params in target_groups.items():
            candidates = [
                trace for trace in sources
                if _group_key(trace, group_keys, missing_keys)
                == tuple(item for item in target_key if item[0] not in set(missing_keys))
            ]
            if len(candidates) != 1:
                if len(candidates) > 1:
                    warnings.warn(
                        f"Broadcast role {role!r} is ambiguous for group {target_key}."
                    )
                continue
            clone = copy.deepcopy(candidates[0])
            for key in missing_keys:
                if key in target_params:
                    clone["params"][key] = target_params[key]
            clone["params"]["role"] = role
            clone["role"] = role
            expanded.append(clone)
        if expanded:
            result[role] = expanded
    return result


def align_time_grids(trace_a, trace_b, relative_tolerance=0.05):
    """Align two traces; interpolate small grid differences onto trace A."""
    ta = np.asarray(trace_a["time_ns"], dtype=float)
    tb = np.asarray(trace_b["time_ns"], dtype=float)
    pa = np.asarray(trace_a["pressure_pa"], dtype=float)
    pb = np.asarray(trace_b["pressure_pa"], dtype=float)
    if len(ta) == len(tb) and np.allclose(ta, tb, rtol=1e-9, atol=1e-9):
        return ta, pa, pb
    overlap_start = max(float(np.min(ta)), float(np.min(tb)))
    overlap_end = min(float(np.max(ta)), float(np.max(tb)))
    if overlap_end <= overlap_start:
        raise ValueError("Time grids do not overlap.")
    span = max(float(np.ptp(ta)), float(np.ptp(tb)), 1e-12)
    lost = 1.0 - (overlap_end - overlap_start) / span
    if lost > relative_tolerance:
        warnings.warn(
            f"Time-grid mismatch removes {lost:.1%} of the span; "
            "interpolating over the common interval."
        )
    mask = (ta >= overlap_start) & (ta <= overlap_end)
    common = ta[mask]
    return common, pa[mask], np.interp(common, tb, pb)


def combine_traces(trace_a, trace_b, operator, role_name):
    time_ns, left, right = align_time_grids(trace_a, trace_b)
    pressure = left - right if operator == "-" else left + right
    params = {
        key: value for key, value in trace_a.get("params", {}).items()
        if key != "role"
    }
    params["role"] = role_name
    return {
        "time_ns": time_ns,
        "pressure_pa": pressure,
        "params": params,
        "role": role_name,
        "source_file": trace_a.get("source_file", ""),
        "source_format": "derived",
        "formula": f"{trace_a.get('role')} {operator} {trace_b.get('role')}",
    }


def derive_signals(collections, derived_definitions, group_keys):
    """Compute formulas per matched group and return updated collections."""
    output = {name: list(traces) for name, traces in collections.items()}
    diagnostics = []
    for definition in derived_definitions:
        name = str(definition.get("name", "")).strip()
        formula = str(definition.get("formula", "")).strip()
        if not name or not formula:
            continue
        match = _FORMULA.match(formula)
        if not match:
            raise ValueError(
                f"Formula {formula!r} must have the form ROLE_A - ROLE_B or ROLE_A + ROLE_B."
            )
        left_role, operator, right_role = match.groups()
        left_traces = output.get(left_role, [])
        right_index = {}
        for trace in output.get(right_role, []):
            right_index.setdefault(_group_key(trace, group_keys), []).append(trace)
        generated = []
        for left_trace in left_traces:
            key = _group_key(left_trace, group_keys)
            candidates = right_index.get(key, [])
            if len(candidates) == 1:
                generated.append(
                    combine_traces(left_trace, candidates[0], operator, name)
                )
            elif not candidates:
                diagnostics.append(f"{name}: no {right_role} match for {key}")
            else:
                diagnostics.append(f"{name}: ambiguous {right_role} match for {key}")
        output[name] = generated
    return output, diagnostics


def flatten_collections(collections):
    return [trace for traces in collections.values() for trace in traces]


def subtract(signal_a, signal_b):
    return np.asarray(signal_a) - np.asarray(signal_b)


def add(signal_a, signal_b):
    return np.asarray(signal_a) + np.asarray(signal_b)


def scale(signal, factor):
    return np.asarray(signal) * factor


def mean(signals):
    return np.mean(np.asarray(signals), axis=0)


def rms(signal):
    values = np.asarray(signal)
    return float(np.sqrt(np.mean(values ** 2)))
