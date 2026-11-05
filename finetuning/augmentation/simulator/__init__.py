# coding=utf-8
"""Vendored mega-asr-sim simulator (MIT, see LICENSE)."""

from .core import (
    NoisePool,
    RIRPool,
    apply_scenario,
    available_scenarios,
    load_audio,
    resolve_scenario_weights,
)
from .scenarios import SCENARIOS
from .severity import sample_severity

__all__ = [
    "NoisePool",
    "RIRPool",
    "SCENARIOS",
    "apply_scenario",
    "available_scenarios",
    "load_audio",
    "resolve_scenario_weights",
    "sample_severity",
]
