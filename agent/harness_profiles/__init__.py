"""Harness profiles — per-model-family prompt customisation registry.

See :mod:`agent.harness_profiles.profiles` for the dataclass and resolution
logic.  This package exists so the module can be extended later (e.g. a
separate data file for W2's tool-description overrides) without bloating a
single flat file.
"""

from agent.harness_profiles.profiles import (
    GENERIC_PROFILE,
    MIMO_PROFILE,
    HarnessProfile,
    get_profile_by_name,
    resolve_profile,
)

__all__ = [
    "HarnessProfile",
    "resolve_profile",
    "get_profile_by_name",
    "GENERIC_PROFILE",
    "MIMO_PROFILE",
]
