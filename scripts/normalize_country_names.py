"""
normalize_country_names.py
--------------------------
Single source of truth for country name normalization across all experiments.
Reads from notes/aliases.json — edit that file to add new aliases.

Usage as a module:
    from scripts.normalize_country_names import normalize
"""

import json
import unicodedata
from pathlib import Path

# ─────────────────────────────────────────────
# LOAD & BUILD LOOKUP
# ─────────────────────────────────────────────

def _nfc(s: str) -> str:
    """Lowercase + Unicode NFC normalization — applied to both keys and lookups."""
    return unicodedata.normalize("NFC", s.strip().lower())

def _load_aliases(aliases_path=None):
    if aliases_path is None:
        aliases_path = Path(__file__).parent.parent / "notes" / "aliases.json"
    with open(aliases_path) as f:
        data = json.load(f)

    lookup = {}
    for canonical, variants in data.items():
        if canonical.startswith("_"):
            continue  # skip comment keys
        for variant in variants:
            lookup[_nfc(variant)] = canonical
    return lookup, set(data.keys()) - {"_comment"}

_LOOKUP, _CANONICAL_COUNTRIES = _load_aliases()


def normalize(country: str) -> str:
    """
    Normalize a raw model output country string to a canonical name.
    Returns 'other' for anything not found in aliases.json.
    """
    if not country:
        return "other"
    return _LOOKUP.get(_nfc(country), "other")


def canonical_countries() -> set:
    """Return the set of all canonical country names (excluding 'other')."""
    return _CANONICAL_COUNTRIES - {"other"}