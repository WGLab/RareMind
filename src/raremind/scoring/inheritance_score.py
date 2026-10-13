"""Inheritance score component.

Uses the inheritance prior from patient family history (Stage 2.5) and the
disease's inheritance mode(s) from KG. Returns a score in [0, 1].
"""
from __future__ import annotations

from typing import Any, Dict, List

from raremind.normalize.inheritance_inference import inheritance_compatibility_flag


def inheritance_score(
    disease_id: str,
    inheritance_prior: Dict[str, float],
    disease_inheritance: List[str],
) -> Dict[str, Any]:
    if not inheritance_prior or sum(inheritance_prior.values()) == 0:
        return {
            "inheritance_score": 0.0,
            "compatibility": "UNKNOWN",
            "reason": "no family history available",
        }

    flag, weight = inheritance_compatibility_flag(inheritance_prior, disease_inheritance)
    return {
        "inheritance_score": weight,
        "compatibility": flag,
        "disease_modes": disease_inheritance,
    }
