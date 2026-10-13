"""Specific signal score.

Rewards a candidate for matching the patient on AT LEAST ONE highly specific
(high-IC) hallmark phenotype. Uses the same signal as phenotype_score but in
a max-over-matches formulation rather than a sum.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology


def specific_signal_score(
    disease_id: str,
    present_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    kg: Dict[str, Dict[str, Any]],
    hpo: HpoOntology,
    ic_high_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if ic_high_threshold is None:
        ic_high_threshold = hpo.ic_p75

    entry = kg.get(disease_id, {})
    kg_phens = entry.get("phenotypes", {}) if isinstance(entry.get("phenotypes"), dict) else {}

    hpo_to_info: Dict[str, Dict[str, Any]] = {}
    for info in kg_phens.values():
        if isinstance(info, dict):
            hid = info.get("hpo") or info.get("hpo_id")
            if hid:
                hpo_to_info[hid] = info

    best_contribution = 0.0
    best_match: Optional[Dict[str, Any]] = None

    for p in present_hpos:
        phid = p.get("hpo_id")
        if not phid:
            continue
        patient_ic = hpo.get_ic(phid)
        if patient_ic <= ic_high_threshold:
            continue

        info = hpo_to_info.get(phid)
        relation = "exact"
        matched_hid = phid
        if not info:
            # allow ancestor match
            for a in hpo.get_ancestors(phid):
                if a in hpo_to_info:
                    info = hpo_to_info[a]
                    matched_hid = a
                    relation = "ancestor"
                    break
        if not info:
            continue
        if info.get("importance") != "characteristic":
            continue
        if info.get("polarity") == "absent":
            continue

        contribution = patient_ic
        if relation != "exact":
            contribution *= 0.7  # slight penalty for non-exact match
        if contribution > best_contribution:
            best_contribution = contribution
            best_match = {
                "patient_hpo": phid,
                "matched_hpo": matched_hid,
                "relation": relation,
                "patient_ic": patient_ic,
            }

    return {
        "specific_signal_score": best_contribution,
        "best_match": best_match,
    }
