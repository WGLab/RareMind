"""Incongruity match score.

Rewards a candidate that explains BOTH the patient's dominant clinical pattern
AND the patient's incongruous phenotypes. This is the score that specifically
lifts cases like CCMS (neurodevelopmental + rib) where the distinguishing
signal is a minority phenotype in a specific organ system.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology


def incongruity_match_score(
    disease_id: str,
    incongruity: Dict[str, Any],
    kg_index: KGIndex,
    hpo: HpoOntology,
) -> Dict[str, Any]:
    if not incongruity:
        return {"incongruity_match_score": 0.0, "details": []}

    dominant = incongruity.get("dominant_branch")
    inc_phens = incongruity.get("incongruous_phenotypes", []) or []
    if not dominant or not inc_phens:
        return {"incongruity_match_score": 0.0, "details": []}

    disease_branches = kg_index.disease_all_branches.get(disease_id, set())
    if dominant not in disease_branches:
        # Disease doesn't match the dominant branch either; no incongruity bridge
        return {"incongruity_match_score": 0.0, "details": []}

    details = []
    total = 0.0
    disease_phens = kg_index.disease_phenotype_hpos.get(disease_id, set())
    disease_charact = kg_index.disease_characteristic_hpos.get(disease_id, set())

    for inc in inc_phens:
        inc_branch = inc.get("branch")
        if not inc_branch:
            continue
        if inc_branch not in disease_branches:
            continue  # disease doesn't reach this incongruous branch

        # Check whether disease has a characteristic phenotype in this branch
        bridge_characteristic = False
        for dphid in disease_charact:
            if inc_branch in hpo.get_branches(dphid):
                bridge_characteristic = True
                break

        if bridge_characteristic:
            credit = inc.get("ic", 0.0) * 2.0
            tier = "strong"
        elif any(inc_branch in hpo.get_branches(dp) for dp in disease_phens):
            credit = inc.get("ic", 0.0) * 0.5
            tier = "weak"
        else:
            continue

        total += credit
        details.append({
            "patient_hpo": inc.get("hpo_id"),
            "patient_mention": inc.get("mention"),
            "matched_branch": inc_branch,
            "tier": tier,
            "credit": credit,
        })

    return {"incongruity_match_score": total, "details": details}
