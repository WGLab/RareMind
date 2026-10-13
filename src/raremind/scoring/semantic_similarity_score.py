"""Ontology semantic-similarity score for phenotype-only rescue.

This component complements the exact/direct phenotype scorer. It gives a
candidate credit when a patient HPO and a disease HPO are not exact neighbors
but share an informative common ancestor in the HPO graph.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology


def _patient_hpo_ids(present_hpos: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in present_hpos:
        hid = item.get("hpo_id")
        if hid and hid not in seen:
            seen.add(str(hid))
            out.append(str(hid))
    return out


def semantic_similarity_score(
    disease_id: str,
    present_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    hpo: HpoOntology,
    max_disease_hpos: int = 80,
) -> Dict[str, Any]:
    """Best-match-average MICA IC between patient and disease HPO sets.

    Score definition:
      patient_to_disease = mean_p max_d MICA_IC(p, d)
      disease_to_patient = mean_d max_p MICA_IC(p, d)
      score = 0.7 * patient_to_disease + 0.3 * disease_to_patient

    The first term asks "does the disease explain the patient?" and receives
    more weight. The second term lightly discourages diseases whose phenotype
    profile is broad and only weakly aligned to the patient.
    """
    patient_hpos = _patient_hpo_ids(present_hpos)
    if not patient_hpos:
        return {"semantic_similarity_score": 0.0}

    disease_hpos = list(kg_index.disease_characteristic_hpos.get(disease_id) or [])
    if not disease_hpos:
        disease_hpos = list(kg_index.disease_phenotype_hpos.get(disease_id) or [])
    if not disease_hpos:
        return {"semantic_similarity_score": 0.0}

    disease_hpos = sorted(disease_hpos, key=lambda hid: hpo.get_ic(hid), reverse=True)[:max_disease_hpos]

    p_to_d_vals: List[float] = []
    best_matches: List[Dict[str, Any]] = []
    for phid in patient_hpos:
        best_dhid = None
        best_ic = 0.0
        for dhid in disease_hpos:
            mica_ic = hpo.get_mica_ic(phid, dhid)
            if mica_ic > best_ic:
                best_ic = mica_ic
                best_dhid = dhid
        p_to_d_vals.append(best_ic)
        if best_dhid:
            best_matches.append({
                "patient_hpo": phid,
                "matched_disease_hpo": best_dhid,
                "mica_ic": best_ic,
            })

    d_to_p_vals: List[float] = []
    for dhid in disease_hpos:
        d_to_p_vals.append(max((hpo.get_mica_ic(phid, dhid) for phid in patient_hpos), default=0.0))

    patient_to_disease = sum(p_to_d_vals) / len(p_to_d_vals) if p_to_d_vals else 0.0
    disease_to_patient = sum(d_to_p_vals) / len(d_to_p_vals) if d_to_p_vals else 0.0
    score = 0.7 * patient_to_disease + 0.3 * disease_to_patient

    return {
        "semantic_similarity_score": float(score),
        "semantic_patient_to_disease": float(patient_to_disease),
        "semantic_disease_to_patient": float(disease_to_patient),
        "semantic_best_matches": best_matches[:10],
    }
