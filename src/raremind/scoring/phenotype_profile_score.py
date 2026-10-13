"""Coverage-aware phenotype profile score.

The original phenotype score is a weighted sum of matched HPO IC divided by
sqrt(number of disease annotations). That is useful for precise hallmark
matches, but it can under-rank diseases whose evidence is a coherent set of
supportive/occasional findings spread across multiple organ systems.

This module adds a profile-style score that asks:
  - How much of the patient's IC-weighted phenotype profile is explained?
  - Does the disease explain multiple patient organ-system branches?
  - Are there enough independent matches to avoid one-feature false positives?
  - Do multiple supportive/occasional features form a coherent pattern?
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology


PROFILE_IMPORTANCE_WEIGHT = {
    "characteristic": 1.0,
    "supportive": 0.85,
    "incidental": 0.25,
}

PROFILE_FREQUENCY_WEIGHT = {
    "very_common": 1.0,
    "common": 0.9,
    "more_than_half": 0.8,
    "occasional": 0.65,
    "rare": 0.45,
    "unknown": 0.6,
    "": 0.6,
    None: 0.6,
}


@dataclass
class PhenotypeProfileConfig:
    direct_relation_floor: float = 0.55
    sibling_credit: float = 0.45
    ancestor_credit: float = 0.35
    max_ancestor_depth: int = 2
    min_patient_ic: float = 0.0
    min_supportive_cluster_matches: int = 3
    min_supportive_cluster_branches: int = 2
    generic_guard_patient_ic_threshold: float = 3.0
    generic_guard_matched_ic_threshold: float = 3.0
    anchor_high_ic_threshold: float = 4.0


def _patient_hpo_id(item: Dict[str, Any]) -> Optional[str]:
    hid = item.get("hpo_id") or item.get("hpo") or item.get("id")
    if hid and str(hid).startswith("HP:"):
        return str(hid)
    return None


def _safe_ic(hpo: HpoOntology, hpo_id: str) -> float:
    """Clamp broad/negative IC terms so they cannot subtract evidence."""
    return max(0.0, float(hpo.get_ic(hpo_id)))


def _label_weight(info: Dict[str, Any]) -> float:
    imp = PROFILE_IMPORTANCE_WEIGHT.get(info.get("importance", "incidental"), 0.25)
    freq = PROFILE_FREQUENCY_WEIGHT.get(info.get("frequency"), 0.6)
    return imp * freq


def _direct_credit(relation: str, patient_ic: float, matched_ic: float, floor: float) -> float:
    if relation == "exact":
        return 1.0
    if relation == "parent":
        ratio = matched_ic / patient_ic if patient_ic > 0 else floor
    elif relation == "child":
        ratio = patient_ic / matched_ic if matched_ic > 0 else floor
    else:
        hi = max(patient_ic, matched_ic)
        lo = min(patient_ic, matched_ic)
        ratio = lo / hi if hi > 0 else floor
    return min(1.0, max(floor, ratio))


def _limited_ancestors(hpo: HpoOntology, hid: str, depth: int) -> Set[str]:
    frontier = {hid}
    out: Set[str] = set()
    for _ in range(depth):
        nxt: Set[str] = set()
        for node in frontier:
            for parent in hpo.get_parents(node):
                if parent not in out:
                    out.add(parent)
                    nxt.add(parent)
        frontier = nxt
        if not frontier:
            break
    return out


def _best_match(
    patient_hpo: str,
    hpo_to_info: Dict[str, Dict[str, Any]],
    hpo: HpoOntology,
    cfg: PhenotypeProfileConfig,
) -> Optional[Tuple[str, Dict[str, Any], str, float]]:
    """Return best disease HPO match as (matched_hpo, info, relation, credit)."""
    patient_ic = _safe_ic(hpo, patient_hpo)

    info = hpo_to_info.get(patient_hpo)
    if info and info.get("polarity") != "absent":
        return patient_hpo, info, "exact", 1.0

    candidates: List[Tuple[float, str, Dict[str, Any], str, float]] = []
    for rel_name, hids in (
        ("parent", hpo.get_parents(patient_hpo)),
        ("child", hpo.get_children(patient_hpo)),
        ("sibling", hpo.get_siblings(patient_hpo)),
    ):
        for hid in hids:
            info = hpo_to_info.get(hid)
            if not info or info.get("polarity") == "absent":
                continue
            if rel_name == "sibling":
                credit = cfg.sibling_credit
            else:
                credit = _direct_credit(rel_name, patient_ic, _safe_ic(hpo, hid), cfg.direct_relation_floor)
            priority = _safe_ic(hpo, hid) * _label_weight(info) * credit
            candidates.append((priority, hid, info, rel_name, credit))

    if not candidates and cfg.max_ancestor_depth > 1:
        for hid in _limited_ancestors(hpo, patient_hpo, cfg.max_ancestor_depth):
            info = hpo_to_info.get(hid)
            if not info or info.get("polarity") == "absent":
                continue
            priority = _safe_ic(hpo, hid) * _label_weight(info) * cfg.ancestor_credit
            candidates.append((priority, hid, info, "ancestor", cfg.ancestor_credit))

    if not candidates:
        return None
    _, hid, info, relation, credit = max(candidates, key=lambda x: x[0])
    return hid, info, relation, credit


def phenotype_profile_score(
    disease_id: str,
    present_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    kg: Dict[str, Dict[str, Any]],
    hpo: HpoOntology,
    cfg: PhenotypeProfileConfig | None = None,
) -> Dict[str, Any]:
    """Compute a coverage-aware patient-to-disease phenotype profile score."""
    if cfg is None:
        cfg = PhenotypeProfileConfig()

    entry = kg.get(disease_id, {})
    kg_phens = entry.get("phenotypes", {}) if isinstance(entry.get("phenotypes"), dict) else {}
    hpo_to_info: Dict[str, Dict[str, Any]] = {}
    for info in kg_phens.values():
        if not isinstance(info, dict):
            continue
        hid = info.get("hpo") or info.get("hpo_id")
        if hid and str(hid).startswith("HP:"):
            hpo_to_info[str(hid)] = info

    patient_terms: List[Tuple[str, float, Set[str]]] = []
    seen: Set[str] = set()
    for p in present_hpos:
        hid = _patient_hpo_id(p)
        if not hid or hid in seen:
            continue
        seen.add(hid)
        ic = max(cfg.min_patient_ic, _safe_ic(hpo, hid))
        patient_terms.append((hid, ic, hpo.get_branches(hid)))

    if not patient_terms or not hpo_to_info:
        return {
            "phenotype_profile_score": 0.0,
            "profile_patient_recall": 0.0,
            "profile_unweighted_recall": 0.0,
            "profile_branch_recall": 0.0,
            "profile_match_coverage": 0.0,
            "profile_supportive_cluster": 0.0,
            "profile_generic_guard": 0.0,
            "profile_anchor_cluster": 0.0,
            "profile_high_ic_recall": 0.0,
            "profile_specific_hit_fraction": 0.0,
            "profile_match_density": 0.0,
            "profile_matched_count": 0,
        }

    total_patient_ic = sum(ic for _, ic, _ in patient_terms)
    if total_patient_ic <= 0:
        total_patient_ic = float(len(patient_terms))

    matched: List[Dict[str, Any]] = []
    weighted_ic = 0.0
    unweighted_ic = 0.0
    matched_branch_ic = 0.0
    all_branch_ic: Dict[str, float] = {}
    matched_branches: Set[str] = set()
    supportive_branches: Set[str] = set()
    supportive_matches = 0
    hit_ic = 0.0
    specific_hit_ic = 0.0
    high_hit_ic = 0.0
    high_total_ic = sum(
        max(1.0, patient_ic) for _, patient_ic, _ in patient_terms
        if patient_ic >= cfg.anchor_high_ic_threshold
    )

    for phid, patient_ic, branches in patient_terms:
        for branch in branches:
            all_branch_ic[branch] = all_branch_ic.get(branch, 0.0) + patient_ic

        match = _best_match(phid, hpo_to_info, hpo, cfg)
        if not match:
            continue
        mhid, info, relation, credit = match
        matched_ic = _safe_ic(hpo, mhid)
        quality_patient_ic = max(1.0, patient_ic)
        label_w = _label_weight(info)
        contribution = patient_ic * credit * label_w
        ic_credit = quality_patient_ic * credit
        weighted_ic += contribution
        unweighted_ic += patient_ic * credit
        matched_branch_ic += patient_ic
        hit_ic += ic_credit
        if patient_ic >= cfg.anchor_high_ic_threshold:
            high_hit_ic += ic_credit
        if (
            quality_patient_ic >= cfg.generic_guard_patient_ic_threshold
            and matched_ic >= cfg.generic_guard_matched_ic_threshold
            and credit >= cfg.direct_relation_floor
        ):
            specific_hit_ic += ic_credit
        matched_branches.update(branches)
        if info.get("importance") == "supportive" or info.get("frequency") in {"occasional", "rare", "unknown", "", None}:
            supportive_matches += 1
            supportive_branches.update(branches)
        matched.append({
            "patient_hpo": phid,
            "matched_hpo": mhid,
            "relation": relation,
            "credit": credit,
            "importance": info.get("importance"),
            "frequency": info.get("frequency"),
            "patient_ic": patient_ic,
            "matched_ic": matched_ic,
            "label_weight": label_w,
            "contribution": contribution,
        })

    patient_recall = weighted_ic / total_patient_ic if total_patient_ic else 0.0
    unweighted_recall = unweighted_ic / total_patient_ic if total_patient_ic else 0.0
    branch_denom = sum(all_branch_ic.values())
    branch_recall = matched_branch_ic / branch_denom if branch_denom else 0.0
    expected_matches = min(8, max(1, len(patient_terms)))
    match_coverage = min(1.0, math.log1p(len(matched)) / math.log1p(expected_matches))
    supportive_cluster = min(1.0, supportive_matches / cfg.min_supportive_cluster_matches)
    supportive_cluster *= min(1.0, len(supportive_branches) / cfg.min_supportive_cluster_branches)
    match_density = min(1.0, math.log1p(len(matched)) / math.log1p(min(10, max(1, len(patient_terms)))))
    high_ic_recall = high_hit_ic / high_total_ic if high_total_ic else 0.0
    specific_hit_fraction = specific_hit_ic / hit_ic if hit_ic else 0.0
    anchor_cluster = math.sqrt(
        max(0.0, min(1.0, high_ic_recall))
        * max(0.0, min(1.0, match_density))
    )

    score = (
        0.52 * patient_recall
        + 0.18 * unweighted_recall
        + 0.15 * branch_recall
        + 0.10 * match_coverage
        + 0.05 * supportive_cluster
    )

    # Tiny-profile guard: one-feature explanations should not dominate a rich
    # patient HPO list even if the one matched feature is rare.
    if len(patient_terms) >= 5 and len(matched) <= 1:
        score *= 0.45
    elif len(patient_terms) >= 8 and len(matched) <= 2:
        score *= 0.70

    return {
        "phenotype_profile_score": float(score),
        "profile_patient_recall": float(patient_recall),
        "profile_unweighted_recall": float(unweighted_recall),
        "profile_branch_recall": float(branch_recall),
        "profile_match_coverage": float(match_coverage),
        "profile_supportive_cluster": float(supportive_cluster),
        "profile_generic_guard": float(specific_hit_fraction),
        "profile_anchor_cluster": float(anchor_cluster),
        "profile_high_ic_recall": float(high_ic_recall),
        "profile_specific_hit_fraction": float(specific_hit_fraction),
        "profile_match_density": float(match_density),
        "profile_matched_count": len(matched),
        "profile_supportive_matches": supportive_matches,
        "profile_matched_branches": len(matched_branches),
        "profile_matches": matched[:20],
    }
