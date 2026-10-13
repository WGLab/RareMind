"""Phenotype score component (Stage 3).

- IC-weighted matching (higher IC = more specific = more weight)
- IC-ratio soft credit for ancestor/descendant matches
- Importance × frequency weighting of disease annotations
- Reuse decay (repeated matches to same disease HPO diminish)
- Frequency-aware NEGATION penalty: cost of an explicit denial scales with
  `importance × frequency` of the disease's phenotype (denying a characteristic-frequent
  feature costs more than denying an incidental-rare feature)
- Competitive IC downweight: if a patient HPO is characteristic in many candidates,
  its effective weight for each candidate is reduced.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from raremind.kg.kg_precompute import KGIndex, IMPORTANCE_WEIGHT, FREQUENCY_WEIGHT
from raremind.normalize.hpo_ontology import HpoOntology


@dataclass
class PhenotypeScoreConfig:
    reuse_decay: float = 0.5
    neg_min_penalty: float = 0.1
    competitive_ic_max_downweight: float = 0.5
    direct_relation_floor: float = 0.5
    cousin_similarity_threshold: float = 0.6
    cousin_lca_ic_threshold: float = 2.0
    disease_size_norm_cap: int = 60


# Stage-3 phenotype matching uses softer disease-label weights than the
# retrieval/hallmark index. A diagnosis can be supported by a constellation of
# lower-frequency features, especially in HPO-only benchmark cases where
# pathognomonic hallmarks may be absent from the patient profile.
PHENOTYPE_IMPORTANCE_WEIGHT = {
    "characteristic": 1.0,
    "supportive": 0.85,
    "incidental": 0.25,
}

PHENOTYPE_FREQUENCY_WEIGHT = {
    "very_common": 1.0,
    "common": 0.9,
    "more_than_half": 0.8,
    "occasional": 0.65,
    "rare": 0.45,
    "unknown": 0.6,
    "": 0.6,
    None: 0.6,
}


ANTONYM_PREFIX_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("macro", "micro"),
    ("hyper", "hypo"),
    ("tachy", "brady"),
    ("increased", "decreased"),
    ("long", "short"),
    ("large", "small"),
    ("high", "low"),
    ("proximal", "distal"),
    ("anterior", "posterior"),
    ("superior", "inferior"),
    ("upper", "lower"),
    ("medial", "lateral"),
    ("early", "late"),
    ("open", "closed"),
)


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", str(text).lower().replace("-", " "))


def _strip_antonym_prefix(tokens: List[str], prefix: str) -> Optional[List[str]]:
    if not tokens:
        return None
    head = tokens[0]
    if head == prefix:
        return tokens[1:]
    if head.startswith(prefix) and len(head) > len(prefix):
        return [head[len(prefix):]] + tokens[1:]
    return None


def _are_antonymic_pair(name_a: str, name_b: str) -> bool:
    toks_a = _normalize_tokens(name_a)
    toks_b = _normalize_tokens(name_b)
    if not toks_a or not toks_b:
        return False

    for left, right in ANTONYM_PREFIX_PAIRS:
        base_a = _strip_antonym_prefix(toks_a, left)
        base_b = _strip_antonym_prefix(toks_b, right)
        if base_a is not None and base_b is not None and base_a == base_b and base_a:
            return True
        base_a = _strip_antonym_prefix(toks_a, right)
        base_b = _strip_antonym_prefix(toks_b, left)
        if base_a is not None and base_b is not None and base_a == base_b and base_a:
            return True
    return False


def _direct_credit(relation: str, patient_ic: float, matched_ic: float, floor: float) -> float:
    if relation == "parent":
        ratio = matched_ic / patient_ic if patient_ic > 0 else floor
    elif relation == "child":
        ratio = patient_ic / matched_ic if matched_ic > 0 else floor
    else:
        hi = max(patient_ic, matched_ic)
        lo = min(patient_ic, matched_ic)
        ratio = (lo / hi) if hi > 0 else floor
    return min(1.0, max(floor, ratio))


def _safe_ic(hpo: HpoOntology, hid: str) -> float:
    return max(0.0, float(hpo.get_ic(hid)))


def _phenotype_label_weights(info: Dict[str, Any]) -> Tuple[float, float]:
    imp_w = PHENOTYPE_IMPORTANCE_WEIGHT.get(info.get("importance", "incidental"), 0.25)
    freq_w = PHENOTYPE_FREQUENCY_WEIGHT.get(info.get("frequency"), 0.6)
    return imp_w, freq_w


def _match_priority(info: Dict[str, Any], hpo: HpoOntology, hid: str, credit: float, bonus: float = 0.0) -> float:
    disease_ic = _safe_ic(hpo, hid)
    imp_w, freq_w = _phenotype_label_weights(info)
    return disease_ic * imp_w * freq_w * credit + bonus


def phenotype_score(
    disease_id: str,
    present_hpos: List[Dict[str, Any]],
    negated_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    kg: Dict[str, Dict[str, Any]],
    hpo: HpoOntology,
    total_candidates: int,
    hpo_normalizer: Any | None = None,
    cfg: PhenotypeScoreConfig | None = None,
) -> Dict[str, Any]:
    """Compute the phenotype score (sum) + details for one disease."""
    if cfg is None:
        cfg = PhenotypeScoreConfig()

    entry = kg.get(disease_id, {})
    kg_phens = entry.get("phenotypes", {}) if isinstance(entry.get("phenotypes"), dict) else {}

    # Map hpo_id → annotation info for this disease
    hpo_to_info: Dict[str, Dict[str, Any]] = {}
    for _, info in kg_phens.items():
        if not isinstance(info, dict):
            continue
        hid = info.get("hpo") or info.get("hpo_id")
        if hid:
            hpo_to_info[str(hid)] = info

    matched_hpos: List[Dict[str, Any]] = []
    score_sum = 0.0
    match_counter_per_kg_hpo: Dict[str, int] = {}

    for p in present_hpos:
        phid = p.get("hpo_id")
        if not phid:
            continue
        phid = str(phid)
        patient_ic = _safe_ic(hpo, phid)
        patient_name = hpo.get_name(phid)

        # Competitive IC downweight
        total_hosts = len(kg_index.hpo_to_diseases.get(phid, set()))
        share = (total_hosts / total_candidates) if total_candidates > 0 else 0.0
        share = min(1.0, share)
        comp_weight = 1.0 - cfg.competitive_ic_max_downweight * share

        # Tier 1: exact match
        info = hpo_to_info.get(phid)
        matched_hid = phid if info else None
        relation = "exact"
        ic_ratio = 1.0
        match_tier = 1
        biolord_similarity: float | None = 1.0
        lca_ic = patient_ic

        if info and info.get("polarity") == "absent":
            info = None
            matched_hid = None

        if not info:
            direct_candidates: List[Tuple[float, str, Dict[str, Any], str, float]] = []
            candidate_groups = (
                ("parent", hpo.get_parents(phid)),
                ("child", hpo.get_children(phid)),
                ("sibling", hpo.get_siblings(phid)),
            )
            for rel_name, candidates in candidate_groups:
                for hid in candidates:
                    candidate_info = hpo_to_info.get(hid)
                    if not candidate_info or candidate_info.get("polarity") == "absent":
                        continue
                    if _are_antonymic_pair(patient_name, hpo.get_name(hid)):
                        continue
                    credit = _direct_credit(
                        rel_name,
                        patient_ic=patient_ic,
                        matched_ic=_safe_ic(hpo, hid),
                        floor=cfg.direct_relation_floor,
                    )
                    direct_candidates.append((
                        _match_priority(candidate_info, hpo, hid, credit),
                        hid,
                        candidate_info,
                        rel_name,
                        credit,
                    ))

            if direct_candidates:
                _, matched_hid, info, relation, ic_ratio = max(direct_candidates, key=lambda x: x[0])
                match_tier = 2
                biolord_similarity = None
                lca_ic = hpo.get_mica_ic(phid, matched_hid)

        if not info and hpo_normalizer is not None:
            direct_related = {phid} | hpo.get_parents(phid) | hpo.get_children(phid) | hpo.get_siblings(phid)
            cousin_candidates: List[Tuple[float, str, Dict[str, Any], float, float]] = []
            for hid, candidate_info in hpo_to_info.items():
                if hid in direct_related or candidate_info.get("polarity") == "absent":
                    continue
                candidate_name = hpo.get_name(hid)
                if _are_antonymic_pair(patient_name, candidate_name):
                    continue
                similarity = float(hpo_normalizer.similarity_by_hpo_id(phid, hid))
                if similarity < cfg.cousin_similarity_threshold:
                    continue
                mica_ic = hpo.get_mica_ic(phid, hid)
                if mica_ic <= cfg.cousin_lca_ic_threshold:
                    continue
                cousin_candidates.append((
                    _match_priority(
                        candidate_info,
                        hpo,
                        hid,
                        cfg.direct_relation_floor,
                        bonus=(0.05 * similarity + 0.01 * mica_ic),
                    ),
                    hid,
                    candidate_info,
                    similarity,
                    mica_ic,
                ))

            if cousin_candidates:
                _, matched_hid, info, biolord_similarity, lca_ic = max(cousin_candidates, key=lambda x: x[0])
                relation = "cousin"
                ic_ratio = cfg.direct_relation_floor
                match_tier = 3

        if not info or not matched_hid:
            continue

        # Reuse decay
        use_count = match_counter_per_kg_hpo.get(matched_hid, 0)
        decay = cfg.reuse_decay ** use_count

        imp_w, freq_w = _phenotype_label_weights(info)

        contribution = (
            patient_ic
            * ic_ratio
            * imp_w
            * freq_w
            * decay
            * comp_weight
        )
        # reliability weighting if present
        reliability = p.get("reliability")
        if reliability == "low":
            contribution *= 0.7

        score_sum += contribution
        matched_hpos.append({
            "patient_hpo": phid,
            "matched_hpo": matched_hid,
            "relation": relation,
            "match_tier": match_tier,
            "importance": info.get("importance"),
            "frequency": info.get("frequency"),
            "patient_ic": patient_ic,
            "ic_ratio": ic_ratio,
            "biolord_similarity": biolord_similarity,
            "lca_ic": lca_ic,
            "competitive_weight": comp_weight,
            "reuse_decay": decay,
            "contribution": contribution,
        })
        match_counter_per_kg_hpo[matched_hid] = use_count + 1

    # Negation penalty
    neg_penalty = 0.0
    for n in negated_hpos or []:
        nhid = n.get("hpo_id")
        if not nhid:
            continue
        nhid = str(nhid)
        info = hpo_to_info.get(nhid)
        if not info:
            # Also penalize if a descendant of the negated term is a disease annotation
            for d in hpo.get_descendants(nhid):
                if d in hpo_to_info:
                    info = hpo_to_info[d]
                    break
        if not info:
            continue
        if info.get("polarity") == "absent":
            # disease expects absent — patient says absent → consistent
            continue
        imp_w = IMPORTANCE_WEIGHT.get(info.get("importance", "incidental"), 0.35)
        freq_w = FREQUENCY_WEIGHT.get(info.get("frequency"), 0.5)
        pen = (cfg.neg_min_penalty + 0.8 * imp_w * freq_w) * _safe_ic(hpo, nhid)
        neg_penalty += pen

    # Disease-size normalization
    size = max(1, len(kg_phens))
    effective_size = min(size, max(1, cfg.disease_size_norm_cap))
    normalized = score_sum / (effective_size ** 0.5)

    raw = normalized - neg_penalty
    return {
        "phenotype_score": raw,
        "raw_score_sum": score_sum,
        "negation_penalty": neg_penalty,
        "matched_hpos": matched_hpos,
        "disease_size": size,
    }
