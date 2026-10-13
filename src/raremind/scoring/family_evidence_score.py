"""Disease-specific family evidence score.

This component treats family history as supportive inherited-disease context,
not as proband phenotype evidence. It compares relatives' genes, diseases,
phenotypes, and affected systems against each candidate disease and then uses
pedigree inheritance compatibility as a small context modifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from rapidfuzz import fuzz

from raremind.core.compat import to_dict
from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.inheritance_inference import inheritance_compatibility_flag


FAMILY_PHENOTYPE_SIM_WEIGHT = 0.8
FAMILY_GENE_WEIGHT = 1.0
FAMILY_DISEASE_WEIGHT = 0.8
FAMILY_SYSTEM_WEIGHT = 0.15
PEDIGREE_MODE_CONTEXT_WEIGHT = 0.25


SYSTEM_KEYWORDS = {
    "respiratory": ("respiratory", "lung", "pulmonary", "airway", "bronch"),
    "gastrointestinal": ("digestive", "gastro", "intestinal", "liver", "hepatic", "biliary"),
    "hepatic": ("liver", "hepatic", "biliary"),
    "cardiovascular": ("cardiovascular", "heart", "cardiac"),
    "nervous": ("nervous", "neuro", "brain"),
    "skeletal": ("musculoskeletal", "limb", "skeletal", "bone"),
    "genitourinary": ("genitourinary", "renal", "kidney", "urinary"),
    "immune": ("immune", "immun"),
    "integument": ("integument", "skin"),
    "blood": ("blood", "hematologic", "anaemia", "anemia"),
}


@dataclass
class FamilyEvidence:
    terms: List[str] = field(default_factory=list)
    hpos: List[Dict[str, Any]] = field(default_factory=list)
    diseases: List[str] = field(default_factory=list)
    genes: Set[str] = field(default_factory=set)
    systems: Set[str] = field(default_factory=set)
    inheritance_prior: Dict[str, float] = field(default_factory=dict)


def _clean_text(text: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower())
    cleaned = cleaned.replace("bilary", "biliary")
    return re.sub(r"\s+", " ", cleaned).strip()


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def prepare_family_evidence(
    patient_state: Any,
    inheritance_prior: Dict[str, float],
    hpo_normalizer: Any | None,
) -> FamilyEvidence:
    """Extract and normalize family-history evidence once per patient."""
    family_history = patient_state.family_history or []
    family_mentions = [
        to_dict(m)
        for m in (patient_state.phenotype_mentions_text or [])
        if to_dict(m).get("attribution") == "family"
    ]

    terms: List[str] = []
    diseases: List[str] = []
    genes: Set[str] = set()
    systems: Set[str] = set()

    for item in family_mentions:
        mention = item.get("mention")
        if mention:
            terms.append(str(mention))

    for rel in family_history:
        rel = to_dict(rel)
        for disease in rel.get("diseases", []) or []:
            diseases.append(str(disease))
            terms.append(str(disease))
        for phenotype in rel.get("phenotypes", []) or []:
            terms.append(str(phenotype))
        for gene in rel.get("genes", []) or []:
            if gene:
                genes.add(str(gene).upper())
        for system in rel.get("affected_systems", []) or []:
            if system:
                systems.add(_clean_text(system))

    terms = _unique_preserve_order(terms)
    diseases = _unique_preserve_order(diseases)

    hpos: List[Dict[str, Any]] = []
    if hpo_normalizer is not None and terms:
        mentions = [
            {
                "mention": term,
                "attribution": "patient",
                "source": "text",
                "evidence": "family history",
            }
            for term in terms
        ]
        try:
            hpos = hpo_normalizer.normalize(mentions, include_negated=False)
        except Exception:
            hpos = []

    return FamilyEvidence(
        terms=terms,
        hpos=hpos,
        diseases=diseases,
        genes=genes,
        systems=systems,
        inheritance_prior=inheritance_prior or {},
    )


def _related_hpo_score(
    family_hpo: str,
    candidate_hpo: str,
    hpo: Any,
    hpo_normalizer: Any | None,
) -> float:
    if family_hpo == candidate_hpo:
        return 1.0

    try:
        if family_hpo in hpo.get_parents(candidate_hpo) or candidate_hpo in hpo.get_parents(family_hpo):
            return 0.65
        if candidate_hpo in hpo.get_siblings(family_hpo):
            return 0.55
    except Exception:
        pass

    try:
        if family_hpo in hpo.get_ancestors(candidate_hpo) or candidate_hpo in hpo.get_ancestors(family_hpo):
            return 0.45
    except Exception:
        pass

    if hpo_normalizer is not None:
        try:
            sim = hpo_normalizer.similarity_by_hpo_id(family_hpo, candidate_hpo)
            if sim >= 0.65:
                return 0.40
        except Exception:
            pass
    return 0.0


def _family_phenotype_support(
    family: FamilyEvidence,
    disease_id: str,
    kg_index: KGIndex,
    hpo: Any,
    hpo_normalizer: Any | None,
) -> float:
    candidate_hpos = kg_index.disease_phenotype_hpos.get(disease_id, set())
    if not family.hpos or not candidate_hpos:
        return 0.0

    scores: List[float] = []
    for fh in family.hpos:
        fhpo = fh.get("hpo_id")
        if not fhpo:
            continue
        best = 0.0
        for dhpo in candidate_hpos:
            best = max(best, _related_hpo_score(fhpo, dhpo, hpo, hpo_normalizer))
            if best >= 1.0:
                break
        if best > 0:
            ic_factor = min(float(fh.get("ic", 0.0)) / 6.0, 1.0)
            scores.append(best * max(ic_factor, 0.35))

    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return min(sum(scores[:5]) / 3.0, 1.0)


def _family_gene_support(family: FamilyEvidence, disease_id: str, kg_index: KGIndex) -> float:
    if not family.genes:
        return 0.0
    disease_genes = kg_index.disease_genes.get(disease_id, set())
    return 1.0 if family.genes & disease_genes else 0.0


def _family_disease_support(family: FamilyEvidence, disease_id: str, kg_index: KGIndex) -> float:
    if not family.diseases:
        return 0.0
    disease_name = _clean_text(kg_index.disease_name.get(disease_id, disease_id))
    group_id = kg_index.disease_group.get(disease_id, "") or disease_id
    group_name = _clean_text(kg_index.disease_name.get(group_id, group_id))

    best = 0.0
    for fam_disease in family.diseases:
        fam = _clean_text(fam_disease)
        if not fam:
            continue
        if fam == disease_name or fam == group_name:
            best = max(best, 1.0)
        else:
            best = max(
                best,
                fuzz.token_set_ratio(fam, disease_name) / 100.0,
                fuzz.token_set_ratio(fam, group_name) / 100.0,
            )
    if best >= 0.90:
        return 1.0
    if best >= 0.75:
        return 0.5
    return 0.0


def _family_system_support(
    family: FamilyEvidence,
    disease_id: str,
    kg_index: KGIndex,
    hpo: Any,
) -> float:
    if not family.systems:
        return 0.0
    branch_names = [
        _clean_text(hpo.get_name(branch_id))
        for branch_id in kg_index.disease_all_branches.get(disease_id, set())
    ]
    if not branch_names:
        return 0.0

    hits = 0
    for system in family.systems:
        keywords = SYSTEM_KEYWORDS.get(system, tuple(system.split()))
        if any(any(keyword in branch for keyword in keywords) for branch in branch_names):
            hits += 1
    return min(hits / max(len(family.systems), 1), 1.0)


def family_evidence_score(
    disease_id: str,
    family: FamilyEvidence,
    kg_index: KGIndex,
    hpo: Any,
    hpo_normalizer: Any | None = None,
) -> Dict[str, Any]:
    """Return a disease-specific family evidence score in [0, 1]."""
    if not (family.terms or family.diseases or family.genes or family.systems):
        return {
            "family_evidence_score": 0.0,
            "pedigree_mode_support": 0.0,
            "family_gene_support": 0.0,
            "family_disease_support": 0.0,
            "family_phenotype_support": 0.0,
            "family_system_support": 0.0,
            "family_evidence_terms": [],
        }

    _, pedigree_mode_support = inheritance_compatibility_flag(
        family.inheritance_prior,
        kg_index.disease_inheritance.get(disease_id, []),
    )
    gene_support = _family_gene_support(family, disease_id, kg_index)
    disease_support = _family_disease_support(family, disease_id, kg_index)
    phenotype_support = _family_phenotype_support(family, disease_id, kg_index, hpo, hpo_normalizer)
    system_support = _family_system_support(family, disease_id, kg_index, hpo)

    related_signal = max(gene_support, disease_support, phenotype_support, system_support)
    raw = (
        FAMILY_GENE_WEIGHT * gene_support
        + FAMILY_DISEASE_WEIGHT * disease_support
        + FAMILY_PHENOTYPE_SIM_WEIGHT * phenotype_support
        + FAMILY_SYSTEM_WEIGHT * system_support
        + PEDIGREE_MODE_CONTEXT_WEIGHT * pedigree_mode_support * related_signal
    )

    return {
        "family_evidence_score": min(raw, 1.0),
        "pedigree_mode_support": pedigree_mode_support,
        "family_gene_support": gene_support,
        "family_disease_support": disease_support,
        "family_phenotype_support": phenotype_support,
        "family_system_support": system_support,
        "family_evidence_terms": family.terms[:12],
    }
