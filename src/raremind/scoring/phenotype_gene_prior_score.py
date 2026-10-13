"""KG-derived phenotype-gene-disease prior.

This is not a patient genotype signal. It derives a weak phenotype-only prior
from the KG topology:

    patient HPO -> diseases annotated with that HPO -> genes of those diseases
    candidate disease genes -> overlap with the inferred gene prior

The intent is to recover some of the precision that tools such as Phen2Gene can
obtain from HPO-gene statistics, while avoiding an uncontrolled expansion from a
common phenotype to every gene seen in the graph.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Set

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology


def build_patient_phenotype_gene_prior(
    present_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    hpo: HpoOntology,
    max_host_diseases_per_hpo: int = 250,
) -> Dict[str, float]:
    """Infer a bounded patient HPO -> gene prior from existing KG edges."""
    gene_scores: Dict[str, float] = defaultdict(float)
    seen_hpos: Set[str] = set()

    for item in present_hpos:
        phid = item.get("hpo_id")
        if not phid:
            continue
        phid = str(phid)
        if phid in seen_hpos:
            continue
        seen_hpos.add(phid)

        host_diseases = set(kg_index.hpo_to_diseases.get(phid) or [])
        if not host_diseases:
            host_diseases = set(kg_index.hpo_to_diseases_all.get(phid) or [])
        if not host_diseases or len(host_diseases) > max_host_diseases_per_hpo:
            continue

        hpo_weight = hpo.get_ic(phid) / math.sqrt(max(1, len(host_diseases)))
        for did in host_diseases:
            genes = kg_index.disease_genes.get(did) or set()
            if not genes:
                continue
            disease_specificity = 1.0 / math.sqrt(max(1, len(genes)))
            for gene in genes:
                gene_host_count = len(kg_index.gene_to_diseases.get(gene, set()))
                gene_specificity = 1.0 / math.sqrt(max(1, gene_host_count))
                gene_scores[gene] += hpo_weight * disease_specificity * gene_specificity

    return dict(gene_scores)


def phenotype_gene_prior_score(
    disease_id: str,
    patient_gene_prior: Mapping[str, float],
    kg_index: KGIndex,
) -> Dict[str, Any]:
    """Score a candidate by overlap with the inferred phenotype-gene prior."""
    if not patient_gene_prior:
        return {"phenotype_gene_prior_score": 0.0, "phenotype_gene_prior_matches": []}

    genes = sorted(kg_index.disease_genes.get(disease_id) or [])
    if not genes:
        return {"phenotype_gene_prior_score": 0.0, "phenotype_gene_prior_matches": []}

    matches = [(gene, float(patient_gene_prior.get(gene, 0.0))) for gene in genes]
    matches = [(gene, score) for gene, score in matches if score > 0]
    if not matches:
        return {"phenotype_gene_prior_score": 0.0, "phenotype_gene_prior_matches": []}

    raw = sum(score for _, score in matches)
    score = raw / math.sqrt(max(1, len(genes)))
    matches.sort(key=lambda x: x[1], reverse=True)

    return {
        "phenotype_gene_prior_score": float(score),
        "phenotype_gene_prior_matches": [
            {"gene": gene, "score": float(value)} for gene, value in matches[:10]
        ],
    }
