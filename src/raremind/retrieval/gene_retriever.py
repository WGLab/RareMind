"""Gene-based candidate retrieval."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from raremind.kg.kg_precompute import KGIndex


def _norm_result(value: Any) -> str:
    return str(value or "").strip().lower()


def _include_gene_for_retrieval(entry: Dict[str, Any], evidence_policy: str) -> bool:
    if evidence_policy == "negative_only":
        return "neg" in _norm_result(entry.get("result"))
    return True


def retrieve_by_gene(
    gene_mentions: List[Dict[str, Any]],
    vcf_summary: List[Dict[str, Any]],
    kg_index: KGIndex,
    evidence_policy: str = "negative_only",
) -> Dict[str, Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}

    genes: Set[str] = set()
    for g in gene_mentions or []:
        if not _include_gene_for_retrieval(g, evidence_policy):
            continue
        gn = g.get("gene")
        if gn:
            genes.add(gn.upper())
    for g in vcf_summary or []:
        if not _include_gene_for_retrieval(g, evidence_policy):
            continue
        gn = g.get("gene")
        if gn:
            genes.add(gn.upper())

    for g in genes:
        diseases = kg_index.gene_to_diseases.get(g, set())
        for did in diseases:
            info = candidates.setdefault(did, {"matched_genes": []})
            info["matched_genes"].append(g)

    return candidates
