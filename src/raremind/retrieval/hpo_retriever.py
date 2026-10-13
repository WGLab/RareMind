"""HPO-based candidate retrieval.

Uses the precomputed hpo_to_diseases / hpo_to_diseases_all index. For each
patient HPO, retrieves candidate diseases that annotate it (possibly through
IC-gated expansion).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set

from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_hierarchy import ic_gated_expand
from raremind.normalize.hpo_ontology import HpoOntology


def retrieve_by_hpo(
    patient_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    hpo: HpoOntology,
    expansion_mode: str = "ic_gated",
    max_depth: int = 1,
    low_ic_threshold: float = 3.5,
    medium_ic_threshold: float = 5.0,
) -> Dict[str, Dict[str, Any]]:
    """Return candidate_id → retrieval info (matched hpos, credit sum)."""
    candidates: Dict[str, Dict[str, Any]] = {}
    for p in patient_hpos:
        hid = p.get("hpo_id")
        if not hid:
            continue
        expansions = ic_gated_expand(
            hid,
            hpo,
            mode=expansion_mode,
            max_depth=max_depth,
            low_ic_threshold=low_ic_threshold,
            medium_ic_threshold=medium_ic_threshold,
        )
        for exp in expansions:
            diseases = kg_index.hpo_to_diseases_all.get(exp.hpo_id, set())
            for did in diseases:
                info = candidates.setdefault(did, {"matched_hpos": [], "credit": 0.0})
                info["matched_hpos"].append({
                    "patient_hpo": hid,
                    "matched_hpo": exp.hpo_id,
                    "relation": exp.relation,
                    "credit": exp.credit,
                })
                info["credit"] += exp.credit
    return candidates
