"""Stage 8: Group-subtype reconciliation.

Combines subtype-track and group-track rankings with evidence-aware α blending:
  posterior(d) = subtype_score(d) × group_score(group_of(d))^α

α selection per candidate:
  - Strong gene evidence for the specific disease → α = 0  (trust subtype)
  - Group hallmarks match but no subtype distinguishers → α = 1  (trust group)
  - Otherwise → α = 0.5 (balanced)

If top subtype is NOT a member of top group → optional LLM tiebreaker call.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from raremind.core.json_utils import safe_json_load
from raremind.core.utils import read_prompt
from raremind.core.compat import to_dict
from raremind.kg.kg_precompute import KGIndex
from .audit import compact_patient_evidence

logger = logging.getLogger(__name__)


def _alpha_for(
    disease_id: str,
    patient_state: Any,
    kg_index: KGIndex,
    cfg: Any,
) -> float:
    """Decide α per candidate based on evidence."""
    # Gene evidence for this disease?
    gene_set = kg_index.disease_genes.get(disease_id, set())
    patient_genes = set()
    for g in patient_state.gene_mentions or []:
        if g.get("gene"):
            patient_genes.add(g["gene"].upper())
    for g in patient_state.vcf_summary or []:
        if g.get("gene"):
            patient_genes.add(g["gene"].upper())

    gene_overlap = bool(gene_set & patient_genes)
    if gene_overlap:
        return cfg.reconciliation.alpha_gene_strong  # 0.0

    # Broad group-level signal: if disease's group has many hallmarks matching the patient
    group_id = kg_index.disease_group.get(disease_id)
    if group_id and group_id in kg_index.disease_hallmarks:
        group_hallmarks = set(kg_index.disease_hallmarks.get(group_id, []))
        all_phens = [to_dict(h) for h in patient_state.normalized_hpo]
        patient_hpos = {h.get("hpo_id") for h in all_phens if h.get("present", True)}
        overlap = len(group_hallmarks & patient_hpos)
        if overlap >= 3:
            return cfg.reconciliation.alpha_group_hallmarks  # 1.0

    return cfg.reconciliation.alpha_default  # 0.5


def reconcile(
    subtype_df: pd.DataFrame,
    group_df: pd.DataFrame,
    patient_state: Any,
    kg_index: KGIndex,
    cfg: Any,
    llm: Any | None = None,
    prompt_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the reconciled top output."""
    if not cfg.reconciliation.enabled:
        top_sub = subtype_df.iloc[0] if len(subtype_df) > 0 else None
        return {
            "top_subtype": top_sub.to_dict() if top_sub is not None else None,
            "top_group": None,
            "method": "disabled",
        }

    if len(subtype_df) == 0 or len(group_df) == 0:
        top_sub = subtype_df.iloc[0] if len(subtype_df) > 0 else None
        top_grp = group_df.iloc[0] if len(group_df) > 0 else None
        return {
            "top_subtype": top_sub.to_dict() if top_sub is not None else None,
            "top_group": top_grp.to_dict() if top_grp is not None else None,
            "method": "trivial",
        }

    score_col_s = "final_score_subtype" if "final_score_subtype" in subtype_df.columns else "total_score"
    score_col_g = "final_score_group" if "final_score_group" in group_df.columns else "total_score"

    # Normalize
    s_scores = subtype_df[score_col_s].astype(float).values
    g_scores = group_df[score_col_g].astype(float).values
    s_scores = s_scores / s_scores.max() if s_scores.max() > 0 else s_scores
    g_scores = g_scores / g_scores.max() if g_scores.max() > 0 else g_scores

    # Per-group map
    group_score_by_id = {gid: float(score) for gid, score in zip(group_df["disease_id"].values, g_scores)}

    # Compute posterior per subtype
    posteriors = []
    alphas = []
    for i, row in enumerate(subtype_df.itertuples(index=False)):
        sid = row.disease_id
        grp_id = (
            getattr(row, "group_id", "")
            or getattr(row, "mondo_group_id", "")
            or kg_index.disease_group.get(sid, "")
            or sid
        )
        alpha = _alpha_for(sid, patient_state, kg_index, cfg)
        alphas.append(alpha)
        g_contrib = group_score_by_id.get(grp_id, 0.0) ** alpha if grp_id else 1.0
        s = s_scores[i] if i < len(s_scores) else 0.0
        posteriors.append(s * g_contrib)

    out = subtype_df.copy()
    out["alpha"] = alphas
    out["reconciled_score"] = posteriors
    out = out.sort_values("reconciled_score", ascending=False).reset_index(drop=True)
    out["reconciled_rank"] = out.index + 1

    top_sub = out.iloc[0]
    top_grp = group_df.iloc[0]

    # Disagreement check
    disagreement = False
    top_sub_group = top_sub.get("group_id") or top_sub.get("mondo_group_id") or kg_index.disease_group.get(top_sub["disease_id"], "")
    if top_sub_group != top_grp["disease_id"]:
        disagreement = True

    tiebreaker = None
    if disagreement and cfg.reconciliation.tiebreaker_on_disagreement and llm is not None and prompt_dir:
        tiebreaker = _run_group_tiebreaker(
            patient_state, top_sub, top_grp, kg_index, llm, prompt_dir
        )

    return {
        "top_subtype": top_sub.to_dict(),
        "top_group": top_grp.to_dict(),
        "reconciled_df": out,
        "disagreement": disagreement,
        "tiebreaker": tiebreaker,
        "method": "alpha_blending",
    }


def _run_group_tiebreaker(
    patient_state: Any,
    top_subtype_row: pd.Series,
    top_group_row: pd.Series,
    kg_index: KGIndex,
    llm: Any,
    prompt_dir: str,
) -> Dict[str, Any]:
    path = Path(prompt_dir)
    system = read_prompt(path / "reasoning" / "group_reconcile_system.txt")
    user_tpl = read_prompt(path / "reasoning" / "group_reconcile_user.txt")

    subtype_id = top_subtype_row["disease_id"]
    subtype_name = top_subtype_row["disease_name"]
    group_id = top_group_row["disease_id"]
    group_name = top_group_row["disease_name"]

    subtype_group_id = (
        top_subtype_row.get("group_id")
        or top_subtype_row.get("mondo_group_id")
        or kg_index.disease_group.get(subtype_id, "")
    )
    subtype_group_name = kg_index.disease_name.get(subtype_group_id, subtype_group_id)

    group_hallmarks = kg_index.disease_hallmark_names.get(group_id, [])[:10]
    subtype_card = kg_index.disease_cards.get(subtype_id, "")

    user = user_tpl.format(
        patient_evidence=compact_patient_evidence(patient_state),
        group_name=group_name, group_id=group_id,
        group_hallmarks="\n".join([f"  - {x}" for x in group_hallmarks]) or "  (none)",
        group_members="(see group definition)",
        subtype_name=subtype_name, subtype_id=subtype_id,
        subtype_group_name=subtype_group_name, subtype_group_id=subtype_group_id,
        subtype_card=subtype_card,
    )

    raw = llm.chat(system, user, task="extraction")
    import json as _json
    raw_str = _json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
    data = safe_json_load(raw_str, prefer="object")
    if not isinstance(data, dict):
        data = {"winner": "subtype", "reasoning": "parse_failed"}
    return data
