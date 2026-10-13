"""Stage 9: LLM clinical synthesis with deterministic ranking provenance."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .scorecard_synthesis import run_synthesis

logger = logging.getLogger(__name__)


def _safe_int(value: Any, default: int) -> int:
    """Convert dataframe scalars without failing on NaN/NA/string blanks."""
    try:
        return int(float(value)) if pd.notna(value) and str(value).strip() else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert dataframe scalars without emitting non-JSON NaN values."""
    try:
        converted = float(value)
        return converted if pd.notna(converted) else default
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "") -> str:
    """Return a usable scalar label without leaking dataframe null markers."""
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        return default
    text = str(value).strip()
    return text if text and text.casefold() not in {"nan", "none", "<na>"} else default


def _state_value(state: Any, key: str, default: Any = None) -> Any:
    return state.get(key, default) if isinstance(state, dict) else getattr(state, key, default)


def _group_key(value: Any) -> str:
    return " ".join(_safe_text(value).casefold().split())


def collapse_reconciled_groups(subtype_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the complete Stage 8 subtype ranking by group name and reindex.

    The highest reconciled subtype fixes each group's position. This allows a
    group represented by several nearby subtypes to appear once without letting
    a low-ranked singleton from the separate Stage 7 group track jump into 9A.
    """
    if subtype_df is None or subtype_df.empty:
        return pd.DataFrame()
    rank_col = next(
        (name for name in ("reconciled_rank", "final_rank", "reranked_rank_subtype", "rank") if name in subtype_df),
        None,
    )
    if rank_col is None:
        raise ValueError("Stage 9A requires a ranked Stage 8 subtype table")
    source = subtype_df.sort_values(rank_col, kind="stable").copy()
    source["_group_label"] = source.apply(
        lambda row: _safe_text(row.get("group_name"))
        or _safe_text(row.get("disease_name"))
        or _safe_text(row.get("group_id"))
        or _safe_text(row.get("disease_id")),
        axis=1,
    )
    source["_group_key"] = source["_group_label"].map(_group_key)
    source.loc[source["_group_key"].eq(""), "_group_key"] = source["disease_id"].map(_group_key)
    score_col = next(
        (name for name in ("reconciled_score", "final_score", "final_score_subtype", "adjusted_score", "total_score") if name in source),
        None,
    )
    rows: list[dict[str, Any]] = []
    for group_rank, (_, members) in enumerate(source.groupby("_group_key", sort=False), start=1):
        leader = members.iloc[0]
        group_name = _safe_text(leader.get("_group_label"))
        group_id = _safe_text(leader.get("group_id")) or _safe_text(leader.get("disease_id"))
        best_subtype_rank = _safe_int(leader.get(rank_col), group_rank)
        rows.append({
            "disease_id": group_id,
            "disease_name": group_name,
            "group_id": group_id,
            "group_name": group_name,
            "group_rank": group_rank,
            "rank": group_rank,
            "adjusted_rank": group_rank,
            "reconciled_rank_group": group_rank,
            "final_score_group": _safe_float(leader.get(score_col, 0.0)) if score_col else 0.0,
            "n_members": len(members),
            "best_subtype_id": _safe_text(leader.get("disease_id")),
            "best_subtype_name": _safe_text(leader.get("disease_name")),
            "best_subtype_reconciled_rank": best_subtype_rank,
            "source_rank_column": rank_col,
        })
    return pd.DataFrame(rows)


def _build_structured_scorecard(
    patient_state: Any,
    final_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    audit_by_id = {a["disease_id"]: a for a in audit_results}

    top = final_df.head(top_k).copy()
    top_cards: List[Dict[str, Any]] = []

    for _, row in top.iterrows():
        did = row["disease_id"]
        audit = audit_by_id.get(did, {})

        composite_rank = int(row.get("rank", 0))
        adjusted_rank = int(row.get("adjusted_rank", composite_rank))
        reranked_rank = int(row.get("reranked_rank_subtype", adjusted_rank))
        reconciled_rank = int(row.get("reconciled_rank", reranked_rank))
        final_rank = int(row.get("final_rank", reconciled_rank))
        final_score = float(row.get("final_score", row.get("reconciled_score", 0.0)))

        total_change = composite_rank - final_rank if composite_rank and final_rank else 0

        # Rank change driver attribution
        drivers = []
        if audit.get("source") == "frontier_underranked":
            drivers.append(f"Frontier consultation flagged as underranked (lens {audit.get('frontier_lens','')})")
        elif audit.get("source") == "frontier_overranked":
            drivers.append(f"Frontier consultation flagged as overranked (lens {audit.get('frontier_lens','')})")
        if audit.get("plausibility") == "strong" and audit.get("source") == "llm_audit":
            drivers.append("Local audit rated strong")
        if audit.get("plausibility") == "weak" and audit.get("source") == "llm_audit":
            drivers.append("Local audit rated weak")
        if float(row.get("llm_validation_score", 0.0)) > 0:
            drivers.append("Grounded LLM validation found phenotype support")
        if float(row.get("llm_validation_score", 0.0)) < 0:
            drivers.append("Grounded LLM validation found phenotype contradiction")
        if float(row.get("family_evidence_score", 0.0)) > 0:
            drivers.append("Family history supports candidate-specific phenotype/gene/disease context")

        top_cards.append({
            "disease_id": did,
            "disease_name": row["disease_name"],
            "group_id": row.get("group_id", ""),
            "group_name": row.get("group_name", ""),
            "composite_rank": composite_rank,
            "adjusted_rank": adjusted_rank,
            "reranked_rank": reranked_rank,
            "reconciled_rank": reconciled_rank,
            "final_rank": final_rank,
            "final_score": final_score,
            "final_fusion_original_weight": float(row.get("final_fusion_original_weight", 0.0)),
            "final_fusion_reconciled_weight": float(row.get("final_fusion_reconciled_weight", 0.0)),
            "total_rank_change": total_change,
            "audit_plausibility": audit.get("plausibility", "not_audited"),
            "supporting_evidence": audit.get("supporting_evidence", []),
            "contradicting_evidence": audit.get("contradicting_evidence", []),
            "missing_expected_evidence": audit.get("missing_expected_evidence", []),
            "key_distinguishing_note": audit.get("key_distinguishing_note", ""),
            "llm_validation_score": float(row.get("llm_validation_score", 0.0)),
            "family_evidence_score": float(row.get("family_evidence_score", 0.0)),
            "pedigree_mode_support": float(row.get("raw_pedigree_mode_score", 0.0)),
            "family_phenotype_support": float(row.get("family_phenotype_support", 0.0)),
            "family_gene_support": float(row.get("family_gene_support", 0.0)),
            "family_disease_support": float(row.get("family_disease_support", 0.0)),
            "family_system_support": float(row.get("family_system_support", 0.0)),
            "frontier_reasoning": audit.get("frontier_reasoning", ""),
            "frontier_lens": audit.get("frontier_lens", ""),
            "rank_change_drivers": drivers,
        })

    # Suggested workup from high-importance missing evidence in top-5
    workup: List[str] = []
    for card in top_cards[:5]:
        for m in card["missing_expected_evidence"]:
            if m.get("importance") == "high":
                workup.append(
                    f"{card['disease_name']}: check {m.get('cue','')}"
                )

    scorecard = {
        "case_signal": _state_value(patient_state, "case_signal", {}) or {},
        "top_candidates": top_cards,
        "suggested_workup": list(dict.fromkeys(workup)),  # dedupe preserving order
        "reconciled": {
            "top_group": reconciled.get("top_group"),
            "top_subtype": reconciled.get("top_subtype"),
            "final_top_subtype": reconciled.get("final_top_subtype"),
            "final_ranking": reconciled.get("final_ranking"),
            "disagreement": reconciled.get("disagreement", False),
            "tiebreaker": reconciled.get("tiebreaker"),
            "method": reconciled.get("method", "n/a"),
        },
        "frontier": {
            "triggered": frontier_output.get("triggered", False) if frontier_output else False,
            "trigger_reason": frontier_output.get("trigger_reason", "") if frontier_output else "",
            "underranked": frontier_output.get("underranked", []) if frontier_output else [],
            "overranked": frontier_output.get("overranked", []) if frontier_output else [],
        },
    }
    return scorecard


def build_scorecard(
    patient_state: Any,
    final_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
    top_k: int = 10,
    *,
    kg: Dict[str, Any],
    llm: Any = None,
    pairwise_subtype: List[Dict[str, Any]] | None = None,
    synthesis_config: Any = None,
) -> Dict[str, Any]:
    """Preserve the fixed ranking and legacy JSON, then synthesize Stage 9.

    ``llm`` is the already-loaded pipeline VllmClient. Only Stage 9 sampling
    parameters are overridden; its model and all upstream state stay unchanged.
    An explicitly configured OpenAI-compatible vLLM endpoint is also supported.
    """
    scorecard = _build_structured_scorecard(
        patient_state, final_df, audit_results, reconciled, frontier_output, top_k
    )
    synthesis = run_synthesis(
        patient_state=patient_state,
        top_cards=scorecard["top_candidates"],
        final_df=final_df,
        audit_results=audit_results,
        reconciled=scorecard["reconciled"],
        kg=kg,
        llm=llm,
        pairwise_subtype=pairwise_subtype,
        config=synthesis_config,
        report_level="subtype",
    )
    scorecard.update(synthesis)
    for card, assessment in zip(scorecard["top_candidates"], synthesis["clinical_assessments"]):
        card["clinical_assessment"] = assessment["text"]
        card["overall_assessment"] = assessment["overall_assessment"]
    return scorecard


def build_group_scorecard(
    patient_state: Any,
    group_df: pd.DataFrame,
    subtype_final_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
    top_k: int = 10,
    *,
    kg: Dict[str, Any],
    llm: Any = None,
    pairwise_group: List[Dict[str, Any]] | None = None,
    pairwise_subtype: List[Dict[str, Any]] | None = None,
    synthesis_config: Any = None,
) -> Dict[str, Any]:
    """Build Stage 9A from Stage 8 subtypes collapsed by group name."""
    rank_col = next(
        (name for name in ("group_rank", "reconciled_rank_group", "rank") if name in group_df),
        None,
    )
    if rank_col is None:
        raise ValueError("Stage 9A group ranking has no rank column")
    score_col = next(
        (name for name in ("final_score_group", "reconciled_score", "final_score", "total_score") if name in group_df),
        None,
    )
    groups = group_df.sort_values(rank_col, kind="stable").copy()
    identity_col = "disease_id" if "disease_id" in groups else "group_id"
    groups["_stage9_group_key"] = groups[identity_col].map(_safe_text)
    groups = groups.drop_duplicates("_stage9_group_key", keep="first").head(top_k)
    cards: List[Dict[str, Any]] = []
    audit_by_id = {str(item.get("disease_id")): item for item in audit_results}
    subtype_pairwise = pairwise_subtype or []
    subtype_rows = subtype_final_df.copy()
    subtype_rows["_stage9_group_key"] = subtype_rows.apply(
        lambda member: _group_key(member.get("group_name"))
        or _group_key(member.get("group_id"))
        or _group_key(member.get("disease_name")),
        axis=1,
    )
    subtype_rows["_stage9_group_id"] = subtype_rows.get(
        "group_id", pd.Series(index=subtype_rows.index, dtype=object)
    ).map(_safe_text)
    for position, (_, row) in enumerate(groups.iterrows(), start=1):
        gid = _safe_text(row.get("disease_id")) or _safe_text(row.get("group_id"))
        group_name = _safe_text(row.get("group_name")) or _safe_text(row.get("disease_name")) or gid
        members = subtype_rows[
            subtype_rows["_stage9_group_key"].eq(_group_key(group_name))
            | subtype_rows["_stage9_group_id"].eq(gid)
        ].copy()
        member_rank = next(
            (name for name in ("reconciled_rank", "final_rank", "reranked_rank_subtype", "rank") if name in members),
            None,
        )
        if member_rank:
            members = members.sort_values(member_rank, kind="stable")
        representatives = []
        for member in members.head(5).to_dict("records"):
            subtype_id = _safe_text(member.get("disease_id"))
            subtype_comparisons = [
                comparison for comparison in subtype_pairwise
                if subtype_id in {
                    str(comparison.get("disease_a_id", "")),
                    str(comparison.get("disease_b_id", "")),
                }
            ]
            representatives.append({
                "disease_id": subtype_id,
                "disease_name": _safe_text(member.get("disease_name")),
                "subtype_rank": _safe_int(member.get(member_rank), 0) if member_rank and pd.notna(member.get(member_rank)) else None,
                "composite_rank": _safe_int(member.get("rank"), 0),
                "audit_adjusted_rank": _safe_int(member.get("adjusted_rank"), 0),
                "pairwise_rank": _safe_int(member.get("reranked_rank_subtype"), 0),
                "score": _safe_float(member.get("reconciled_score", member.get("final_score", member.get("total_score", 0.0)))),
                "audit_analysis": audit_by_id.get(subtype_id, {}),
                "pairwise_analysis": subtype_comparisons,
            })
        final_score = _safe_float(row.get(score_col, 0.0)) if score_col else 0.0
        composite_rank = _safe_int(row.get("rank"), position)
        cards.append({
            "disease_id": gid,
            "disease_name": group_name,
            "group_id": gid,
            "group_name": group_name,
            "composite_rank": composite_rank,
            "adjusted_rank": _safe_int(row.get("adjusted_rank"), position),
            "reranked_rank": _safe_int(row.get(rank_col), position),
            "reconciled_rank": _safe_int(row.get(rank_col), position),
            "final_rank": position,
            "final_score": final_score,
            "total_rank_change": composite_rank - position,
            "audit_plausibility": str(row.get("audit_plausibility", "not_audited")),
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "missing_expected_evidence": [],
            "key_distinguishing_note": "",
            "family_evidence_score": _safe_float(row.get("family_evidence_score", 0.0)),
            "representative_subtypes": representatives,
            "member_count": _safe_int(row.get("n_members"), len(members)),
            "rank_change_drivers": [],
        })
    scorecard = {
        "report_level": "group",
        "case_signal": _state_value(patient_state, "case_signal", {}) or {},
        "top_candidates": cards,
        "suggested_workup": [],
        "reconciled": {
            "top_group": reconciled.get("top_group"),
            "top_subtype": reconciled.get("top_subtype"),
            "disagreement": reconciled.get("disagreement", False),
            "method": "stage8_reconciled_group_name_collapse",
        },
        "frontier": {
            "triggered": frontier_output.get("triggered", False) if frontier_output else False,
            "trigger_reason": frontier_output.get("trigger_reason", "") if frontier_output else "",
        },
    }
    synthesis = run_synthesis(
        patient_state=patient_state,
        top_cards=cards,
        final_df=group_df,
        audit_results=audit_results,
        reconciled=scorecard["reconciled"],
        kg=kg,
        llm=llm,
        pairwise_subtype=pairwise_group,
        config=synthesis_config,
        report_level="group",
    )
    scorecard.update(synthesis)
    for card, assessment in zip(cards, synthesis["clinical_assessments"]):
        card["clinical_assessment"] = assessment["text"]
        card["overall_assessment"] = assessment["overall_assessment"]
    return scorecard


def format_scorecard_text(scorecard: Dict[str, Any]) -> str:
    """Render the clinical synthesis first; still support historical JSON files."""
    if scorecard.get("synthesis_report"):
        return scorecard["synthesis_report"]
    return _format_legacy_scorecard_text(scorecard)


def _format_legacy_scorecard_text(scorecard: Dict[str, Any]) -> str:
    """Human-readable version."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("RareMind CLINICAL SCORECARD")
    lines.append("=" * 70)
    lines.append("")

    rec = scorecard.get("reconciled", {})
    if rec.get("final_top_subtype"):
        ts = rec["final_top_subtype"]
        final_info = rec.get("final_ranking", {}) or {}
        method = final_info.get("method", "final")
        lines.append(f"TOP CANDIDATE ({method}): {ts.get('disease_name')} ({ts.get('disease_id')})")
    elif rec.get("top_subtype"):
        ts = rec["top_subtype"]
        lines.append(f"TOP CANDIDATE (reconciled): {ts.get('disease_name')} ({ts.get('disease_id')})")
    if rec.get("top_group"):
        tg = rec["top_group"]
        lines.append(f"TOP GROUP: {tg.get('disease_name')} ({tg.get('disease_id')})")
    if rec.get("disagreement"):
        lines.append(f"  ⚠ Group / subtype disagreement. Tiebreaker: {rec.get('tiebreaker')}")
    lines.append("")

    fr = scorecard.get("frontier", {})
    if fr.get("triggered"):
        lines.append(f"FRONTIER CONSULTATION: triggered ({fr.get('trigger_reason','')})")
        if fr.get("underranked"):
            lines.append(f"  Underranked: {[u['disease_name'] for u in fr['underranked']]}")
        if fr.get("overranked"):
            lines.append(f"  Overranked:  {[u['disease_name'] for u in fr['overranked']]}")
    else:
        lines.append("FRONTIER CONSULTATION: not triggered")
    lines.append("")

    lines.append("TOP CANDIDATES")
    lines.append("-" * 70)
    for i, c in enumerate(scorecard.get("top_candidates", []), 1):
        lines.append(f"{i}. {c['disease_name']} ({c['disease_id']})")
        lines.append(
            f"   ranks: composite={c['composite_rank']} → audit={c['adjusted_rank']} "
            f"→ pairwise={c['reranked_rank']} → reconciled={c['reconciled_rank']} "
            f"→ final={c['final_rank']} "
            f"(change: {c['total_rank_change']:+d})"
        )
        if c.get("final_score"):
            lines.append(f"   final score: {c['final_score']:.3f}")
        lines.append(f"   audit: {c['audit_plausibility']}")
        if c.get("llm_validation_score"):
            lines.append(f"   llm validation score: {c['llm_validation_score']:+.2f}")
        if c.get("family_evidence_score"):
            lines.append(f"   family evidence score: {c['family_evidence_score']:.2f}")
        if c.get("frontier_reasoning"):
            lines.append(f"   frontier ({c.get('frontier_lens','')}): {c['frontier_reasoning']}")
        if c.get("key_distinguishing_note"):
            lines.append(f"   note: {c['key_distinguishing_note']}")
        if c["supporting_evidence"]:
            lines.append("   ✓ Supporting:")
            for s in c["supporting_evidence"][:4]:
                lines.append(f"     - {s.get('cue','')}")
        if c["contradicting_evidence"]:
            lines.append("   ✗ Contradicting:")
            for s in c["contradicting_evidence"][:3]:
                lines.append(f"     - {s.get('cue','')}")
        if c["missing_expected_evidence"]:
            highs = [m for m in c["missing_expected_evidence"] if m.get("importance") == "high"]
            if highs:
                lines.append("   ? Missing (high importance):")
                for m in highs[:3]:
                    lines.append(f"     - {m.get('cue','')}")
        if c.get("rank_change_drivers"):
            lines.append("   rank change drivers:")
            for d in c["rank_change_drivers"]:
                lines.append(f"     - {d}")
        lines.append("")

    if scorecard.get("suggested_workup"):
        lines.append("SUGGESTED ADDITIONAL WORKUP")
        lines.append("-" * 70)
        for w in scorecard["suggested_workup"]:
            lines.append(f"  - {w}")
        lines.append("")

    return "\n".join(lines)


def build_rank_trajectory(
    ranked_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reranked_df: pd.DataFrame,
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
) -> pd.DataFrame:
    """Build the per-candidate rank trajectory table for investigation."""
    df = ranked_df.copy()
    if "group_id" not in df.columns:
        df["group_id"] = df.get("mondo_group_id", "")
    if "group_name" not in df.columns:
        df["group_name"] = ""

    audit_by_id = {a["disease_id"]: a for a in audit_results}

    # Frontier flags
    frontier_flag: Dict[str, str] = {}
    frontier_reason: Dict[str, str] = {}
    for u in (frontier_output.get("underranked", []) if frontier_output else []):
        frontier_flag[u.get("disease_id", "")] = "underranked"
        frontier_reason[u.get("disease_id", "")] = u.get("reasoning", "")
    for o in (frontier_output.get("overranked", []) if frontier_output else []):
        frontier_flag[o.get("disease_id", "")] = "overranked"
        frontier_reason[o.get("disease_id", "")] = o.get("reasoning", "")

    df["frontier_flag"] = df["disease_id"].map(lambda d: frontier_flag.get(d, "none"))
    df["frontier_reasoning"] = df["disease_id"].map(lambda d: frontier_reason.get(d, ""))

    # Audit
    df["audit_plausibility"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("plausibility", "not_audited"))
    df["audit_multiplier"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("multiplier", 1.0))
    df["llm_validation_raw_score"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("validation_raw_score", 0.0))
    df["llm_validation_source"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("validation_source", ""))

    # Merge reranked positions
    rerank_cols = [c for c in reranked_df.columns if c.startswith("reranked_") or c.startswith("final_")]
    merged = df.merge(
        reranked_df[["disease_id"] + rerank_cols].drop_duplicates("disease_id"),
        on="disease_id", how="left"
    )

    # Reconciled rank
    rec_df = reconciled.get("reconciled_df")
    if isinstance(rec_df, pd.DataFrame) and "reconciled_rank" in rec_df.columns:
        merged = merged.merge(
            rec_df[["disease_id", "reconciled_rank", "reconciled_score", "alpha"]].drop_duplicates("disease_id"),
            on="disease_id", how="left",
        )

    # Rank change
    if "reconciled_rank" in merged.columns:
        merged["rank_change_total"] = merged["rank"] - merged["reconciled_rank"]

    # Final calibrated rank
    final_df = reconciled.get("final_df")
    if isinstance(final_df, pd.DataFrame) and "final_rank" in final_df.columns:
        final_cols = [
            "disease_id", "final_rank", "final_score",
            "final_original_score_norm", "final_reconciled_score_norm",
            "final_fusion_original_weight", "final_fusion_reconciled_weight",
            "final_fusion_method",
        ]
        final_cols = [c for c in final_cols if c in final_df.columns]
        merged = merged.merge(
            final_df[final_cols].drop_duplicates("disease_id"),
            on="disease_id", how="left",
        )
        merged["rank_change_total"] = merged["rank"] - merged["final_rank"]

    return merged
