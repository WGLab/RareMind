"""Stage 6: Adaptive pairwise adjudication.

Adaptive prompt selection:
  - curated DDX rule available: differential-context prompt
  - otherwise high hallmark overlap (> 0.6): fingerprint-focused (discriminator features only)
  - otherwise: card-based prompt

Pair-skipping heuristics:
  - Skip when audit gap is >= 2 levels (strong vs weak, strong vs implausible, moderate vs implausible)
  - Skip when total_score gap > 2x

Audit evidence bullets and frontier reasoning notes are forwarded to the LLM.
"""
from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from raremind.core.json_utils import safe_json_load
from raremind.core.utils import read_prompt
from raremind.core.compat import to_dict
from raremind.kg.kg_precompute import KGIndex
from .audit import compact_patient_evidence

logger = logging.getLogger(__name__)


PLAUSIBILITY_ORDER = {"strong": 3, "moderate": 2, "weak": 1, "implausible": 0}


def _pairwise_cfg(cfg: Any, key: str, default: Any) -> Any:
    pairwise_cfg = getattr(cfg, "pairwise", {})
    if isinstance(pairwise_cfg, dict):
        return pairwise_cfg.get(key, default)
    return getattr(pairwise_cfg, key, default)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def _fmt_features(
    hpos: List[str],
    kg_index: KGIndex,
    kg: Dict[str, Any],
    hpo_ontology: Any,
    limit: int = 8,
) -> str:
    lines = []
    for h in hpos[:limit]:
        lines.append(f"  - {hpo_ontology.get_name(h)} ({h}) [IC={hpo_ontology.get_ic(h):.1f}]")
    return "\n".join(lines) if lines else "  (none)"


def _ddx_rule_between(
    disease_a: Dict[str, Any],
    disease_b_id: str,
    disease_b_name: str,
) -> str:
    """Extract a curated DDX rule from disease_a's KG entry that mentions disease_b."""
    disease_b_name_lc = str(disease_b_name or "").lower().strip()
    disease_b_id_lc = str(disease_b_id or "").lower().strip()

    def _comparison_rule(payload: Dict[str, Any]) -> str:
        comparison = payload.get("comparison", [])
        if not isinstance(comparison, list):
            comparison = []
        bits = []
        for item in comparison:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature", "")).strip()
            description = str(item.get("description", "")).strip()
            if feature and description:
                bits.append(f"{feature}: {description}")
            elif feature:
                bits.append(feature)
            elif description:
                bits.append(description)
        return "; ".join(bits[:6]).strip()

    def _iter_candidates(block: Any):
        if isinstance(block, list):
            for item in block:
                if isinstance(item, dict):
                    target = (
                        item.get("target_name", "")
                        or item.get("disease", "")
                        or item.get("name", "")
                    )
                    rule = (
                        str(item.get("rule", "")).strip()
                        or _comparison_rule(item)
                    )
                    ids = [
                        str(item.get("target_id", "")).strip(),
                        str(item.get("mondo", "")).strip(),
                        str(item.get("omim", "")).strip(),
                    ]
                    yield target, ids, rule
        elif isinstance(block, dict):
            for target, payload in block.items():
                if not isinstance(payload, dict):
                    continue
                details = payload.get("details", []) if isinstance(payload.get("details"), list) else []
                detail_bits = []
                for detail in details:
                    if not isinstance(detail, dict):
                        continue
                    text = (
                        detail.get("differential_disease_state")
                        or detail.get("feature")
                        or (detail.get("evidence", {}) or {}).get("text", "")
                    )
                    text = str(text).strip()
                    if text:
                        detail_bits.append(text)
                rule = (
                    str(payload.get("rule", "")).strip()
                    or _comparison_rule(payload)
                    or "; ".join(detail_bits[:6])
                )
                ids = [
                    str(payload.get("target_id", "")).strip(),
                    str(payload.get("mondo", "")).strip(),
                    str(payload.get("omim", "")).strip(),
                ]
                target_name = payload.get("target_name", "") or payload.get("name", "") or target
                yield target_name, ids, rule

    diff_blocks = [
        disease_a.get("differentials"),
        disease_a.get("differential"),
        (disease_a.get("meta", {}) or {}).get("differentials") if isinstance(disease_a.get("meta"), dict) else None,
        (disease_a.get("meta", {}) or {}).get("differential") if isinstance(disease_a.get("meta"), dict) else None,
    ]

    for block in diff_blocks:
        for target, ids, rule in _iter_candidates(block):
            target_lc = str(target or "").lower().strip()
            id_hits = [x.lower() for x in ids if x]
            if disease_b_name_lc and disease_b_name_lc in target_lc and rule:
                return rule
            if disease_b_id_lc and disease_b_id_lc in target_lc and rule:
                return rule
            if disease_b_id_lc and disease_b_id_lc in id_hits and rule:
                return rule
    return ""


def _build_audit_bullet_block(
    audit_a: Dict[str, Any] | None,
    audit_b: Dict[str, Any] | None,
) -> str:
    if not audit_a and not audit_b:
        return ""
    parts = ["\nAUDIT EVIDENCE"]
    for label, a in [("A", audit_a), ("B", audit_b)]:
        if not a:
            continue
        note = a.get("key_distinguishing_note", "")
        support_count = len(a.get("supporting_evidence", []))
        contra_count = len(a.get("contradicting_evidence", []))
        parts.append(
            f"  {label}: plausibility={a.get('plausibility','?')}; "
            f"support={support_count}; contradict={contra_count}; note={note}"
        )
    return "\n".join(parts) + "\n"


def _build_frontier_note(
    flag_a: Dict[str, Any] | None,
    flag_b: Dict[str, Any] | None,
) -> str:
    notes = []
    for label, f in [("A", flag_a), ("B", flag_b)]:
        if not f:
            continue
        notes.append(
            f"  {label}: frontier flagged as {f.get('flag_type','')} "
            f"(lens {f.get('lens','')}): {f.get('reasoning','')}"
        )
    if not notes:
        return ""
    return "\nFRONTIER NOTES\n" + "\n".join(notes) + "\n"


def build_pairwise_prompts(
    patient_state: Any,
    row_a: pd.Series,
    row_b: pd.Series,
    audit_a: Dict[str, Any] | None,
    audit_b: Dict[str, Any] | None,
    kg_index: KGIndex,
    kg: Dict[str, Any],
    hpo_ontology: Any,
    prompt_dir: Path,
    cfg: Any,
    frontier_flag_a: Dict[str, Any] | None = None,
    frontier_flag_b: Dict[str, Any] | None = None,
) -> Tuple[str, str, str]:
    """Return (system_prompt, user_prompt, prompt_type)."""

    did_a, did_b = row_a["disease_id"], row_b["disease_id"]
    dname_a, dname_b = row_a["disease_name"], row_b["disease_name"]

    hall_a = set(kg_index.disease_hallmarks.get(did_a, []))
    hall_b = set(kg_index.disease_hallmarks.get(did_b, []))
    jaccard = _jaccard(hall_a, hall_b)

    ddx_rule = _ddx_rule_between(kg.get(did_a, {}), did_b, dname_b)
    if not ddx_rule:
        ddx_rule = _ddx_rule_between(kg.get(did_b, {}), did_a, dname_a)

    patient_evidence = compact_patient_evidence(patient_state)
    audit_bullets = _build_audit_bullet_block(audit_a, audit_b) if cfg.pairwise.use_audit_bullets else ""
    frontier_note = _build_frontier_note(frontier_flag_a, frontier_flag_b)

    all_phens = [to_dict(h) for h in patient_state.normalized_hpo]
    patient_hpo_set = {h.get("hpo_id") for h in all_phens if h.get("present", True)}
    patient_negated_set = {h.get("hpo_id") for h in all_phens if not h.get("present", True)}

    # DIFFERENTIAL path: curated disease-vs-disease rules are most specific.
    if ddx_rule:
        system = read_prompt(prompt_dir / "reasoning" / "pairwise_differential_system.txt")
        user_tpl = read_prompt(prompt_dir / "reasoning" / "pairwise_differential_user.txt")
        user = user_tpl.format(
            patient_evidence=patient_evidence,
            disease_a_name=dname_a, disease_a_id=did_a,
            disease_a_card=kg_index.disease_cards.get(did_a, ""),
            disease_b_name=dname_b, disease_b_id=did_b,
            disease_b_card=kg_index.disease_cards.get(did_b, ""),
            ddx_rule=ddx_rule,
            audit_bullets=audit_bullets,
            frontier_note=frontier_note,
        )
        return system, user, "differential"

    # FINGERPRINT path
    if cfg.pairwise.adaptive_prompts and jaccard > cfg.pairwise.overlap_threshold_high:
        discrim_a = list(hall_a - hall_b)
        discrim_b = list(hall_b - hall_a)
        patient_has_a = [h for h in discrim_a if h in patient_hpo_set]
        patient_lacks_a = [h for h in discrim_a if h in patient_negated_set]
        patient_has_b = [h for h in discrim_b if h in patient_hpo_set]
        patient_lacks_b = [h for h in discrim_b if h in patient_negated_set]

        system = read_prompt(prompt_dir / "reasoning" / "pairwise_fingerprint_system.txt")
        user_tpl = read_prompt(prompt_dir / "reasoning" / "pairwise_fingerprint_user.txt")
        user = user_tpl.format(
            patient_evidence=patient_evidence,
            disease_a_name=dname_a, disease_a_id=did_a,
            disease_b_name=dname_b, disease_b_id=did_b,
            discriminators_a=_fmt_features(discrim_a, kg_index, kg, hpo_ontology),
            discriminators_b=_fmt_features(discrim_b, kg_index, kg, hpo_ontology),
            patient_has_a=_fmt_features(patient_has_a, kg_index, kg, hpo_ontology),
            patient_lacks_a=_fmt_features(patient_lacks_a, kg_index, kg, hpo_ontology),
            patient_has_b=_fmt_features(patient_has_b, kg_index, kg, hpo_ontology),
            patient_lacks_b=_fmt_features(patient_lacks_b, kg_index, kg, hpo_ontology),
            audit_bullets=audit_bullets,
            frontier_note=frontier_note,
        )
        return system, user, "fingerprint"

    # CARD fallback
    system = read_prompt(prompt_dir / "reasoning" / "pairwise_kg_system.txt")
    user_tpl = read_prompt(prompt_dir / "reasoning" / "pairwise_kg_user.txt")
    user = user_tpl.format(
        patient_evidence=patient_evidence,
        disease_a_name=dname_a, disease_a_id=did_a,
        disease_a_card=kg_index.disease_cards.get(did_a, ""),
        disease_b_name=dname_b, disease_b_id=did_b,
        disease_b_card=kg_index.disease_cards.get(did_b, ""),
        audit_bullets=audit_bullets,
        frontier_note=frontier_note,
    )
    return system, user, "kg"


def parse_pairwise(raw: str) -> Dict[str, Any]:
    data = safe_json_load(raw, prefer="object")
    if not isinstance(data, dict):
        return {"winner": "tie", "strength": "weak", "reasoning": "parse_failed"}
    w = str(data.get("winner", "tie")).strip()
    if w not in {"A", "B", "tie"}:
        w = "tie"
    s = str(data.get("strength", "weak")).strip().lower()
    if s not in {"strong", "moderate", "weak"}:
        s = "weak"
    return {
        "winner": w,
        "strength": s,
        "reasoning": str(data.get("reasoning", "")),
    }


def should_skip_pair(
    row_a: pd.Series,
    row_b: pd.Series,
    cfg: Any,
) -> bool:
    """Return True if this pair should be skipped (outcome effectively predetermined)."""
    if cfg.pairwise.skip_large_gaps:
        a_cat = row_a.get("audit_plausibility", "moderate")
        b_cat = row_b.get("audit_plausibility", "moderate")
        if a_cat in PLAUSIBILITY_ORDER and b_cat in PLAUSIBILITY_ORDER:
            if abs(PLAUSIBILITY_ORDER[a_cat] - PLAUSIBILITY_ORDER[b_cat]) >= 2:
                return True
    a_score = row_a.get("adjusted_score", row_a.get("total_score", 0.0))
    b_score = row_b.get("adjusted_score", row_b.get("total_score", 0.0))
    gap = cfg.pairwise.skip_score_gap_multiplier
    if a_score > 0 and b_score > 0:
        if max(a_score, b_score) > gap * min(a_score, b_score):
            return True
    return False


def _patient_hpo_ids(patient_state: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in getattr(patient_state, "normalized_hpo", []) or []:
        h = to_dict(item)
        if not h.get("present", True):
            continue
        hid = h.get("hpo_id")
        if hid and hid not in seen:
            seen.add(hid)
            out.append(str(hid))
    return out


def _branch_concentration(patient_hpos: List[str], hpo_ontology: Any) -> float:
    branch_counts: Dict[str, int] = {}
    for hid in patient_hpos:
        branches = hpo_ontology.get_branches(hid)
        if not branches:
            branch_counts["NO_BRANCH"] = branch_counts.get("NO_BRANCH", 0) + 1
        for branch in branches:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
    total = sum(branch_counts.values())
    return float(max(branch_counts.values()) / total) if total else 0.0


def resolve_pairwise_top_n(
    patient_state: Any,
    ranked_df: pd.DataFrame,
    hpo_ontology: Any,
    cfg: Any,
    frontier_flags: Optional[Dict[str, Dict[str, Any]]] = None,
    track: str = "subtype",
) -> int:
    """Return pairwise window size.

    The default floor remains cfg.pairwise.top_n, usually 30. If enabled, the
    window expands up to cfg.pairwise.dynamic_max_top_n, usually 50, only when
    the current case has evidence that the truth could plausibly sit just
    outside Top-30.
    """
    base_n = int(_pairwise_cfg(cfg, "top_n", 30))
    max_n = int(_pairwise_cfg(cfg, "dynamic_max_top_n", 50))
    if len(ranked_df) <= base_n:
        return min(base_n, len(ranked_df))
    if not bool(_pairwise_cfg(cfg, "dynamic_top_n_enabled", False)):
        return min(base_n, len(ranked_df))

    max_n = min(max(base_n, max_n), len(ranked_df))
    patient_hpos = _patient_hpo_ids(patient_state)
    branch_conc = _branch_concentration(patient_hpos, hpo_ontology)
    sparse_threshold = int(_pairwise_cfg(cfg, "dynamic_sparse_hpo_threshold", 8))
    branch_threshold = float(_pairwise_cfg(cfg, "dynamic_branch_concentration_threshold", 0.75))
    score_ratio_threshold = float(_pairwise_cfg(cfg, "dynamic_rank30_score_ratio_threshold", 0.70))

    score_col = "adjusted_score" if "adjusted_score" in ranked_df.columns else "total_score"
    score_ratio = 0.0
    if score_col in ranked_df.columns and len(ranked_df) >= base_n:
        top_score = float(ranked_df.iloc[0].get(score_col, 0.0) or 0.0)
        base_score = float(ranked_df.iloc[base_n - 1].get(score_col, 0.0) or 0.0)
        if top_score > 0:
            score_ratio = base_score / top_score

    frontier_flags = frontier_flags or {}
    frontier_outside_base = False
    if frontier_flags and max_n > base_n and "disease_id" in ranked_df.columns:
        tail_ids = set(str(x) for x in ranked_df.iloc[base_n:max_n]["disease_id"].tolist())
        frontier_outside_base = any(str(did) in tail_ids for did in frontier_flags)

    reasons = []
    if len(patient_hpos) <= sparse_threshold:
        reasons.append(f"sparse_hpos={len(patient_hpos)}")
    if branch_conc >= branch_threshold:
        reasons.append(f"branch_concentration={branch_conc:.2f}")
    if score_ratio >= score_ratio_threshold:
        reasons.append(f"rank{base_n}_score_ratio={score_ratio:.2f}")
    if frontier_outside_base:
        reasons.append("frontier_flag_in_tail")

    if reasons:
        logger.info(
            f"Pairwise ({track}): dynamic Top-N expanded {base_n}->{max_n} "
            f"because {', '.join(reasons)}"
        )
        return max_n

    logger.info(
        f"Pairwise ({track}): dynamic Top-N kept at {base_n} "
        f"(hpos={len(patient_hpos)}, branch_concentration={branch_conc:.2f}, "
        f"rank{base_n}_score_ratio={score_ratio:.2f})"
    )
    return base_n


def run_pairwise_batch(
    llm: Any,
    patient_state: Any,
    ranked_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    kg_index: KGIndex,
    kg: Dict[str, Any],
    hpo_ontology: Any,
    cfg: Any,
    prompt_dir: str,
    frontier_flags: Optional[Dict[str, Dict[str, Any]]] = None,
    track: str = "subtype",
) -> List[Dict[str, Any]]:
    """Run all pairwise comparisons for the top-N candidates."""
    prompt_path = Path(prompt_dir)
    frontier_flags = frontier_flags or {}

    top_n = resolve_pairwise_top_n(
        patient_state,
        ranked_df,
        hpo_ontology,
        cfg,
        frontier_flags=frontier_flags,
        track=track,
    )
    subset = ranked_df.head(top_n).reset_index(drop=True)

    audit_by_id = {a["disease_id"]: a for a in audit_results}

    pair_indices = list(combinations(range(len(subset)), 2))
    logger.info(f"Pairwise ({track}): {len(pair_indices)} raw pairs before skipping")

    system_prompt: Optional[str] = None
    users: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, j in pair_indices:
        row_a = subset.iloc[i]
        row_b = subset.iloc[j]

        if should_skip_pair(row_a, row_b, cfg):
            continue

        system, user, ptype = build_pairwise_prompts(
            patient_state, row_a, row_b,
            audit_by_id.get(row_a["disease_id"]),
            audit_by_id.get(row_b["disease_id"]),
            kg_index, kg, hpo_ontology, prompt_path, cfg,
            frontier_flag_a=frontier_flags.get(row_a["disease_id"]),
            frontier_flag_b=frontier_flags.get(row_b["disease_id"]),
        )
        system_prompt = system
        users.append(user)
        metas.append({
            "disease_a_id": row_a["disease_id"],
            "disease_a_name": row_a["disease_name"],
            "disease_b_id": row_b["disease_id"],
            "disease_b_name": row_b["disease_name"],
            "prompt_type": ptype,
            "rank_a": int(row_a["rank"]) if "rank" in row_a else i + 1,
            "rank_b": int(row_b["rank"]) if "rank" in row_b else j + 1,
        })
    logger.info(f"Pairwise ({track}): {len(users)} filtered pairs before skipping")

    results: List[Dict[str, Any]] = []
    if users:
        raws = llm.chat_batch(system_prompt or "", users, task="extraction")
        import json as _json
        for raw, meta in zip(raws, metas):
            raw_str = _json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
            parsed = parse_pairwise(raw_str)
            results.append({**meta, **parsed, "raw_response": raw_str})

    # Bidirectional swap test for frontier-flagged pairs
    if cfg.pairwise.bidirectional_swap and frontier_flags:
        flagged_ids = set(frontier_flags.keys())
        swap_users: List[str] = []
        swap_metas: List[Dict[str, Any]] = []
        for meta in metas:
            if meta["disease_a_id"] in flagged_ids or meta["disease_b_id"] in flagged_ids:
                # Swap: run with A and B reversed
                row_a = subset[subset["disease_id"] == meta["disease_b_id"]].iloc[0]
                row_b = subset[subset["disease_id"] == meta["disease_a_id"]].iloc[0]
                system, user, ptype = build_pairwise_prompts(
                    patient_state, row_a, row_b,
                    audit_by_id.get(row_a["disease_id"]),
                    audit_by_id.get(row_b["disease_id"]),
                    kg_index, kg, hpo_ontology, prompt_path, cfg,
                    frontier_flag_a=frontier_flags.get(row_a["disease_id"]),
                    frontier_flag_b=frontier_flags.get(row_b["disease_id"]),
                )
                swap_users.append(user)
                swap_metas.append({
                    "disease_a_id": meta["disease_b_id"],
                    "disease_a_name": meta["disease_b_name"],
                    "disease_b_id": meta["disease_a_id"],
                    "disease_b_name": meta["disease_a_name"],
                    "prompt_type": ptype + "_swap",
                    "rank_a": meta["rank_b"],
                    "rank_b": meta["rank_a"],
                })
        if swap_users:
            import json as _json
            swap_raws = llm.chat_batch(system_prompt or "", swap_users, task="extraction")
            for raw, meta in zip(swap_raws, swap_metas):
                raw_str = _json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
                parsed = parse_pairwise(raw_str)
                results.append({**meta, **parsed, "raw_response": raw_str})

    logger.info(f"Pairwise ({track}): {len(results)} verdicts produced")
    return results
