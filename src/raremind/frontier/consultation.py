"""Stage 4: Frontier Clinical Reasoning Consultation.

This stage runs AT MOST ONCE per patient. It is triggered when the local
deterministic pipeline signals uncertainty or incongruity. The frontier
model reviews the candidate list through 5 clinical reasoning lenses:
  A) Incongruous features
  B) Temporal pattern
  C) Demographic alignment
  D) Gene-phenotype coherence
  E) Critical absences

The model returns ONLY noticeable candidates — underranked and overranked —
with verbatim names from the candidate list. This is NOT zero-shot diagnosis.

Output:
  - Underranked candidates -> audit_category = "strong"
  - Overranked candidates -> audit_category = "weak"
  - Unmentioned candidates -> proceed to normal audit

Name matching:
  1. Exact string match on disease_name
  2. Case-insensitive match
  3. BioLORD embedding similarity (>=0.9)
  4. Log warning if no match found
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from raremind.core.json_utils import safe_json_load
from raremind.core.utils import read_prompt

logger = logging.getLogger(__name__)


MONDO_REF_RE = re.compile(r"\b(MONDO:\d+)\b", flags=re.IGNORECASE)


def should_trigger_frontier(
    incongruity_strength: str,
    ranked_df: pd.DataFrame,
    cfg: Any,
) -> Tuple[bool, str]:
    """Check the trigger conditions. Returns (should_fire, reason)."""
    trig = cfg.frontier.trigger

    if getattr(trig, "force_always", False):
        return True, "forced"

    if getattr(trig, "on_incongruity", True) and incongruity_strength in ("moderate", "strong"):
        return True, f"incongruity={incongruity_strength}"

    if getattr(trig, "on_top5_ambiguity", True) and len(ranked_df) >= 5:
        top1 = ranked_df.iloc[0]["total_score"]
        top5 = ranked_df.iloc[4]["total_score"]
        if top1 > 0 and (top1 - top5) / top1 < 0.10:
            return True, f"top5_ambiguity: rank1={top1:.3f}, rank5={top5:.3f}"

    if getattr(trig, "on_low_specific_signal", True) and "specific_signal_score" in ranked_df.columns:
        top20 = ranked_df.head(20)
        if top20["specific_signal_score"].max() == 0:
            return True, "no specific signal in top-20"

    return False, "no trigger"


def build_frontier_prompt(
    patient_evidence: Dict[str, Any],
    ranked_df: pd.DataFrame,
    incongruity: Dict[str, Any],
    top_n: int,
    prompt_dir: Path,
) -> Tuple[str, str]:
    """Construct the frontier consultation prompts (system + user)."""
    system = read_prompt(prompt_dir / "frontier" / "frontier_consultation_system.txt")
    user_tpl = read_prompt(prompt_dir / "frontier" / "frontier_consultation_user.txt")

    # Format patient profile (human-readable, not raw JSON)
    demo = patient_evidence.get("demographics", {})
    demo_line = (
        f"Age: {demo.get('age', 'unknown')}; "
        f"Sex: {demo.get('sex', 'unknown')}; "
        f"Ethnicity: {demo.get('ethnicity', 'unknown')}"
    )

    # Phenotypes sorted by onset when available
    phens_present = []
    for p in patient_evidence.get("phenotypes", []):
        if not p.get("present", True):
            continue
        t = p.get("text") or p.get("mention")
        hid = p.get("hpo_id") or ""
        onset = p.get("onset") or ""
        phens_present.append(f"- {t} ({hid}){f' [onset: {onset}]' if onset else ''}")

    phens_negated = []
    for p in patient_evidence.get("phenotypes", []):
        if p.get("present", True):
            continue
        t = p.get("text") or p.get("mention")
        hid = p.get("hpo_id") or ""
        phens_negated.append(f"- {t} ({hid})")

    # Temporal summary
    temporal = patient_evidence.get("temporal_view", {})
    earliest = temporal.get("earliest_features", [])
    temporal_line = (
        "Earliest features: " + "; ".join(earliest) if earliest else "No explicit onset information."
    )

    # Incongruity summary
    incon_lines = []
    if incongruity.get("overall_incongruity_strength") in ("moderate", "strong"):
        dom = incongruity.get("dominant_branch_name") or incongruity.get("dominant_branch", "")
        incon_lines.append(f"Dominant clinical picture: {dom}")
        for item in incongruity.get("incongruous_phenotypes", [])[:5]:
            incon_lines.append(
                f"  - Incongruous: {item.get('mention')} ({item.get('hpo_id')}) "
                f"[branch: {item.get('branch_name', item.get('branch', ''))}]"
            )
    incon_block = "\n".join(incon_lines) if incon_lines else "No significant incongruity detected."

    # Gene evidence
    genes = patient_evidence.get("gene_evidence", {})
    gene_mentions = [g.get("gene") for g in genes.get("gene_mentions", []) if g.get("gene")]
    vcf_genes = [g.get("gene") for g in genes.get("vcf_summary", []) if g.get("gene")]
    gene_line = (
        f"Gene mentions: {', '.join(gene_mentions) if gene_mentions else 'none'}. "
        f"VCF: {', '.join(vcf_genes) if vcf_genes else 'none'}."
    )

    # Family history summary
    fam = patient_evidence.get("family_history", [])
    fam_lines = []
    for f in fam[:5]:
        rel = f.get("relation", "")
        diseases = ", ".join(f.get("diseases", []) or [])
        phens = ", ".join(f.get("phenotypes", []) or [])
        fam_lines.append(
            f"- {rel}: diseases=[{diseases}] phenotypes=[{phens}]"
        )
    fam_block = "\n".join(fam_lines) if fam_lines else "No family history reported."

    patient_profile = f"""Demographics: {demo_line}

Present phenotypes ({len(phens_present)}):
{chr(10).join(phens_present) if phens_present else '(none)'}

Explicitly absent / negated:
{chr(10).join(phens_negated) if phens_negated else '(none)'}

Temporal pattern: {temporal_line}

Incongruity analysis:
{incon_block}

Gene evidence: {gene_line}

Family history:
{fam_block}
"""

    # Candidate list (names + MONDO IDs + current rank)
    df_top = ranked_df.head(top_n)
    cand_lines = []
    for i, row in df_top.iterrows():
        cand_lines.append(
            f"{row.get('rank', i + 1)}. {row['disease_name']} ({row['disease_id']})"
        )
    candidate_block = "\n".join(cand_lines)

    user = user_tpl.format(
        patient_profile=patient_profile,
        candidate_list=candidate_block,
        top_n=top_n,
    )
    return system, user


def match_disease_name(
    name: str,
    candidate_df: pd.DataFrame,
    threshold: float = 90.0,
) -> Optional[Dict[str, Any]]:
    """Return the matching row from candidate_df for a disease name, or None.

    Matching order:
      1. Exact match on disease_name
      2. Case-insensitive match
      3. Match on disease_id if the name looks like MONDO:xxx
      4. Fuzzy match with rapidfuzz (threshold default 90)
    """
    if not name:
        return None

    name_clean, mondo_id = _split_disease_reference(name)

    # Exact
    exact = candidate_df[candidate_df["disease_name"] == name_clean]
    if len(exact) > 0:
        return exact.iloc[0].to_dict()

    # Case-insensitive
    lower = candidate_df["disease_name"].str.lower() == name_clean.lower()
    if lower.any():
        return candidate_df[lower].iloc[0].to_dict()

    # MONDO ID
    candidate_id = mondo_id or (name_clean.upper() if name_clean.upper().startswith("MONDO:") else "")
    if candidate_id:
        did = candidate_df[candidate_df["disease_id"].str.upper() == candidate_id.upper()]
        if len(did) > 0:
            return did.iloc[0].to_dict()

    # Fuzzy
    names = candidate_df["disease_name"].tolist()
    match = process.extractOne(name_clean, names, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return candidate_df[candidate_df["disease_name"] == match[0]].iloc[0].to_dict()

    return None


def _split_disease_reference(raw_name: str) -> Tuple[str, str]:
    """Separate an LLM-returned disease reference into a clean name and MONDO ID.

    Frontier outputs often use candidate-list text such as
    "Noonan syndrome (MONDO:0018997)". Matching should prefer the clean disease
    name first, then use the MONDO ID as a fallback.
    """
    text = str(raw_name or "").strip()
    if not text:
        return "", ""

    mondo_match = MONDO_REF_RE.search(text)
    mondo_id = mondo_match.group(1).upper() if mondo_match else ""
    clean = MONDO_REF_RE.sub("", text)
    clean = re.sub(r"\(\s*\)", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip(" \t\r\n-:;,()")
    return clean or mondo_id, mondo_id


def parse_frontier_output(
    raw_output: str,
    candidate_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse frontier output and resolve disease names to candidate rows."""
    data = safe_json_load(raw_output, prefer="object")

    underranked_raw = data.get("underranked", []) if isinstance(data, dict) else []
    overranked_raw = data.get("overranked", []) if isinstance(data, dict) else []

    def resolve(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            name = e.get("disease_name", "")
            matched = match_disease_name(name, candidate_df)
            if matched is None:
                logger.warning(f"Frontier mentioned disease not found in candidates: '{name}'")
                e["matched"] = False
                out.append(e)
                continue
            e["matched"] = True
            e["disease_id"] = matched["disease_id"]
            e["disease_name"] = matched["disease_name"]
            e["current_rank"] = int(matched.get("rank", 0)) if pd.notna(matched.get("rank")) else None
            out.append(e)
        return out

    return {
        "underranked": resolve(underranked_raw),
        "overranked": resolve(overranked_raw),
        "raw_output": raw_output,
    }


def run_frontier_consultation(
    frontier_client: Any,
    patient_evidence: Dict[str, Any],
    ranked_df: pd.DataFrame,
    incongruity: Dict[str, Any],
    cfg: Any,
    prompt_dir: str,
) -> Dict[str, Any]:
    """Run Stage 4. Returns the parsed frontier output or {} if not triggered."""
    prompt_path = Path(prompt_dir)
    top_n = cfg.frontier.top_n_candidates

    # Trigger check
    fire, reason = should_trigger_frontier(
        incongruity.get("overall_incongruity_strength", "none"),
        ranked_df,
        cfg,
    )
    if not fire:
        logger.info(f"Frontier consultation skipped: {reason}")
        return {"triggered": False, "trigger_reason": reason}

    logger.info(f"Frontier consultation triggered: {reason}")

    # Build prompts
    system, user = build_frontier_prompt(
        patient_evidence, ranked_df, incongruity, top_n, prompt_path
    )

    # Call
    raw = frontier_client.chat(system, user)

    # Parse
    parsed = parse_frontier_output(raw, ranked_df.head(top_n))
    parsed["triggered"] = True
    parsed["trigger_reason"] = reason
    parsed["system_prompt"] = system
    parsed["user_prompt"] = user

    return parsed
