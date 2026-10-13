"""Stage 5: Per-candidate audit.

Produces ordinal plausibility (strong/moderate/weak/implausible) per candidate
with grounded evidence (every quote validated against the actual input).
Frontier-underranked candidates are promoted directly; overranked candidates
still go through the audit before they can be excluded downstream.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from raremind.core.json_utils import safe_json_load, validate_quote
from raremind.core.utils import read_prompt
from raremind.core.compat import to_dict
from raremind.core.config import audit_top_n_candidates
from raremind.kg.kg_precompute import KGIndex, IMPORTANCE_WEIGHT, FREQUENCY_WEIGHT
from raremind.scoring.phenotype_score import _are_antonymic_pair

logger = logging.getLogger(__name__)


PLAUSIBILITY_MULTIPLIER = {
    "strong": 1.3,
    "moderate": 1.0,
    "weak": 0.7,
    "implausible": 0.3,
}

HPO_ID_RE = re.compile(r"HP:\d{7}")
QUOTED_PHENOTYPE_RE = re.compile(
    r"(?P<label>.*?)\((?P<hpo>HP:\d{7})\)\s*\[IC=(?P<ic>[0-9.]+)(?:;\s*(?P<importance>[^;\]]+))?(?:;\s*(?P<frequency>[^\]]+))?\]"
)
HPO_MENTION_RE = re.compile(r"(?P<label>[^()\n\r]*)\((?P<hpo>HP:\d{7})\)")
LLM_VALIDATION_WEIGHT = 1.0

PRESENT_POLARITY = "present"
ABSENT_POLARITY = "absent"
AUDIT_OUTPUT_KEYS = (
    "disease_id",
    "disease_name",
    "plausibility",
    "supporting_evidence",
    "contradicting_evidence",
    "missing_expected_evidence",
    "key_distinguishing_note",
)


def compact_patient_evidence(patient_state: Any, top_n_hpos: int = 12) -> str:
    """Format the patient evidence compactly for audit / pairwise prompts."""
    lines: List[str] = []

    demo = patient_state.demographics
    if isinstance(demo, dict):
        def _val(k: str) -> str:
            v = demo.get(k, {})
            if isinstance(v, dict):
                return str(v.get("value", ""))
            return str(v) if v else ""
        lines.append(
            f"Demographics: age={_val('age')}, sex={_val('sex')}, ethnicity={_val('ethnicity')}"
        )

    # Present HPOs sorted by IC desc
    all_phens = [to_dict(h) for h in patient_state.normalized_hpo]
    present = [h for h in all_phens if h.get("present", True)]
    present.sort(key=lambda h: h.get("ic", 0.0), reverse=True)
    if present:
        lines.append("Present phenotypes (top by specificity):")
        for h in present[:top_n_hpos]:
            lines.append(
                f"  - {h.get('mention','')} ({h.get('hpo_id','')}) [IC={h.get('ic',0.0):.1f}]"
            )

    negated = [h for h in all_phens if not h.get("present", True)]
    if negated:
        lines.append("Explicitly absent / negated:")
        for h in negated[:8]:
            lines.append(f"  - {h.get('mention','')} ({h.get('hpo_id','')})")

    # Temporal
    tv = to_dict(patient_state.temporal_view)
    early = tv.get("earliest_features", []) if isinstance(tv, dict) else []
    if early:
        lines.append("Earliest features: " + "; ".join(early))

    # Genes
    if patient_state.gene_mentions or patient_state.vcf_summary:
        g_text = [g.get("gene", "") for g in (patient_state.gene_mentions or []) if g.get("gene")]
        g_vcf = [g.get("gene", "") for g in (patient_state.vcf_summary or []) if g.get("gene")]
        lines.append(f"Gene evidence: text={g_text}, vcf={g_vcf}")

    # Family
    if patient_state.family_history:
        lines.append("Family history present.")
        for f in patient_state.family_history[:3]:
            lines.append(
                f"  - {f.get('relation','')} affected with {f.get('diseases',[])} / {f.get('phenotypes',[])}"
            )

    # Incongruity
    inc = to_dict(patient_state.incongruity)
    if isinstance(inc, dict) and inc.get("overall_incongruity_strength") in ("moderate", "strong"):
        lines.append(
            f"Incongruity ({inc['overall_incongruity_strength']}): dominant="
            f"{inc.get('dominant_branch_name','')}; outliers="
            + ", ".join([x.get("mention","") for x in inc.get("incongruous_phenotypes", [])[:3]])
        )

    return "\n".join(lines)


def select_narrative_excerpt(
    narrative: str,
    patient_mentions: List[str],
    max_words: int = 120,
) -> str:
    """Micro-search within a candidate's narrative for the best-matching passage."""
    if not narrative:
        return ""

    # Split into ~2-sentence chunks
    sentences = re.split(r"(?<=[.!?])\s+", narrative)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        if len((current + " " + s).split()) <= max_words:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    if not chunks:
        return narrative[: max_words * 6]

    patient_words = set()
    for m in patient_mentions:
        for w in re.findall(r"[A-Za-z]+", str(m).lower()):
            if len(w) > 3:
                patient_words.add(w)

    best_chunk = chunks[0]
    best_score = -1
    for c in chunks:
        c_words = set(re.findall(r"[A-Za-z]+", c.lower()))
        score = len(c_words & patient_words)
        if score > best_score:
            best_score = score
            best_chunk = c
    return best_chunk


def _split_quote_segments(quote: str) -> List[str]:
    parts = []
    for raw in re.split(r"[\r\n]+", str(quote or "")):
        piece = raw.strip().lstrip("-* ").strip()
        if piece:
            parts.append(piece)
    return parts


def _validate_grounded_quote(quote: str, source_text: str) -> bool:
    if not quote or not source_text:
        return False
    parts = _split_quote_segments(quote)
    if len(parts) <= 1:
        return validate_quote(quote, source_text)
    return all(validate_quote(part, source_text) for part in parts)


def _source_quote_for_hpo(source_text: str, hpo_id: str) -> str:
    for part in _split_quote_segments(source_text):
        if hpo_id in part:
            return part
    return ""


def _grounded_quote(quote: str, source_text: str) -> str:
    """Return grounded quote lines; invalid lines in a multi-line claim are dropped."""
    if not quote or not source_text:
        return ""
    grounded = []
    seen: set[str] = set()
    for part in _split_quote_segments(quote):
        candidate = part if validate_quote(part, source_text) else ""
        if not candidate:
            for hpo_id in HPO_ID_RE.findall(part):
                candidate = _source_quote_for_hpo(source_text, hpo_id)
                if candidate:
                    break
        if candidate and candidate not in seen:
            grounded.append(candidate)
            seen.add(candidate)
    return "\n".join(grounded)


def _normalize_fragment(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", str(text).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _fragment_tokens(text: str) -> set[str]:
    return {tok for tok in _normalize_fragment(text).split() if len(tok) > 2}


def _cue_is_covered(cue: str, evidence_entry: Dict[str, Any]) -> bool:
    cue_norm = _normalize_fragment(cue)
    cue_tokens = _fragment_tokens(cue)
    if not cue_norm or not cue_tokens:
        return False

    variants = [evidence_entry.get("cue", ""), evidence_entry.get("patient_quote", ""), evidence_entry.get("disease_quote", "")]
    variants.extend(_split_quote_segments(evidence_entry.get("disease_quote", "")))
    for variant in variants:
        variant_norm = _normalize_fragment(variant)
        variant_tokens = _fragment_tokens(variant)
        if not variant_norm:
            continue
        if cue_norm in variant_norm or variant_norm in cue_norm:
            return True
        if cue_tokens <= variant_tokens:
            return True
    return False


def _filter_missing_expected_evidence(
    missing: List[Dict[str, Any]],
    supporting: List[Dict[str, Any]],
    contradicting: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    covered = list(supporting) + list(contradicting)
    out: List[Dict[str, Any]] = []
    for item in missing:
        cue = item.get("cue", "")
        if cue and any(_cue_is_covered(cue, entry) for entry in covered):
            continue
        out.append(item)
    return out


def _infer_polarity(source_text: str, hpo_id: str, start: int = 0) -> str:
    """Infer phenotype polarity from the compact patient/disease evidence section."""
    text = source_text or ""
    lower_prefix = text[: max(0, start)].lower()
    neg_anchor = lower_prefix.rfind("explicitly absent / negated:")
    present_anchor = lower_prefix.rfind("present phenotypes")
    absent_anchor = lower_prefix.rfind("typically absent / excluded:")
    hallmark_anchor = lower_prefix.rfind("hallmark phenotypes:")

    if neg_anchor > present_anchor:
        return ABSENT_POLARITY
    if absent_anchor > hallmark_anchor:
        return ABSENT_POLARITY
    return PRESENT_POLARITY


def _source_phenotypes_by_hpo(source_text: str) -> Dict[str, Dict[str, Any]]:
    by_hpo: Dict[str, Dict[str, Any]] = {}
    for match in QUOTED_PHENOTYPE_RE.finditer(source_text or ""):
        by_hpo[match.group("hpo")] = {
            "label": match.group("label").strip(),
            "hpo_id": match.group("hpo"),
            "ic": float(match.group("ic")),
            "importance": (match.group("importance") or "").strip().lower() or None,
            "frequency": (match.group("frequency") or "").strip().lower() or None,
            "polarity": _infer_polarity(source_text, match.group("hpo"), match.start()),
        }
    for match in HPO_MENTION_RE.finditer(source_text or ""):
        hpo_id = match.group("hpo")
        by_hpo.setdefault(hpo_id, {
            "label": match.group("label").strip(),
            "hpo_id": hpo_id,
            "ic": None,
            "importance": None,
            "frequency": None,
            "polarity": _infer_polarity(source_text, hpo_id, match.start()),
        })
    return by_hpo


def _source_phenotypes_by_label(source_text: str) -> Dict[str, Dict[str, Any]]:
    by_label: Dict[str, Dict[str, Any]] = {}
    for phenotype in _source_phenotypes_by_hpo(source_text).values():
        label_norm = _normalize_fragment(phenotype.get("label", ""))
        if label_norm:
            by_label[label_norm] = phenotype
    return by_label


def _fallback_ic(hpo_id: str, hpo: Any | None) -> float | None:
    if hpo is None or not hpo_id:
        return None
    try:
        return float(hpo.get_ic(hpo_id))
    except Exception:
        return None


def _extract_quote_phenotypes(
    quote: str,
    source_text: str = "",
    hpo: Any | None = None,
) -> List[Dict[str, Any]]:
    phenotypes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    source_by_hpo = _source_phenotypes_by_hpo(source_text)
    source_by_label = _source_phenotypes_by_label(source_text)

    for segment in _split_quote_segments(quote):
        found_hpo = False
        for match in QUOTED_PHENOTYPE_RE.finditer(segment):
            seen.add(match.group("hpo"))
            found_hpo = True
            phenotypes.append({
                "label": match.group("label").strip(),
                "hpo_id": match.group("hpo"),
                "ic": float(match.group("ic")),
                "importance": (match.group("importance") or "").strip().lower() or None,
                "frequency": (match.group("frequency") or "").strip().lower() or None,
                "polarity": (source_by_hpo.get(match.group("hpo"), {}) or {}).get("polarity", PRESENT_POLARITY),
            })
        for match in HPO_MENTION_RE.finditer(segment):
            hpo_id = match.group("hpo")
            if hpo_id in seen:
                continue
            found_hpo = True
            fallback = source_by_hpo.get(hpo_id, {})
            ic = fallback.get("ic")
            if ic is None:
                ic = _fallback_ic(hpo_id, hpo)
            phenotypes.append({
                "label": (fallback.get("label") or match.group("label")).strip(),
                "hpo_id": hpo_id,
                "ic": ic,
                "importance": fallback.get("importance"),
                "frequency": fallback.get("frequency"),
                "polarity": fallback.get("polarity", PRESENT_POLARITY),
            })
            seen.add(hpo_id)
        if found_hpo:
            continue

        # Some local models quote a grounded phenotype name but omit the HPO ID.
        # Recover the HPO/IC from the patient evidence or disease card.
        fallback = source_by_label.get(_normalize_fragment(segment))
        hpo_id = fallback.get("hpo_id", "") if fallback else ""
        if hpo_id and hpo_id not in seen:
            phenotypes.append({
                "label": fallback.get("label", segment).strip(),
                "hpo_id": hpo_id,
                "ic": fallback.get("ic") if fallback.get("ic") is not None else _fallback_ic(hpo_id, hpo),
                "importance": fallback.get("importance"),
                "frequency": fallback.get("frequency"),
                "polarity": fallback.get("polarity", PRESENT_POLARITY),
            })
            seen.add(hpo_id)
    return phenotypes


def _best_validation_score(
    entry: Dict[str, Any],
    patient_evidence_text: str = "",
    disease_card_text: str = "",
    hpo: Any | None = None,
) -> float:
    patient_phens = _extract_quote_phenotypes(entry.get("patient_quote", ""), patient_evidence_text, hpo)
    disease_phens = _extract_quote_phenotypes(entry.get("disease_quote", ""), disease_card_text, hpo)
    if not patient_phens or not disease_phens:
        return 0.0

    best = 0.0
    for patient in patient_phens:
        patient_ic = patient.get("ic") or 0.0
        for disease in disease_phens:
            if not _valid_support_pair(patient, disease, hpo):
                continue
            disease_ic = disease.get("ic") or 0.0
            imp_w = IMPORTANCE_WEIGHT.get(disease.get("importance"), 0.75)
            freq_w = FREQUENCY_WEIGHT.get(disease.get("frequency"), 0.5)
            best = max(best, max(patient_ic, disease_ic) * imp_w * freq_w)
    return best


def _hpo_same_or_related(patient_hpo: str, disease_hpo: str, hpo: Any | None = None) -> bool:
    if not patient_hpo or not disease_hpo:
        return False
    if patient_hpo == disease_hpo:
        return True
    if hpo is None:
        return False
    try:
        return (
            disease_hpo in hpo.get_ancestors(patient_hpo, include_self=False)
            or disease_hpo in hpo.get_descendants(patient_hpo, include_self=False)
            or patient_hpo in hpo.get_ancestors(disease_hpo, include_self=False)
            or patient_hpo in hpo.get_descendants(disease_hpo, include_self=False)
        )
    except Exception:
        return False


def _hpo_directly_related(patient_hpo: str, disease_hpo: str, hpo: Any | None = None) -> bool:
    if not patient_hpo or not disease_hpo:
        return False
    if patient_hpo == disease_hpo:
        return True
    if hpo is None:
        return False
    try:
        patient_direct = hpo.get_parents(patient_hpo) | hpo.get_children(patient_hpo) | hpo.get_siblings(patient_hpo)
        disease_direct = hpo.get_parents(disease_hpo) | hpo.get_children(disease_hpo) | hpo.get_siblings(disease_hpo)
        return disease_hpo in patient_direct or patient_hpo in disease_direct
    except Exception:
        return False


def _valid_support_pair(patient: Dict[str, Any], disease: Dict[str, Any], hpo: Any | None = None) -> bool:
    if _are_antonymic_pair(patient.get("label", ""), disease.get("label", "")):
        return False

    patient_hpo = patient.get("hpo_id", "")
    disease_hpo = disease.get("hpo_id", "")
    if patient_hpo == disease_hpo:
        return True
    if _hpo_directly_related(patient_hpo, disease_hpo, hpo):
        return True
    if hpo is None:
        return False
    try:
        return float(hpo.get_mica_ic(patient_hpo, disease_hpo)) > 2.0
    except Exception:
        return False


def _valid_contradiction_pair(patient: Dict[str, Any], disease: Dict[str, Any], hpo: Any | None = None) -> bool:
    patient_pol = patient.get("polarity", PRESENT_POLARITY)
    disease_pol = disease.get("polarity", PRESENT_POLARITY)
    related = _hpo_same_or_related(patient.get("hpo_id", ""), disease.get("hpo_id", ""), hpo)

    # Patient present vs disease expects absent, or patient explicitly absent vs disease expects present.
    if related and patient_pol != disease_pol:
        return True

    # Mutually opposite present phenotypes, e.g. macrocephaly vs microcephaly.
    if patient_pol == PRESENT_POLARITY and disease_pol == PRESENT_POLARITY:
        if _are_antonymic_pair(patient.get("label", ""), disease.get("label", "")):
            return True

    return False


def _literal_audit_load(raw: str) -> Dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json|python)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None

    # Try the whole object first, then the largest apparent object prefix.
    candidates = [text[start:]]
    end = text.rfind("}")
    if end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _find_key_value_start(raw: str, key: str) -> int:
    match = re.search(rf"['\"]{re.escape(key)}['\"]\s*:", raw)
    if not match:
        return -1
    idx = match.end()
    while idx < len(raw) and raw[idx].isspace():
        idx += 1
    return idx


def _balanced_span(raw: str, start: int, opener: str, closer: str) -> Tuple[int, int] | None:
    if start < 0 or start >= len(raw) or raw[start] != opener:
        return None
    depth = 0
    in_str: str | None = None
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return start, idx + 1
    return None


def _complete_object_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        span = _balanced_span(text, start, "{", "}")
        if not span:
            break
        blocks.append(text[span[0] : span[1]])
        idx = span[1]
    return blocks


def _literal_list_for_key(raw: str, key: str) -> List[Any]:
    start = _find_key_value_start(raw, key)
    if start < 0 or start >= len(raw) or raw[start] != "[":
        return []
    span = _balanced_span(raw, start, "[", "]")
    if span:
        list_text = raw[span[0] : span[1]]
    else:
        end = len(raw)
        for other_key in AUDIT_OUTPUT_KEYS:
            if other_key == key:
                continue
            match = re.search(rf",\s*['\"]{re.escape(other_key)}['\"]\s*:", raw[start + 1 :])
            if match:
                end = min(end, start + 1 + match.start())
        list_text = raw[start:end]
    try:
        parsed = ast.literal_eval(list_text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        items: List[Any] = []
        for block in _complete_object_blocks(list_text):
            try:
                parsed = ast.literal_eval(block)
            except Exception:
                continue
            if isinstance(parsed, dict):
                items.append(parsed)
        return items


def _literal_scalar_for_key(raw: str, key: str) -> str:
    start = _find_key_value_start(raw, key)
    if start < 0 or start >= len(raw):
        return ""
    quote = raw[start]
    if quote not in ("'", '"'):
        match = re.match(r"([^,\n\r}\]]+)", raw[start:])
        return match.group(1).strip() if match else ""

    out: List[str] = []
    idx = start + 1
    escape = False
    while idx < len(raw):
        ch = raw[idx]
        if escape:
            out.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == quote:
            return "".join(out)
        else:
            out.append(ch)
        idx += 1
    return "".join(out)


def _repair_audit_dict(raw: str) -> Dict[str, Any]:
    """Salvage known audit fields from JSON/Python-dict-like malformed output."""
    text = str(raw or "")
    literal = _literal_audit_load(text)
    if literal is not None:
        return literal

    repaired: Dict[str, Any] = {}
    for key in ("disease_id", "disease_name", "plausibility", "key_distinguishing_note"):
        value = _literal_scalar_for_key(text, key)
        if value:
            repaired[key] = value

    for key in ("supporting_evidence", "contradicting_evidence", "missing_expected_evidence"):
        items = []
        for item in _literal_list_for_key(text, key):
            if isinstance(item, dict):
                items.append(item)
        if items:
            repaired[key] = items

    return repaired


def _filter_contradicting_entries(
    entries: List[Dict[str, Any]],
    patient_evidence_text: str = "",
    disease_card_text: str = "",
    hpo: Any | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]:
    filtered: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    valid_pairs: List[Tuple[str, str]] = []
    seen_pairs: set[Tuple[str, str]] = set()
    seen_disease_hpos: set[str] = set()

    for entry in entries:
        patient_phens = _extract_quote_phenotypes(entry.get("patient_quote", ""), patient_evidence_text, hpo)
        disease_phens = _extract_quote_phenotypes(entry.get("disease_quote", ""), disease_card_text, hpo)
        keep_pair: Tuple[str, str] | None = None
        for patient in patient_phens:
            for disease in disease_phens:
                pair = (patient.get("hpo_id", ""), disease.get("hpo_id", ""))
                if pair in seen_pairs:
                    continue
                if _valid_contradiction_pair(patient, disease, hpo):
                    keep_pair = pair
                    break
            if keep_pair:
                break
        if not keep_pair:
            discarded.append(entry)
            continue

        disease_hpo = keep_pair[1]
        # Avoid repeated penalty inflation when the LLM reuses one disease feature.
        if disease_hpo in seen_disease_hpos:
            discarded.append(entry)
            continue
        seen_pairs.add(keep_pair)
        seen_disease_hpos.add(disease_hpo)
        valid_pairs.append(keep_pair)
        filtered.append(entry)

    return filtered, discarded, valid_pairs


def _signed_validation_score(
    supporting: List[Dict[str, Any]],
    contradicting: List[Dict[str, Any]],
    patient_evidence_text: str = "",
    disease_card_text: str = "",
    hpo: Any | None = None,
) -> Dict[str, float | str]:
    support_score = sum(
        _best_validation_score(entry, patient_evidence_text, disease_card_text, hpo)
        for entry in supporting
    )
    contradiction_score = sum(
        _best_validation_score(entry, patient_evidence_text, disease_card_text, hpo)
        for entry in contradicting
    )
    raw_score = support_score - contradiction_score
    return {
        "validation_support_score": support_score,
        "validation_contradiction_score": contradiction_score,
        "validation_raw_score": raw_score,
        "validation_source": "llm_validation" if raw_score != 0 else "",
    }


def build_audit_prompts(
    patient_state: Any,
    candidate_row: pd.Series,
    kg_index: KGIndex,
    prompt_dir: Path,
    use_narrative: bool = True,
    frontier_note: str = "",
) -> Tuple[str, str]:
    system = read_prompt(prompt_dir / "reasoning" / "audit_system.txt")
    user_tpl = read_prompt(prompt_dir / "reasoning" / "audit_user.txt")

    did = candidate_row["disease_id"]
    dname = candidate_row["disease_name"]
    card = kg_index.disease_cards.get(did, f"Disease: {dname} ({did})")

    narrative_block = ""
    if use_narrative and did in kg_index.disease_narrative:
        patient_mentions = [
            to_dict(h).get("mention", "")
            for h in patient_state.normalized_hpo
        ]
        excerpt = select_narrative_excerpt(kg_index.disease_narrative[did], patient_mentions)
        if excerpt:
            narrative_block = (
                "NARRATIVE EXCERPT (for reference; quotes must still be verbatim):\n"
                f"\"{excerpt}\"\n\n"
            )

    user = user_tpl.format(
        patient_evidence=compact_patient_evidence(patient_state),
        disease_name=dname,
        disease_id=did,
        current_rank=candidate_row.get("rank", 0),
        disease_card=card,
        narrative_block=narrative_block,
        frontier_note=("\nFRONTIER NOTE: " + frontier_note + "\n") if frontier_note else "",
    )
    return system, user


def parse_audit_output(
    raw: str,
    disease_id: str,
    disease_name: str,
    patient_evidence_text: str,
    disease_card_text: str,
    hpo: Any | None = None,
) -> Dict[str, Any]:
    data = safe_json_load(raw, prefer="object")
    if isinstance(data, dict) and isinstance(data.get("raw"), str):
        repaired = safe_json_load(data["raw"], prefer="object")
        if isinstance(repaired, dict) and "raw" not in repaired:
            data = repaired
        else:
            data = _repair_audit_dict(data["raw"])
    elif not isinstance(data, dict):
        data = _repair_audit_dict(str(raw))
    elif "raw" in data and len(data) == 1:
        data = _repair_audit_dict(str(data.get("raw", "")))
    if not isinstance(data, dict):
        data = {}

    plaus = str(data.get("plausibility", "")).lower().strip()
    if plaus not in PLAUSIBILITY_MULTIPLIER:
        plaus = "moderate"

    # Quote-validate every evidence entry
    def _filter_entries(entries: Any) -> List[Dict[str, Any]]:
        out = []
        if not isinstance(entries, list):
            return out
        for e in entries:
            if not isinstance(e, dict):
                continue
            pq = e.get("patient_quote", "")
            dq = e.get("disease_quote", "")
            if not pq or not dq:
                continue
            grounded_pq = _grounded_quote(pq, patient_evidence_text)
            grounded_dq = _grounded_quote(dq, disease_card_text)
            if not grounded_pq:
                continue
            if not grounded_dq:
                continue
            out.append({
                "cue": e.get("cue", ""),
                "patient_quote": grounded_pq,
                "disease_quote": grounded_dq,
            })
        return out

    supporting = _filter_entries(data.get("supporting_evidence", []))
    contradicting_raw = _filter_entries(data.get("contradicting_evidence", []))
    contradicting, discarded_contradicting, valid_contradiction_pairs = _filter_contradicting_entries(
        contradicting_raw,
        patient_evidence_text=patient_evidence_text,
        disease_card_text=disease_card_text,
        hpo=hpo,
    )

    missing_raw = data.get("missing_expected_evidence", [])
    missing = []
    for m in missing_raw if isinstance(missing_raw, list) else []:
        if isinstance(m, dict):
            missing.append({
                "cue": m.get("cue", ""),
                "importance": m.get("importance", "medium"),
            })
    missing = _filter_missing_expected_evidence(missing, supporting, contradicting)
    validation = _signed_validation_score(
        supporting,
        contradicting,
        patient_evidence_text=patient_evidence_text,
        disease_card_text=disease_card_text,
        hpo=hpo,
    )

    return {
        "disease_id": disease_id,
        "disease_name": disease_name,
        "plausibility": plaus,
        "multiplier": PLAUSIBILITY_MULTIPLIER[plaus],
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "discarded_contradicting_evidence": discarded_contradicting,
        "valid_contradiction_pairs": [
            {"patient_hpo": patient_hpo, "disease_hpo": disease_hpo}
            for patient_hpo, disease_hpo in valid_contradiction_pairs
        ],
        "missing_expected_evidence": missing,
        "key_distinguishing_note": str(data.get("key_distinguishing_note", "")),
        **validation,
    }


def run_audit_batch(
    llm: Any,
    patient_state: Any,
    ranked_df: pd.DataFrame,
    kg_index: KGIndex,
    cfg: Any,
    prompt_dir: str,
    hpo: Any | None = None,
    frontier_flags: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Audit top-N candidates.

    Frontier-underranked candidates get auto-strong rescue entries. Frontier-
    overranked candidates are warnings, so they still go through normal audit.
    """
    prompt_path = Path(prompt_dir)
    frontier_flags = frontier_flags or {}

    top_n = audit_top_n_candidates(cfg)
    to_audit = ranked_df.head(top_n).copy()

    results: List[Dict[str, Any]] = []
    batch_users = []
    batch_meta: List[Dict[str, Any]] = []

    system_prompt: Optional[str] = None

    for _, row in tqdm(to_audit.iterrows(), total=len(to_audit), desc="Stage 5 audit prompts"):
        did = row["disease_id"]
        # Frontier-flagged → auto-assign, skip LLM
        flag = frontier_flags.get(did)
        # Only frontier-underranked candidates skip audit. Frontier-overranked
        # candidates still need local audit confirmation before demotion.
        if (
            cfg.audit.skip_if_frontier_flagged
            and flag
            and flag.get("flag_type") == "underranked"
        ):
            category = "strong"
            results.append({
                "disease_id": did,
                "disease_name": row["disease_name"],
                "plausibility": category,
                "multiplier": PLAUSIBILITY_MULTIPLIER[category],
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "missing_expected_evidence": [],
                "key_distinguishing_note": flag.get("reasoning", ""),
                "validation_support_score": 0.0,
                "validation_contradiction_score": 0.0,
                "validation_raw_score": 0.0,
                "validation_source": "",
                "source": f"frontier_{flag.get('flag_type', 'flag')}",
                "frontier_reasoning": flag.get("reasoning", ""),
                "frontier_lens": flag.get("lens", ""),
            })
            continue

        # Build prompts
        frontier_note = ""
        system, user = build_audit_prompts(
            patient_state, row, kg_index, prompt_path,
            use_narrative=cfg.audit.use_narrative_excerpts,
            frontier_note=frontier_note,
        )
        system_prompt = system
        batch_users.append(user)
        batch_meta.append({
            "disease_id": did,
            "disease_name": row["disease_name"],
            "user_prompt": user,
        })

    # Batch-call LLM
    if batch_users:
        raw_outs = llm.chat_batch(system_prompt or "", batch_users, task="extraction")
        patient_text = compact_patient_evidence(patient_state)
        for raw, meta in zip(raw_outs, batch_meta):
            disease_card = kg_index.disease_cards.get(meta["disease_id"], "")
            if isinstance(raw, (dict, list)):
                import json as _json
                raw_str = _json.dumps(raw)
            else:
                raw_str = str(raw)
            parsed = parse_audit_output(
                raw_str, meta["disease_id"], meta["disease_name"],
                patient_text, disease_card, hpo=hpo,
            )
            parsed["source"] = "llm_audit"
            parsed["user_prompt"] = meta["user_prompt"]
            parsed["raw_response"] = raw_str
            results.append(parsed)

    return results


def apply_audit_multipliers(
    ranked_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Apply audit multipliers to composite scores to produce adjusted ranking."""
    mult_by_id = {r["disease_id"]: r["multiplier"] for r in audit_results}
    cat_by_id = {r["disease_id"]: r["plausibility"] for r in audit_results}
    src_by_id = {r["disease_id"]: r.get("source", "") for r in audit_results}
    validation_by_id = {r["disease_id"]: float(r.get("validation_raw_score", 0.0)) for r in audit_results}
    validation_src_by_id = {r["disease_id"]: r.get("validation_source", "") for r in audit_results}

    df = ranked_df.copy()
    df["audit_plausibility"] = df["disease_id"].map(cat_by_id).fillna("not_audited")
    df["audit_multiplier"] = df["disease_id"].map(mult_by_id).fillna(1.0)
    df["audit_source"] = df["disease_id"].map(src_by_id).fillna("")
    df["llm_validation_raw_score"] = df["disease_id"].map(validation_by_id).fillna(0.0).astype(float)
    max_abs_validation = float(df["llm_validation_raw_score"].abs().max()) if len(df) else 0.0
    if max_abs_validation > 0:
        df["llm_validation_score"] = df["llm_validation_raw_score"] / max_abs_validation
    else:
        df["llm_validation_score"] = 0.0
    df["llm_validation_source"] = df["disease_id"].map(validation_src_by_id).fillna("")
    df["adjusted_score"] = (
        df["total_score"] * df["audit_multiplier"]
        + LLM_VALIDATION_WEIGHT * df["llm_validation_score"]
    )
    df = df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    df["adjusted_rank"] = df.index + 1
    return df
