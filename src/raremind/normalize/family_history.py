"""Precision-first post-processing for LLM-extracted family history."""
from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


FAMILY_HISTORY_SANITIZER_VERSION = "family-history-evidence-grounded-v3"

_CLINICAL_CONTENT_FIELDS = ("diseases", "phenotypes", "genes")
_EMPTY_TEXT = {
    "", "none", "null", "unknown", "not known", "not reported",
    "not documented", "n/a", "na", "nil", "[]", "{}",
}

_RELATION_ALIASES = {
    "mother": ("mother", "mom", "mum", "mama"),
    "father": ("father", "dad", "daddy", "papa"),
    "brother": ("brother",),
    "sister": ("sister",),
    "son": ("son",),
    "daughter": ("daughter",),
    "cousin": ("cousin",),
    "uncle": ("uncle",),
    "aunt": ("aunt",),
    "grandfather": ("grandfather", "grandpa", "granddad"),
    "grandmother": ("grandmother", "grandma", "granny", "nana"),
    "other": (
        "parent", "sibling", "siblings", "relative", "relatives",
        "family member", "family members", "niece", "nephew",
    ),
}
_ALIAS_TO_RELATION = {
    alias: relation
    for relation, aliases in _RELATION_ALIASES.items()
    for alias in aliases
}
_ALL_RELATION_TERMS = sorted({*_ALIAS_TO_RELATION, "parents"}, key=len, reverse=True)
_RELATION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in _ALL_RELATION_TERMS) + r")\b",
    flags=re.I,
)
_FAMILY_SECTION_PATTERN = re.compile(
    r"\b(?:family\s+(?:medical\s+)?history|family\s+hx|fhx|pedigree)\b",
    flags=re.I,
)
_UNAFFECTED_PATTERN = re.compile(
    r"\b(?:unaffected|healthy|asymptomatic|tested\s+negative|negative\s+for|"
    r"without\s+(?:features|symptoms|the\s+condition)|"
    r"no\s+(?:history|features|symptoms|evidence)\s+of)\b",
    flags=re.I,
)
_RELATIVE_MEDICAL_PREDICATE = (
    r"(?:has|had|have|with|affected\s+by|diagnosed\s+with|developed|"
    r"carries|carrier\s+of|tested\s+(?:positive|negative)|positive\s+for|"
    r"negative\s+for|history\s+of|died\s+(?:from|of)|unaffected|healthy|"
    r"asymptomatic|without|no\s+(?:history|features|symptoms|evidence))"
)
_SUPPORT_STOPWORDS = {
    "a", "an", "and", "the", "of", "with", "in", "on", "for",
    "syndrome", "disease", "disorder", "condition", "type", "autosomal",
    "dominant", "recessive", "related",
}


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_match(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize_space(value).casefold()).strip()


def _has_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _EMPTY_TEXT
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_meaningful_value(item) for item in value)
    return bool(value)


def has_family_clinical_content(entry: Dict[str, Any]) -> bool:
    return any(_has_meaningful_value(entry.get(field)) for field in _CLINICAL_CONTENT_FIELDS)


def _sentence_units(text: str) -> List[str]:
    units = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    return [_normalize_space(unit) for unit in units if _normalize_space(unit)]


def family_history_candidate_context(note: str, *, max_characters: int = 6000) -> str:
    """Return conservative candidate passages, or empty when none exist."""
    note = str(note or "")
    if not note.strip():
        return ""
    units = _sentence_units(note)
    selected: set[int] = set()
    for index, unit in enumerate(units):
        if _RELATION_PATTERN.search(unit):
            selected.update(range(max(0, index - 1), min(len(units), index + 2)))
        if _FAMILY_SECTION_PATTERN.search(unit):
            selected.update(range(index, min(len(units), index + 8)))
    if not selected:
        return ""
    return "\n".join(units[index] for index in sorted(selected))[:max_characters].rstrip()


def _relation_targets(value: Any) -> List[str]:
    normalized = _normalize_match(value)
    if normalized == "parents":
        return ["mother", "father"]
    if normalized in _ALIAS_TO_RELATION:
        return [_ALIAS_TO_RELATION[normalized]]
    if normalized in _RELATION_ALIASES:
        return [normalized]
    return []


def _relation_supported(relation: str, evidence: str) -> bool:
    evidence_norm = _normalize_match(evidence)
    aliases = list(_RELATION_ALIASES.get(relation, ()))
    if relation in {"mother", "father"}:
        aliases.append("parents")
    return any(re.search(rf"\b{re.escape(alias)}\b", evidence_norm) for alias in aliases)


def _has_relative_medical_assertion(relation: str, evidence: str) -> bool:
    aliases = list(_RELATION_ALIASES.get(relation, ()))
    if relation in {"mother", "father"}:
        aliases.append("parents")
    expression = r"(?:" + "|".join(re.escape(alias) for alias in aliases) + r")"
    patterns = (
        rf"\b{expression}\b.{{0,90}}\b{_RELATIVE_MEDICAL_PREDICATE}\b",
        rf"\b{_RELATIVE_MEDICAL_PREDICATE}\b.{{0,90}}\b{expression}\b",
        rf"\b{expression}\b.{{0,160}}\b(?:he|she|they)\s+{_RELATIVE_MEDICAL_PREDICATE}\b",
        rf"\b{expression}(?:'s|s')\b.{{0,90}}",
        # Common terse documentation: "Mother: cardiomyopathy" or
        # "Grandma - breast cancer".
        rf"\b{expression}\b\s*[:\-]\s*\S+",
    )
    return any(re.search(pattern, evidence, flags=re.I) for pattern in patterns)


def _supported_item(item: Any, evidence: str) -> bool:
    item_norm = _normalize_match(item)
    evidence_norm = _normalize_match(evidence)
    if not item_norm or item_norm in _EMPTY_TEXT:
        return False
    if re.search(rf"\b{re.escape(item_norm)}\b", evidence_norm):
        return True
    tokens = [
        token for token in item_norm.split()
        if token not in _SUPPORT_STOPWORDS and len(token) > 1
    ]
    return bool(tokens) and all(
        re.search(rf"\b{re.escape(token)}\b", evidence_norm) for token in tokens
    )


def _supported_values(value: Any, evidence: str) -> List[Any]:
    values = value if isinstance(value, list) else ([value] if _has_meaningful_value(value) else [])
    return [copy.deepcopy(item) for item in values if _supported_item(item, evidence)]


def _evidence_is_verbatim(evidence: str, note: str) -> bool:
    evidence_norm = _normalize_space(evidence).casefold()
    note_norm = _normalize_space(note).casefold()
    return bool(evidence_norm) and evidence_norm in note_norm and len(evidence.split()) <= 40


def sanitize_family_history(
    entries: Any,
    *,
    note: str | None = None,
    return_audit: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Remove records unsupported by explicit, verbatim note evidence.

    Supplying ``note`` enables strict evidence grounding. Calls without the
    original note retain the earlier clinical-content filter for compatibility
    with already validated cached state.
    """
    source = entries if isinstance(entries, list) else []
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for raw_entry in source:
        if not isinstance(raw_entry, dict):
            reasons["invalid_entry"] += 1
            continue
        entry = copy.deepcopy(raw_entry)

        if note is None:
            if has_family_clinical_content(entry):
                kept.append(entry)
            else:
                removed.append(entry)
                reasons["no_clinical_content"] += 1
            continue

        evidence = _normalize_space(entry.get("evidence"))
        if not _evidence_is_verbatim(evidence, note):
            removed.append(entry)
            reasons["evidence_not_verbatim_or_too_long"] += 1
            continue
        targets = _relation_targets(entry.get("relation"))
        if not targets:
            removed.append(entry)
            reasons["invalid_relation"] += 1
            continue

        for relation in targets:
            candidate = copy.deepcopy(entry)
            candidate["relation"] = relation
            if not _relation_supported(relation, evidence):
                removed.append(candidate)
                reasons["relation_not_in_evidence"] += 1
                continue
            if not _has_relative_medical_assertion(relation, evidence):
                removed.append(candidate)
                reasons["no_explicit_relative_medical_assertion"] += 1
                continue

            for field in ("diseases", "affected_systems", "phenotypes", "genes"):
                candidate[field] = _supported_values(candidate.get(field), evidence)
            age = candidate.get("age")
            if _has_meaningful_value(age) and not _supported_item(age, evidence):
                candidate["age"] = None

            affected = candidate.get("affected")
            if not isinstance(affected, bool):
                removed.append(candidate)
                reasons["affected_not_boolean"] += 1
                continue
            if affected:
                if not has_family_clinical_content(candidate):
                    removed.append(candidate)
                    reasons["no_supported_clinical_content"] += 1
                    continue
            else:
                if not _UNAFFECTED_PATTERN.search(evidence):
                    removed.append(candidate)
                    reasons["unaffected_not_explicit"] += 1
                    continue
                candidate["diseases"] = []
                candidate["phenotypes"] = []

            candidate["evidence"] = evidence
            kept.append(candidate)

    deduplicated: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, bool]] = set()
    for entry in kept:
        key = (
            str(entry.get("relation", "")),
            _normalize_match(entry.get("evidence", "")),
            bool(entry.get("affected")),
        )
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        deduplicated.append(entry)

    # One collective ``parents`` input may legitimately expand into two output
    # records, so input_count - retained_count is not a removal count.
    removed_count = len(removed) + reasons["invalid_entry"]
    audit = {
        "sanitizer_version": FAMILY_HISTORY_SANITIZER_VERSION,
        "input_count": len(source),
        "retained_count": len(deduplicated),
        "removed_count": removed_count,
        "removal_reasons": dict(sorted(reasons.items())),
        "removed_no_clinical_content_count": reasons["no_clinical_content"],
        "removed_unsupported_affected_count": removed_count,
        "removed_invalid_count": reasons["invalid_entry"],
        "removed_no_clinical_content_entries": removed,
        "removed_unsupported_affected_entries": removed,
    }
    return (deduplicated, audit) if return_audit else deduplicated
