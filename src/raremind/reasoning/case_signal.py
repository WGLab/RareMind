"""Stage 3.5: non-gating rare-disease/genetic case-signal assessment.

This module deliberately estimates whether the *case presentation* is
rare-disease-like.  It does not estimate whether the leading diagnosis is
correct and it never changes candidate ranks or controls pipeline execution.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from string import Template
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs" / "prompts" / "reasoning" / "case_signal_interpretation.md"
)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read either mapping-style or attribute-style configuration locally."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def score_case_signal_features(
    features: Mapping[str, float | int], config: Any = None
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Apply the provisional v2 heuristic to an already-computed feature row."""
    cfg = config or {}
    raw_top_reference = float(_cfg_get(cfg, "raw_top_reference", 8.0))
    max_ic_reference = float(_cfg_get(cfg, "max_ic_reference", 10.0))
    high_ic_count_reference = float(_cfg_get(cfg, "high_ic_count_reference", 3.0))
    organ_system_reference = float(_cfg_get(cfg, "organ_system_reference", 8.0))
    score_shape = float(np.mean([
        _clip(float(features["top1_z_score"]) / 5.0),
        _clip(float(features["rank1_rank10_relative_gap"]) / 0.25),
        _clip(1.0 - max(0.0, float(features["candidates_within_5_percent"]) - 1.0) / 9.0),
        _clip(float(features["raw_top_score"]) / max(raw_top_reference, 1e-9)),
    ]))
    hpo_specificity = float(np.mean([
        _clip(float(features["max_hpo_ic"]) / max(max_ic_reference, 1e-9)),
        _clip(float(features["high_ic_hpo_count"]) / max(high_ic_count_reference, 1e-9)),
        _clip(float(features["organ_system_count"]) / max(organ_system_reference, 1e-9)),
    ]))
    family_score = float(features["family_history_ordinal"]) / 3.0
    weights = _cfg_get(cfg, "group_weights", {}) or {}
    w_shape = float(_cfg_get(weights, "score_shape", 0.30))
    w_hpo = float(_cfg_get(weights, "hpo_specificity", 0.65))
    w_family = float(_cfg_get(weights, "family_history", 0.05))
    weight_total = w_shape + w_hpo + w_family
    signal = (
        w_shape * score_shape + w_hpo * hpo_specificity + w_family * family_score
    ) / max(weight_total, 1e-9)
    return 100.0 * _clip(signal), {
        "score_shape": 100.0 * score_shape,
        "hpo_specificity": 100.0 * hpo_specificity,
        "family_history": 100.0 * family_score,
    }, {
        "raw_top_score": raw_top_reference,
        "max_hpo_ic": max_ic_reference,
        "high_ic_hpo_count": high_ic_count_reference,
        "organ_system_count": organ_system_reference,
    }


def _family_ordinal(family_history: Any) -> tuple[int, dict[str, Any]]:
    entries = family_history if isinstance(family_history, list) else []
    affected = [entry for entry in entries if isinstance(entry, Mapping) and entry.get("affected") is True]
    serialized = json.dumps(entries, ensure_ascii=False, default=str).lower()
    consanguinity = (
        any(entry.get("consanguinity") is True for entry in entries if isinstance(entry, Mapping))
        or (
            "consanguin" in serialized
            and not re.search(r"\b(?:no|not|den(?:y|ies|ied)|without)\b.{0,30}\bconsanguin", serialized)
        )
    )
    recurrent_findings: Counter[str] = Counter()
    for entry in affected:
        for field in ("diseases", "phenotypes", "genes"):
            values = entry.get(field, []) or []
            if not isinstance(values, list):
                values = [values]
            recurrent_findings.update(
                str(value).strip().casefold() for value in values if str(value).strip()
            )
    recurrence = any(count >= 2 for count in recurrent_findings.values()) or "multiple affected" in serialized
    if consanguinity or recurrence:
        value = 3
    elif affected:
        value = 2
    elif entries:
        value = 1
    else:
        value = 0
    return value, {
        "documented_entries": len(entries),
        "affected_relatives": len(affected),
        "consanguinity_documented": consanguinity,
        "recurrence_documented": recurrence,
        "recurrent_findings": sorted(
            finding for finding, count in recurrent_findings.items() if count >= 2
        ),
    }


def _deterministic_explanation(result: dict[str, Any]) -> dict[str, Any]:
    f = result["features"]
    reasons: list[str] = []
    if f["top1_z_score"] >= 3:
        reasons.append("The leading Stage 3 candidate is well separated from the full score distribution.")
    elif f["top1_z_score"] < 1:
        reasons.append("The leading candidate is not clearly separated from the full score distribution.")
    if f["candidates_within_5_percent"] <= 3:
        reasons.append("Few candidates are crowded near the leading score.")
    elif f["candidates_within_5_percent"] >= 10:
        reasons.append("Many candidates have nearly indistinguishable scores, indicating a nonspecific match.")
    if f["max_hpo_ic"] >= 5 or f["high_ic_hpo_count"] > 0:
        reasons.append("The extracted phenotype profile contains specific HPO evidence.")
    if f["organ_system_count"] >= 2:
        reasons.append("The phenotype profile spans multiple organ-system branches.")
    if f["family_history_ordinal"] >= 2:
        reasons.append("The structured family history provides affected-relative or recurrence support.")
    elif f["family_history_ordinal"] == 0:
        reasons.append("No usable family-history support was documented; this is neutral for simplex or de novo disease.")
    return {
        "summary": (
            f"This presentation has {result['signal_category']} rare-disease/genetic signal "
            f"(uncalibrated score {result['confidence_score']:.1f}/100)."
        ),
        "reasons": reasons[:4],
        "caveat": "This score describes case signal only; it does not validate the leading diagnosis or establish a clinical diagnosis.",
        "source": "deterministic_fallback",
    }


def _llm_interpretation(llm: Any, result: dict[str, Any]) -> dict[str, Any] | None:
    if llm is None or not PROMPT_PATH.is_file():
        return None
    template = Template(PROMPT_PATH.read_text(encoding="utf-8"))
    user = template.substitute(
        prediction=result["predicted_class"],
        category=result["signal_category"],
        score=f"{result['confidence_score']:.1f}",
        features=json.dumps(result["features"], ensure_ascii=False, indent=2),
        components=json.dumps(result["component_scores"], ensure_ascii=False, indent=2),
    )
    response = llm.chat(
        "You explain a fixed, deterministic RareMind case-signal score. Do not change the score or infer a diagnosis.",
        user,
        task="extraction",
    )
    if not isinstance(response, Mapping):
        return None
    reasons = response.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "summary": str(response.get("summary", "")).strip(),
        "reasons": [str(reason).strip() for reason in reasons if str(reason).strip()][:4],
        "caveat": str(response.get("caveat", "")).strip(),
        "source": "llm_interpretation",
    }


def build_case_signal_assessment(
    *,
    patient_state: Any,
    ranked_df: pd.DataFrame,
    hpo: Any,
    config: Any = None,
    llm: Any = None,
) -> dict[str, Any]:
    """Return eight raw features, a fixed heuristic score, and a brief explanation."""
    if ranked_df is None or ranked_df.empty or "total_score" not in ranked_df:
        raise ValueError("Stage 3.5 requires a non-empty Stage 3 total_score vector")
    cfg = config or {}
    score_values = pd.to_numeric(ranked_df["total_score"], errors="coerce").dropna().to_numpy(float)
    if not len(score_values):
        raise ValueError("Stage 3.5 found no numeric Stage 3 scores")
    score_values = np.sort(score_values)[::-1]
    top = float(score_values[0])
    tenth = float(score_values[min(9, len(score_values) - 1)])
    spread = float(score_values.max() - score_values.min())
    standard_deviation = float(score_values.std(ddof=0))
    top1_z = (top - float(score_values.mean())) / standard_deviation if standard_deviation > 0 else 0.0
    relative_gap = (top - tenth) / spread if spread > 0 else 0.0
    crowd_fraction = float(_cfg_get(cfg, "crowding_fraction", 0.95))
    crowding = int(np.sum(score_values >= crowd_fraction * top)) if top > 0 else int(len(score_values))

    phenotypes = []
    for raw in getattr(patient_state, "normalized_hpo", []) or []:
        item = _plain(raw)
        if not isinstance(item, Mapping):
            continue
        if item.get("present", True) is not True or item.get("attribution", "patient") != "patient":
            continue
        phenotypes.append(item)
    deduplicated = {str(item.get("hpo_id")): item for item in phenotypes if item.get("hpo_id")}
    ic_values = [float(item.get("ic") or 0.0) for item in deduplicated.values()]
    high_ic_threshold = float(_cfg_get(cfg, "high_ic_threshold", 6.0))
    branches: set[str] = set()
    for hpo_id in deduplicated:
        branches.update(hpo.get_branches(hpo_id) if hpo is not None else set())
    branch_names = sorted(
        {name for hpo_id in deduplicated for name in (hpo.get_branch_names(hpo_id) if hpo is not None else [])}
    )
    family_ordinal, family_details = _family_ordinal(getattr(patient_state, "family_history", []))

    features = {
        "top1_z_score": float(top1_z),
        "rank1_rank10_relative_gap": float(relative_gap),
        "candidates_within_5_percent": crowding,
        "raw_top_score": top,
        "max_hpo_ic": max(ic_values, default=0.0),
        "high_ic_hpo_count": int(sum(value > high_ic_threshold for value in ic_values)),
        "organ_system_count": len(branches),
        "family_history_ordinal": family_ordinal,
    }
    confidence_score, component_scores, component_references = score_case_signal_features(
        features, cfg
    )
    weights = _cfg_get(cfg, "group_weights", {}) or {}
    w_shape = float(_cfg_get(weights, "score_shape", 0.30))
    w_hpo = float(_cfg_get(weights, "hpo_specificity", 0.65))
    w_family = float(_cfg_get(weights, "family_history", 0.05))
    genetic_threshold = float(_cfg_get(cfg, "genetic_threshold", 75.0))
    low_threshold = float(_cfg_get(cfg, "low_signal_threshold", 35.0))
    high_threshold = float(_cfg_get(cfg, "high_signal_threshold", 75.0))
    category = "high" if confidence_score >= high_threshold else "low" if confidence_score < low_threshold else "intermediate"
    result = {
        "stage": "3.5",
        "purpose": "non-gating rare-disease/genetic case-signal assessment",
        "predicted_class": "genetic_or_rare_disease" if confidence_score >= genetic_threshold else "nonspecific",
        "signal_category": category,
        "confidence_score": confidence_score,
        "score_is_calibrated_probability": False,
        "pipeline_continues_regardless_of_prediction": True,
        "features": features,
        "component_scores": component_scores,
        "feature_context": {
            "candidate_count": int(len(score_values)),
            "present_patient_hpo_count": len(deduplicated),
            "high_ic_threshold": high_ic_threshold,
            "organ_system_branches": branch_names,
            "family_history": family_details,
        },
        "method": {
            "version": "raremind-case-signal-v2-provisional",
            "weights": {"score_shape": w_shape, "hpo_specificity": w_hpo, "family_history": w_family},
            "component_references": component_references,
            "classification_threshold": genetic_threshold,
            "calibration_status": (
                "provisional control-informed heuristic; validate sensitivity and specificity "
                "on confirmed rare-disease and control cohorts"
            ),
        },
    }
    deterministic = _deterministic_explanation(result)
    interpretation = None
    if bool(_cfg_get(cfg, "llm_interpretation", True)):
        try:
            interpretation = _llm_interpretation(llm, result)
        except Exception as exc:
            deterministic["fallback_reason"] = str(exc)
    result["interpretation"] = interpretation or deterministic
    return result
