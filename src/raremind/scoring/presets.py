"""Named Stage 3 scoring presets.

These presets let the production pipeline run the exact Stage 3 formulas
selected from the debug sweeps without maintaining separate config files.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from raremind.core.config import AttrDict, cfg_get


SCORING_PRESETS: Dict[str, Dict[str, Any]] = {
    "exp076": {
        "description": "Soft phenotype + profile + semantic score with disease-size cap 30.",
        "scoring": {
            "use_specific_signal": True,
            "use_incongruity_match": False,
            "use_cooccurrence_pairs": True,
            "use_semantic_similarity": True,
            "use_phenotype_profile": True,
            "use_dynamic_profile_first": False,
            "use_phenotype_gene_prior": False,
            "use_semantic_sparse_rescue": False,
            "use_generic_guard": False,
            "use_anchor_cluster": False,
            "use_adaptive_weights": False,
            "phenotype_disease_size_norm_cap": 30,
            "semantic_sparse_rescue": {
                "enabled": False,
                "weight": 0.0,
                "max_patient_hpos": 8,
                "branch_concentration_threshold": 0.75,
                "noise_guard": False,
                "noise_hpo_threshold": 20,
                "low_ic_fraction_threshold": 0.5,
            },
            "weights": {
                "phenotype": 2.0,
                "genotype": 3.0,
                "family_evidence": 1.0,
                "demographics": 1.0,
                "cases": 1.0,
                "specific_signal": 0.5,
                "incongruity_match": 0.0,
                "cooccurrence_pairs": 0.5,
                "semantic_similarity": 1.0,
                "phenotype_profile": 4.0,
                "dynamic_profile_first": 0.0,
                "phenotype_gene_prior": 0.0,
                "semantic_sparse_rescue": 0.0,
                "generic_guard": 0.0,
                "anchor_cluster": 0.0,
            },
        },
    },
    "exp199": {
        "description": "exp076 plus sparse/branch-concentrated semantic rescue.",
        "scoring": {
            "use_specific_signal": True,
            "use_incongruity_match": False,
            "use_cooccurrence_pairs": True,
            "use_semantic_similarity": True,
            "use_phenotype_profile": True,
            "use_dynamic_profile_first": False,
            "use_phenotype_gene_prior": False,
            "use_semantic_sparse_rescue": True,
            "use_generic_guard": False,
            "use_anchor_cluster": False,
            "use_adaptive_weights": False,
            "phenotype_disease_size_norm_cap": 30,
            "semantic_sparse_rescue": {
                "enabled": True,
                "weight": 1.0,
                "max_patient_hpos": 8,
                "branch_concentration_threshold": 0.75,
                "noise_guard": False,
                "noise_hpo_threshold": 20,
                "low_ic_fraction_threshold": 0.5,
            },
            "weights": {
                "phenotype": 2.0,
                "genotype": 3.0,
                "family_evidence": 1.0,
                "demographics": 1.0,
                "cases": 1.0,
                "specific_signal": 0.5,
                "incongruity_match": 0.0,
                "cooccurrence_pairs": 0.5,
                "semantic_similarity": 1.0,
                "phenotype_profile": 4.0,
                "dynamic_profile_first": 0.0,
                "phenotype_gene_prior": 0.0,
                "semantic_sparse_rescue": 1.0,
                "generic_guard": 0.0,
                "anchor_cluster": 0.0,
            },
        },
    },
    "exp290": {
        "description": "Final full phenotype formula: exp076 plus generic guard and anchor-cluster evidence quality.",
        "scoring": {
            "use_specific_signal": True,
            "use_incongruity_match": False,
            "use_cooccurrence_pairs": True,
            "use_semantic_similarity": True,
            "use_phenotype_profile": True,
            "use_dynamic_profile_first": False,
            "use_phenotype_gene_prior": False,
            "use_semantic_sparse_rescue": False,
            "use_generic_guard": True,
            "use_anchor_cluster": True,
            "use_adaptive_weights": False,
            "phenotype_disease_size_norm_cap": 30,
            "semantic_sparse_rescue": {
                "enabled": False,
                "weight": 0.0,
                "max_patient_hpos": 8,
                "branch_concentration_threshold": 0.75,
                "noise_guard": False,
                "noise_hpo_threshold": 20,
                "low_ic_fraction_threshold": 0.5,
            },
            "weights": {
                "phenotype": 2.0,
                "genotype": 3.0,
                "family_evidence": 1.0,
                "demographics": 1.0,
                "cases": 1.0,
                "specific_signal": 0.5,
                "incongruity_match": 0.0,
                "cooccurrence_pairs": 0.5,
                "semantic_similarity": 1.0,
                "phenotype_profile": 3.75,
                "dynamic_profile_first": 0.0,
                "phenotype_gene_prior": 0.0,
                "semantic_sparse_rescue": 0.0,
                "generic_guard": 1.0,
                "anchor_cluster": 0.5,
            },
        },
    },
}


def available_scoring_presets() -> list[str]:
    return sorted(SCORING_PRESETS)


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def apply_scoring_preset(cfg: AttrDict, setting: str | None) -> AttrDict:
    """Apply a named scoring preset to a loaded config object in-place."""
    if setting is None:
        setting = cfg_get(cfg_get(cfg, "scoring", {}), "setting", None)
    if not setting:
        return cfg

    normalized = str(setting).strip().lower()
    if normalized in {"", "default", "none"}:
        cfg.project["setting"] = "default"
        return cfg
    if normalized not in SCORING_PRESETS:
        allowed = ", ".join(available_scoring_presets())
        raise ValueError(f"Unknown scoring setting '{setting}'. Available settings: {allowed}")

    cfg_dict = cfg.to_dict() if isinstance(cfg, AttrDict) else dict(cfg)
    preset = SCORING_PRESETS[normalized]
    _deep_update(cfg_dict, {"scoring": preset["scoring"]})
    cfg_dict.setdefault("project", {})
    cfg_dict["project"]["setting"] = normalized
    cfg_dict["project"]["setting_description"] = cfg_get(preset, "description", "")
    cfg_dict.setdefault("scoring", {})
    cfg_dict["scoring"]["setting"] = normalized
    return AttrDict(cfg_dict)
