"""Pydantic data classes representing per-patient pipeline state.

Kept compatible with the original rare_dx_mcp state module; new fields are
additive so existing code that consumes these objects still works.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Phenotype representations
# ---------------------------------------------------------------
class PhenotypeMention(BaseModel):
    mention: str
    attribution: str
    onset: Optional[str] = None
    evidence: Optional[str] = None


class NormalizedPhenotype(BaseModel):
    hpo_id: str
    hpo_name: str
    attribution: str
    score: float
    source: str
    mention: str
    onset: Optional[str] = None
    evidence: Optional[str] = None
    ic: Optional[float] = None
    present: bool = True  # True for patient-present, False for negated


# ---------------------------------------------------------------
# Demographics / testing / genetics
# ---------------------------------------------------------------
class Demographics(BaseModel):
    age: Dict[str, Any] = Field(default_factory=dict)
    sex: Dict[str, Any] = Field(default_factory=dict)
    ethnicity: Dict[str, Any] = Field(default_factory=dict)


class TestingInfo(BaseModel):
    tests: List[Dict[str, Any]] = Field(default_factory=list)
    gene_tests: List[Dict[str, Any]] = Field(default_factory=list)


class VcfVariant(BaseModel):
    gene: str
    hgvs: Optional[str] = None
    consequence: Optional[str] = None
    clinvar: Optional[str] = None
    gnomad_af: Optional[float] = None
    score: Optional[float] = None


class VcfSummary(BaseModel):
    variants: List[VcfVariant] = Field(default_factory=list)
    gene_scores: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------
# Candidates / ranking
# ---------------------------------------------------------------
class CandidateDisease(BaseModel):
    disease_id: str
    score_hint: float = 0.0
    degree_weight: float = 0.0
    matched_hpo: List[str] = Field(default_factory=list)
    hpo_degree: int = -1
    matched_genes: List[str] = Field(default_factory=list)


class RankedDisease(BaseModel):
    disease_id: str
    score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------
# Temporal / incongruity (NEW in RareMind)
# ---------------------------------------------------------------
class TemporalView(BaseModel):
    earliest_features: List[str] = Field(default_factory=list)
    onset_ordering: List[Dict[str, Any]] = Field(default_factory=list)
    age_at_presentation: Optional[str] = None


class IncongruityInfo(BaseModel):
    dominant_branch: Optional[str] = None
    dominant_branch_name: Optional[str] = None
    branch_profile: Dict[str, float] = Field(default_factory=dict)
    incongruous_phenotypes: List[Dict[str, Any]] = Field(default_factory=list)
    overall_incongruity_strength: str = "none"  # "strong" | "moderate" | "weak" | "none"


# ---------------------------------------------------------------
# Full patient case state
# ---------------------------------------------------------------
class PatientCaseState(BaseModel):
    case_id: str
    note_path: Optional[str] = None
    note_text: Optional[str] = None
    image_path: Optional[str] = None
    vcf_path: Optional[str] = None
    free_hpo_path: Optional[str] = None

    # extracted
    phenotype_mentions_text: List[Dict[str, Any]] = Field(default_factory=list)
    phenotype_mentions_vision: List[Dict[str, Any]] = Field(default_factory=list)
    phenotype_mentions_free_hpo: List[Dict[str, Any]] = Field(default_factory=list)
    demographics: Any = Field(default_factory=dict)
    family_history: Any = Field(default_factory=list)
    testing: Any = Field(default_factory=list)
    gene_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    vcf_summary: List[Dict[str, Any]] = Field(default_factory=list)

    # normalized + derived
    normalized_hpo: List[NormalizedPhenotype] = Field(default_factory=list)
    family_ctx: Any = None
    inheritance_prior: Dict[str, float] = Field(default_factory=dict)
    temporal_view: TemporalView = Field(default_factory=TemporalView)
    incongruity: IncongruityInfo = Field(default_factory=IncongruityInfo)

    # retrieval / ranking
    candidates: List[CandidateDisease] = Field(default_factory=list)
    public_disease_cases: Any = None
    public_gene_cases: Any = None

    # per-stage outputs
    ranked: Any = None                       # DataFrame after Stage 3
    case_signal: Dict[str, Any] = Field(default_factory=dict)  # Stage 3.5, non-gating
    frontier_output: Dict[str, Any] = Field(default_factory=dict)
    audit_results: List[Dict[str, Any]] = Field(default_factory=list)
    ranking_after_audit: Any = None          # DataFrame after Stage 5
    pairwise_results_subtype: List[Dict[str, Any]] = Field(default_factory=list)
    pairwise_results_group: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_subtype: Any = None             # DataFrame after Stage 7
    reranked_group: Any = None
    reconciled: Dict[str, Any] = Field(default_factory=dict)
    scorecard: Dict[str, Any] = Field(default_factory=dict)
    group_scorecard: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
