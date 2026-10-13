from __future__ import annotations

import json
import re
import logging, torch
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from raremind.core.config import AttrDict, cfg_get, retrieval_initial_top_k
from raremind.core.logging import setup_logger
from raremind.core.state import (
    PatientCaseState, NormalizedPhenotype, TemporalView, IncongruityInfo,
)
from raremind.core.utils import ensure_dir, write_json

from raremind.llm.vllm_client import VllmClient
from raremind.llm.vllm_vision_client import VllmVisionClient
from raremind.llm.vision_api_client import ApiVisionClient

from raremind.agents.text_agents import (
    run_phenotype_extractor_batch,
    run_demographics_extractor_batch,
    run_family_history_extractor_batch,
    run_testing_extractor_batch,
    run_gene_mentions_extractor_batch,
    run_stage1_text_extractors_batch,
)
from raremind.agents.vision_agents import (
    run_vision_extractor_batch,
    filter_vision_against_text,
)

from raremind.normalize.biolord_embedder import BioLordEmbedder
from raremind.normalize.hpo_ontology import HpoOntology
from raremind.normalize.normalizers import HpoNormalizer
from raremind.normalize.mondo_normalizer import MondoNormalizer
from raremind.normalize.temporal_parser import build_temporal_view
from raremind.normalize.inheritance_inference import infer_inheritance_prior
from raremind.normalize.family_history import sanitize_family_history
from raremind.normalize.incongruity_detector import detect_incongruity
from raremind.normalize.disease_id_mapper import DiseaseIdMapper

from raremind.kg.kg_loader import load_kg, load_hierarchy
from raremind.kg.kg_precompute import precompute_kg_index, KGIndex
from raremind.genomics.adapters import discover_genomics_result, load_genomics_results
from raremind.pipeline.vision_prefetch import release_vision_client

from raremind.retrieval.hpo_retriever import retrieve_by_hpo
from raremind.retrieval.gene_retriever import retrieve_by_gene
from raremind.retrieval.cooccurrence_retriever import retrieve_by_cooccurrence
from raremind.retrieval.pubcase_finder import search_PubCaseFinder

from raremind.scoring.composite_ranker import score_candidates

from raremind.frontier.client import FrontierClient
from raremind.frontier.consultation import run_frontier_consultation

from raremind.reasoning.audit import run_audit_batch, apply_audit_multipliers
from raremind.reasoning.pairwise import run_pairwise_batch
from raremind.reasoning.rank_centrality import aggregate_rank
from raremind.reasoning.reconciliation import reconcile
from raremind.reasoning.final_fusion import apply_final_fusion
from raremind.reasoning.scorecard import (
    build_scorecard, build_group_scorecard, collapse_reconciled_groups,
    format_scorecard_text, build_rank_trajectory,
)
from raremind.reasoning.case_signal import build_case_signal_assessment
from raremind.reasoning.report_pdf import render_stage9_pdf
from raremind.reasoning.next_steps import (
    build_next_step_recommendations,
    RECOMMENDATION_COLUMNS,
)

logger = setup_logger("raremind")


SOURCE_PRIORITY = {
    "text": 0,
    "free_hpo": 1,
    "vision": 2,
}


class RareMindHost:
    _STAGE8_RESUME_FILES = (
        "stage1_extraction.json",
        "stage2_patient_evidence.json",
        "stage3_composite_ranking.tsv",
        "stage5_audit_results.json",
        "stage6_pairwise_results_subtype.json",
        "stage8_reconciled.json",
        "stage8_final_fusion.tsv",
    )

    def __init__(self, cfg: AttrDict):
        self.cfg = cfg
        self.kg: Dict[str, Dict[str, Any]] = {}
        self.kg_index: Optional[KGIndex] = None
        self.hpo: Optional[HpoOntology] = None
        self.hpo_normalizer: Optional[HpoNormalizer] = None
        self.mondo_normalizer: Optional[MondoNormalizer] = None
        self.embedder: Optional[BioLordEmbedder] = None
        self.text_llm: Optional[VllmClient] = None
        self.vision_llm: Optional[Any] = None
        self.frontier: Optional[FrontierClient] = None
        self.disease_mapper: Optional[DiseaseIdMapper] = None
        self.hierarchy: Dict[str, Any] = {}

        self.prompt_dir = str(Path("configs/prompts"))

    # ---------------------------------------------------------------
    # One-time setup
    # ---------------------------------------------------------------
    def load(self) -> None:
        """Load KG, ontology, build indexes, load LLMs."""
        logger.info("=== RareMind host load ===")

        # Embedder (lazy-loaded model)
        self.embedder = BioLordEmbedder(
            model_name=self.cfg.normalization.embed_model,
            cache_dir=self.cfg.paths.cache_dir,
        )

        # HPO ontology
        self.hpo = HpoOntology(self.cfg.paths.hpo_obo)

        # KG
        self.kg = load_kg(self.cfg.paths.kg_path)
        self.hpo.compute_ic_from_kg(self.kg)

        # Precompute indexes
        self.kg_index = precompute_kg_index(self.kg, self.hpo)

        # Hierarchy
        if self.cfg.paths.hierarchy:
            self.hierarchy = load_hierarchy(self.cfg.paths.hierarchy)
        self._apply_hierarchy_groups()

        # HPO normalizer
        self.hpo_normalizer = HpoNormalizer(
            self.hpo, self.embedder,
            similarity_threshold=self.cfg.normalization.similarity_threshold,
        )
        self.hpo_normalizer.build_index()

        # Mondo normalizer (optional)
        if self.cfg.paths.full_mondo:
            self.mondo_normalizer = MondoNormalizer(
                self.cfg.paths.full_mondo, self.embedder,
                similarity_threshold=self.cfg.normalization.similarity_threshold,
            )
            self.mondo_normalizer.load()

        # Disease ID mapper
        self.disease_mapper = DiseaseIdMapper(
            omim2mondo_path=self.cfg.paths.omim2path,
            orphanet2mondo_path=self.cfg.paths.orphanet2path,
        )

        # Text LLM (required for Stage 1, 5, 6, 8)
        tcfg = self.cfg.models.text_llm
        self.text_llm = VllmClient(
            model_path=tcfg.model_name,
            temperature=tcfg.temperature,
            max_tokens=tcfg.max_tokens,
            tensor_parallel_size=torch.cuda.device_count(),#tcfg.tensor_parallel_size,
            gpu_memory_utilization= tcfg.gpu_memory_utilization,
            max_batch_size=tcfg.max_batch_size,
            enable_thinking=getattr(tcfg, "thinking", False),
        )

        # Vision LLM (optional)
        # Defer loading until first use to save memory when no image is provided.

        # Frontier client
        if self.cfg.frontier.enabled:
            self.frontier = FrontierClient(
                provider=self.cfg.frontier.provider,
                model_name=self.cfg.frontier.model_name,
                api_key_env=self.cfg.frontier.api_key_env,
                api_base_url=self.cfg.frontier.api_base_url,
                temperature=self.cfg.frontier.temperature,
                max_tokens=self.cfg.frontier.max_tokens,
                timeout_seconds=self.cfg.frontier.timeout_seconds,
                local_llm=self.text_llm if self.cfg.frontier.provider == "local" else None,
            )
            logger.info(f"Frontier client ready: provider={self.cfg.frontier.provider}")

        logger.info("=== RareMind host load complete ===")

    def _apply_hierarchy_groups(self) -> None:
        """Apply hierarchy.json group assignments to the KG index.

        The hierarchy file is keyed by disease MONDO ID and stores the selected
        meaningful group/ancestor. Diseases missing from the file become
        singleton groups.
        """
        if not self.kg_index:
            return

        hierarchy = self.hierarchy or {}
        assigned = 0
        for did, disease_name in list(self.kg_index.disease_name.items()):
            record = hierarchy.get(did, {}) if isinstance(hierarchy, dict) else {}
            if isinstance(record, dict):
                group_id = str(record.get("group_id") or did).strip() or did
                group_name = str(record.get("group_name") or record.get("name") or "").strip()
            else:
                group_id = did
                group_name = ""

            if not group_name:
                group_name = self.kg_index.disease_name.get(
                    group_id,
                    disease_name if group_id == did else group_id,
                )

            self.kg_index.disease_group[did] = group_id
            self.kg_index.disease_name.setdefault(group_id, group_name)
            assigned += 1

        logger.info(f"Applied MONDO hierarchy groups for {assigned} diseases")

    def _ensure_group_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ranking outputs expose group_id and group_name columns."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "disease_id" not in df.columns:
            return df
        if not self.kg_index:
            return df

        out = df.copy()
        if "group_id" not in out.columns:
            out["group_id"] = ""
        if "group_name" not in out.columns:
            out["group_name"] = ""

        legacy_group = out["mondo_group_id"] if "mondo_group_id" in out.columns else pd.Series([""] * len(out), index=out.index)

        def _clean(value: Any) -> str:
            text = str(value or "").strip()
            return "" if text.lower() in {"nan", "none"} else text

        def _group_id(row: pd.Series) -> str:
            did = _clean(row.get("disease_id", ""))
            existing = _clean(row.get("group_id", ""))
            legacy = _clean(legacy_group.loc[row.name])
            return existing or legacy or self.kg_index.disease_group.get(did, "") or did

        out["group_id"] = out.apply(_group_id, axis=1)

        def _group_name(row: pd.Series) -> str:
            existing = _clean(row.get("group_name", ""))
            if existing:
                return existing
            gid = _clean(row.get("group_id", ""))
            did = _clean(row.get("disease_id", ""))
            if gid == did:
                return _clean(row.get("disease_name", "")) or self.kg_index.disease_name.get(did, did)
            return self.kg_index.disease_name.get(gid, gid)

        out["group_name"] = out.apply(_group_name, axis=1)
        return out

    def _ensure_vision_llm(self) -> None:
        if self.vision_llm is None:
            vcfg = self.cfg.models.vision_llm
            vision_runtime = getattr(self.cfg, "vision", {})
            provider = str(getattr(vision_runtime, "provider", "local")).lower()
            if provider in {"local", "vllm"}:
                self.vision_llm = VllmVisionClient(
                    model_path=vcfg.model_name,
                    temperature=vcfg.temperature,
                    max_tokens=vcfg.max_tokens,
                    tensor_parallel_size=torch.cuda.device_count(),#vcfg.tensor_parallel_size,
                    gpu_memory_utilization=vcfg.gpu_memory_utilization,
                    max_batch_size=vcfg.max_batch_size,
                )
            elif provider in {"openrouter", "openai"}:
                self.vision_llm = ApiVisionClient(
                    provider=provider,
                    model_name=getattr(vision_runtime, "model_name", vcfg.model_name),
                    api_key_env=getattr(vision_runtime, "api_key_env", "OPENROUTER_API_KEY"),
                    api_base_url=getattr(vision_runtime, "api_base_url", "https://openrouter.ai/api/v1"),
                    temperature=vcfg.temperature,
                    max_tokens=vcfg.max_tokens,
                    timeout_seconds=getattr(vision_runtime, "timeout_seconds", 120),
                    max_batch_size=vcfg.max_batch_size,
                )
            else:
                raise ValueError(f"Unknown vision provider: {provider}")

    def _release_vision_llm(self) -> None:
        if self.vision_llm is not None:
            release_vision_client(self.vision_llm)
            self.vision_llm = None

    def _build_extracted_entities(self, stage1_out: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten Stage 1 extractor outputs into an auditable entity list."""
        entities: List[Dict[str, Any]] = []
        list_fields = [
            ("text_phenotypes", "phenotype", "text"),
            ("vision_phenotypes", "phenotype", "vision"),
            ("free_hpo_mentions", "phenotype", "free_hpo"),
            ("family_history", "family_history", "text"),
            ("testing", "test", "text"),
            ("gene_mentions", "gene", "text"),
            ("vcf_summary", "gene", "genomics"),
        ]
        for field, entity_type, source in list_fields:
            for item in stage1_out.get(field, []) or []:
                if not isinstance(item, dict):
                    continue
                ent = dict(item)
                ent.setdefault("entity_type", entity_type)
                ent.setdefault("source", source)
                ent.setdefault("stage1_field", field)
                entities.append(ent)

        demographics = stage1_out.get("demographics")
        if isinstance(demographics, dict):
            for key in ("age", "sex", "ethnicity"):
                value = demographics.get(key)
                if value not in (None, "", {}, []):
                    entities.append({
                        "entity_type": "demographic",
                        "source": "text",
                        "stage1_field": "demographics",
                        "attribute": key,
                        "value": value,
                    })
        return entities

    def _deduplicate_normalized_hpo(
        self,
        items: List[NormalizedPhenotype],
    ) -> List[NormalizedPhenotype]:
        """Collapse repeated normalized HPO evidence before scoring.

        Present and negated mentions are kept separate because they represent
        different clinical assertions. For duplicates, prefer text/free-HPO over
        vision and then the higher normalizer confidence.
        """
        best: Dict[tuple[str, bool], NormalizedPhenotype] = {}

        def rank(item: NormalizedPhenotype) -> tuple[int, float, int]:
            source_rank = SOURCE_PRIORITY.get(str(item.source or ""), 9)
            evidence_len = len(re.sub(r"\s+", " ", str(item.evidence or "")).strip())
            return (source_rank, -float(item.score or 0.0), -evidence_len)

        for item in items:
            key = (str(item.hpo_id), bool(item.present))
            if key not in best or rank(item) < rank(best[key]):
                best[key] = item
        return list(best.values())

    # ---------------------------------------------------------------
    # Per-patient pipeline
    # ---------------------------------------------------------------
    def run_patient(
        self,
        case_id: str,
        note_path: Optional[str] = None,
        image_path: Optional[str] = None,
        vcf_path: Optional[str] = None,
        free_hpo_path: Optional[str] = None,
        vision_json_path: Optional[str] = None,
        stage1_json_path: Optional[str] = None,
        genomics_result_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        control: bool = False,
    ) -> Dict[str, Any]:
        """Run RareMind, optionally stopping after the Stage 3.5 case signal."""
        out_dir = ensure_dir(output_dir or f"outputs/{case_id}")
        logger.info(f"=== Running case {case_id} ===")

        state = PatientCaseState(
            case_id=case_id,
            note_path=note_path,
            image_path=image_path,
            vcf_path=vcf_path,
            free_hpo_path=free_hpo_path,
        )

        # =================== STAGE 1 ===================
        logger.info("--- Stage 1: Multi-modal extraction ---")
        stage1_out = self._stage1_extraction(
            state,
            note_path,
            image_path,
            free_hpo_path,
            vision_json_path=vision_json_path,
            stage1_json_path=stage1_json_path,
            genomics_result_path=genomics_result_path,
        )
        write_json(stage1_out, out_dir / "stage1_extraction.json")

        # =================== STAGE 2 ===================
        logger.info("--- Stage 2: Normalization + post-processing ---")
        self._stage2_normalization(state)
        self._save_patient_evidence(state, out_dir)

        # =================== STAGE 3 ===================
        logger.info("--- Stage 3: Candidate retrieval + composite scoring ---")
        ranked_df, cooccurrence_candidates = self._stage3_retrieval_scoring(state)
        ranked_df = self._ensure_group_columns(ranked_df)
        state.ranked = ranked_df
        ranked_df.to_csv(out_dir / "stage3_composite_ranking.tsv", sep="\t", index=False)

        # =================== STAGE 3.5 ===================
        logger.info("--- Stage 3.5: Non-gating case-signal assessment ---")
        case_signal_cfg = cfg_get(self.cfg, "case_signal", {})
        if bool(cfg_get(case_signal_cfg, "enabled", True)) or control:
            state.case_signal = build_case_signal_assessment(
                patient_state=state,
                ranked_df=ranked_df,
                hpo=self.hpo,
                config=case_signal_cfg,
                llm=self.text_llm,
            )
            write_json(state.case_signal, out_dir / "stage3_5_case_signal.json")

        if control:
            control_summary = {
                "case_id": case_id,
                "status": "complete_through_stage3_5",
                "control_mode": True,
                "completed_stages": [1, 2, 3, 3.5],
                "stopped_before_stage": 4,
                "case_signal": state.case_signal,
            }
            write_json(control_summary, out_dir / "control_run_summary.json")
            logger.info(
                "=== Control case %s done after Stage 3.5; reranking and later stages skipped ===",
                case_id,
            )
            return {
                "case_id": case_id,
                "status": "complete_through_stage3_5",
                "control_mode": True,
                "top": ranked_df.head(5).to_dict("records"),
                "output_dir": str(out_dir),
                "reconciled": {},
                "case_signal": state.case_signal,
                "next_steps": [],
            }

        # =================== STAGE 4 ===================
        logger.info("--- Stage 4: Frontier consultation (conditional) ---")
        frontier_output, frontier_flags = self._stage4_frontier(state, ranked_df, out_dir)
        state.frontier_output = frontier_output

        # =================== STAGE 5 ===================
        logger.info("--- Stage 5: Audit ---")
        audit_results, ranking_after_audit = self._stage5_audit(
            state, ranked_df, frontier_flags
        )
        ranking_after_audit = self._ensure_group_columns(ranking_after_audit)
        state.audit_results = audit_results
        state.ranking_after_audit = ranking_after_audit
        write_json(audit_results, out_dir / "stage5_audit_results.json")
        ranking_after_audit.to_csv(out_dir / "stage5_ranking_after_audit.tsv", sep="\t", index=False)

        # =================== STAGE 6 ===================
        logger.info("--- Stage 6: Pairwise adjudication ---")
        pairwise_subtype = self._stage6_pairwise(
            state, ranking_after_audit, audit_results, frontier_flags, track="subtype"
        )
        state.pairwise_results_subtype = pairwise_subtype
        write_json(pairwise_subtype, out_dir / "stage6_pairwise_results_subtype.json")

        # Group-level pairwise (on group IDs)
        group_ranked_df = self._aggregate_to_group(ranking_after_audit, frontier_flags)
        pairwise_group = self._stage6_pairwise(
            state, group_ranked_df, audit_results, frontier_flags, track="group",
        ) if len(group_ranked_df) > 1 else []
        state.pairwise_results_group = pairwise_group
        write_json(pairwise_group, out_dir / "stage6_pairwise_results_group.json")

        # =================== STAGE 7 ===================
        logger.info("--- Stage 7: Rank aggregation ---")
        reranked_subtype = aggregate_rank(
            ranking_after_audit, pairwise_subtype, self.cfg, track_name="subtype"
        )
        reranked_subtype = self._ensure_group_columns(reranked_subtype)
        state.reranked_subtype = reranked_subtype
        reranked_subtype.to_csv(out_dir / "stage7_reranked_subtype.tsv", sep="\t", index=False)

        reranked_group = aggregate_rank(
            group_ranked_df, pairwise_group, self.cfg, track_name="group"
        ) if len(group_ranked_df) > 1 else group_ranked_df
        reranked_group = self._ensure_group_columns(reranked_group)
        state.reranked_group = reranked_group
        reranked_group.to_csv(out_dir / "stage7_reranked_group.tsv", sep="\t", index=False)

        # =================== STAGE 8 ===================
        logger.info("--- Stage 8: Reconciliation ---")
        reconciled = reconcile(
            reranked_subtype, reranked_group, state, self.kg_index, self.cfg,
            llm=self.text_llm, prompt_dir=self.prompt_dir,
        )
        reconciled_df = reconciled.get("reconciled_df")
        if isinstance(reconciled_df, pd.DataFrame):
            reconciled_df = self._ensure_group_columns(reconciled_df)
            reconciled["reconciled_df"] = reconciled_df
            reconciled_df.to_csv(out_dir / "stage8_reconciled_ranking.tsv", sep="\t", index=False)

            final_df, final_ranking = apply_final_fusion(reconciled_df, self.cfg)
            final_df = self._ensure_group_columns(final_df)
            reconciled["final_df"] = final_df
            reconciled["final_ranking"] = final_ranking
            reconciled["final_top_subtype"] = final_ranking.get("top_subtype")
            final_df.to_csv(out_dir / "stage8_final_fusion.tsv", sep="\t", index=False)
        else:
            reconciled["final_ranking"] = {"enabled": False, "method": "unavailable", "top_subtype": None}
            reconciled["final_top_subtype"] = reconciled.get("top_subtype")

        state.reconciled = {
            "top_subtype": reconciled.get("top_subtype"),
            "top_group": reconciled.get("top_group"),
            "final_top_subtype": reconciled.get("final_top_subtype"),
            "final_ranking": reconciled.get("final_ranking"),
            "disagreement": reconciled.get("disagreement", False),
            "tiebreaker": reconciled.get("tiebreaker"),
            "method": reconciled.get("method", ""),
        }
        write_json(state.reconciled, out_dir / "stage8_reconciled.json")

        # =================== STAGE 9A / 9B ===================
        logger.info("--- Stage 9A: Group-level clinical synthesis ---")
        final_df_for_scorecard = (
            reconciled["final_df"] if isinstance(reconciled.get("final_df"), pd.DataFrame)
            else reconciled["reconciled_df"] if isinstance(reconciled.get("reconciled_df"), pd.DataFrame)
            else reranked_subtype
        )
        final_df_for_scorecard = self._ensure_group_columns(final_df_for_scorecard)
        if isinstance(reconciled.get("final_df"), pd.DataFrame):
            reconciled["final_df"] = final_df_for_scorecard
        elif isinstance(reconciled.get("reconciled_df"), pd.DataFrame):
            reconciled["reconciled_df"] = final_df_for_scorecard
        subtype_reconciled_for_group = (
            reconciled_df if isinstance(reconciled_df, pd.DataFrame) else reranked_subtype
        )
        subtype_reconciled_for_group = self._ensure_group_columns(subtype_reconciled_for_group)
        group_report_ranking = collapse_reconciled_groups(subtype_reconciled_for_group)
        group_report_ranking.to_csv(
            out_dir / "stage9a_group_ranking.tsv", sep="\t", index=False
        )
        group_scorecard = build_group_scorecard(
            state, group_report_ranking, subtype_reconciled_for_group, audit_results,
            reconciled, frontier_output,
            top_k=self.cfg.output.scorecard_top_k,
            kg=self.kg,
            llm=self.text_llm,
            pairwise_group=pairwise_group,
            pairwise_subtype=pairwise_subtype,
            synthesis_config=cfg_get(self.cfg, "stage9", {}),
        )
        state.group_scorecard = group_scorecard
        if self.cfg.output.save_scorecard_json:
            write_json(group_scorecard, out_dir / "stage9a_group_reasoning_report.json")
        if self.cfg.output.save_scorecard_txt:
            (out_dir / "stage9a_group_reasoning_report.txt").write_text(
                format_scorecard_text(group_scorecard).rstrip() + "\n", encoding="utf-8"
            )
        if bool(cfg_get(self.cfg.output, "save_scorecard_pdf", True)):
            render_stage9_pdf(
                group_scorecard, out_dir / "stage9a_group_reasoning_report.pdf",
                case_id=state.case_id, report_level="a",
            )

        logger.info("--- Stage 9B: Exact subtype-level clinical synthesis ---")
        scorecard = build_scorecard(
            state, final_df_for_scorecard, audit_results,
            reconciled, frontier_output,
            top_k=self.cfg.output.scorecard_top_k,
            kg=self.kg,
            llm=self.text_llm,
            pairwise_subtype=pairwise_subtype,
            synthesis_config=cfg_get(self.cfg, "stage9", {}),
        )
        state.scorecard = scorecard
        scorecard["case_signal"] = state.case_signal

        if self.cfg.output.save_scorecard_json:
            write_json(scorecard, out_dir / "stage9_reasoning_report.json")
            write_json(scorecard, out_dir / "stage9b_subtype_reasoning_report.json")
        if self.cfg.output.save_scorecard_txt:
            report_text = format_scorecard_text(scorecard)
            (out_dir / "stage9_reasoning_report.txt").write_text(
                report_text.rstrip() + "\n", encoding="utf-8"
            )
            (out_dir / "stage9b_subtype_reasoning_report.txt").write_text(
                report_text.rstrip() + "\n", encoding="utf-8"
            )
        if bool(cfg_get(self.cfg.output, "save_scorecard_pdf", True)):
            render_stage9_pdf(
                scorecard, out_dir / "stage9b_subtype_reasoning_report.pdf",
                case_id=state.case_id, report_level="b",
            )
        (out_dir / "stage9_reasoning_report.pdf").unlink(missing_ok=True)

        # Rank trajectory
        if self.cfg.output.save_rank_trajectory:
            traj = build_rank_trajectory(
                ranked_df, audit_results, reranked_subtype, reconciled, frontier_output
            )
            traj.to_csv(out_dir / "rank_trajectory.tsv", sep="\t", index=False)

        # =================== STAGE 10 ===================
        logger.info("--- Stage 10: Next-test and next-step recommendations ---")
        next_cfg = cfg_get(self.cfg, "next_steps", {})
        next_steps_payload: Dict[str, Any]
        recommendations = pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
        if bool(cfg_get(next_cfg, "enabled", True)):
            disease_top_k = int(cfg_get(next_cfg, "disease_top_k", 10))
            action_top_k = int(cfg_get(next_cfg, "action_top_k", 10))
            action_fields = cfg_get(
                next_cfg, "action_fields", ["testing", "initial_evaluations"]
            )
            cluster_threshold = float(
                cfg_get(next_cfg, "cluster_similarity_threshold", 0.90)
            )
            recommendations, _raw_actions, selected_groups, metadata = (
                build_next_step_recommendations(
                    final_df=final_df_for_scorecard,
                    kg=self.kg,
                    embedder=self.embedder,
                    disease_top_k=disease_top_k,
                    action_top_k=action_top_k,
                    fields=action_fields,
                    cluster_threshold=cluster_threshold,
                )
            )
            next_steps_payload = {
                "case_id": case_id,
                "status": "complete",
                "metadata": metadata,
                "source_disease_groups": selected_groups[
                    [
                        column for column in (
                            "final_rank", "group_id", "group_name",
                            "disease_id", "disease_name",
                        ) if column in selected_groups.columns
                    ]
                ].to_dict("records"),
                "recommendations": recommendations.to_dict("records"),
            }
        else:
            next_steps_payload = {
                "case_id": case_id,
                "status": "disabled",
                "metadata": {},
                "source_disease_groups": [],
                "recommendations": [],
            }

        # Stage 10 intentionally has one canonical artifact. Recommendation-level
        # provenance is already represented by the supporting group/disease fields;
        # run metadata is repeated on each row for a self-contained TSV.
        stage10_table = recommendations.copy()
        if stage10_table.empty:
            stage10_table = pd.DataFrame([
                {column: pd.NA for column in RECOMMENDATION_COLUMNS}
            ])
        stage10_metadata = next_steps_payload.get("metadata", {})
        context_columns = {
            "case_id": case_id,
            "status": next_steps_payload.get("status", "complete"),
            "disease_top_k": stage10_metadata.get("disease_top_k"),
            "action_top_k": stage10_metadata.get("action_top_k"),
            "source_fields": "|".join(stage10_metadata.get("source_fields", [])),
            "selected_disease_groups": stage10_metadata.get("selected_disease_groups"),
            "raw_action_mentions": stage10_metadata.get("raw_action_mentions"),
            "normalization_method": stage10_metadata.get("normalization_method"),
            "embedding_model": stage10_metadata.get("embedding_model"),
            "cluster_similarity_threshold": stage10_metadata.get(
                "cluster_similarity_threshold"
            ),
            "source_disease_groups_json": json.dumps(
                next_steps_payload.get("source_disease_groups", []),
                ensure_ascii=False,
                default=str,
            ),
        }
        for column, value in reversed(tuple(context_columns.items())):
            stage10_table.insert(0, column, value)
        for legacy_name in (
            "stage10_next_steps.json",
            "stage10_next_steps.txt",
            "stage10_next_step_evidence.tsv",
        ):
            (out_dir / legacy_name).unlink(missing_ok=True)
        stage10_table.to_csv(
            out_dir / "stage10_next_steps.tsv", sep="\t", index=False
        )

        logger.info(f"=== Case {case_id} done ===")

        return {
            "case_id": case_id,
            "top": scorecard.get("top_candidates", [])[:5],
            "output_dir": str(out_dir),
            "reconciled": state.reconciled,
            "case_signal": state.case_signal,
            "next_steps": next_steps_payload.get("recommendations", []),
        }

    @classmethod
    def can_resume_from_stage8(cls, output_dir: str | Path) -> bool:
        """Whether saved artifacts are sufficient to run only Stages 9–10."""
        folder = Path(output_dir)
        return all((folder / name).is_file() for name in cls._STAGE8_RESUME_FILES)

    @staticmethod
    def _read_saved_json(path: Path) -> Any:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)

    @classmethod
    def _valid_saved_stage9(cls, path: Path) -> Optional[Dict[str, Any]]:
        if not path.is_file():
            return None
        try:
            value = cls._read_saved_json(path)
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(value, dict):
            return None
        if not str(value.get("synthesis_report", "")).strip():
            return None
        cards = value.get("top_candidates")
        if not isinstance(cards, list) or not cards:
            return None
        return value

    def _save_stage10_from_final_ranking(
        self,
        *,
        case_id: str,
        final_df: pd.DataFrame,
        out_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Generate the single canonical Stage 10 TSV from a saved Final Rank."""
        next_cfg = cfg_get(self.cfg, "next_steps", {})
        recommendations = pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
        selected_groups = pd.DataFrame()
        metadata: Dict[str, Any] = {}
        status = "disabled"
        if bool(cfg_get(next_cfg, "enabled", True)):
            recommendations, _raw_actions, selected_groups, metadata = (
                build_next_step_recommendations(
                    final_df=final_df,
                    kg=self.kg,
                    embedder=self.embedder,
                    disease_top_k=int(cfg_get(next_cfg, "disease_top_k", 10)),
                    action_top_k=int(cfg_get(next_cfg, "action_top_k", 10)),
                    fields=cfg_get(
                        next_cfg, "action_fields", ["testing", "initial_evaluations"]
                    ),
                    cluster_threshold=float(
                        cfg_get(next_cfg, "cluster_similarity_threshold", 0.90)
                    ),
                )
            )
            status = "complete"

        source_columns = [
            column for column in (
                "final_rank", "group_id", "group_name", "disease_id", "disease_name",
            ) if column in selected_groups.columns
        ]
        source_groups = (
            selected_groups[source_columns].to_dict("records")
            if source_columns else []
        )
        table = recommendations.copy()
        if table.empty:
            table = pd.DataFrame([
                {column: pd.NA for column in RECOMMENDATION_COLUMNS}
            ])
        context_columns = {
            "case_id": case_id,
            "status": status,
            "disease_top_k": metadata.get("disease_top_k"),
            "action_top_k": metadata.get("action_top_k"),
            "source_fields": "|".join(metadata.get("source_fields", [])),
            "selected_disease_groups": metadata.get("selected_disease_groups"),
            "raw_action_mentions": metadata.get("raw_action_mentions"),
            "normalization_method": metadata.get("normalization_method"),
            "embedding_model": metadata.get("embedding_model"),
            "cluster_similarity_threshold": metadata.get("cluster_similarity_threshold"),
            "source_disease_groups_json": json.dumps(
                source_groups, ensure_ascii=False, default=str,
            ),
        }
        for column, value in reversed(tuple(context_columns.items())):
            table.insert(0, column, value)
        table.to_csv(out_dir / "stage10_next_steps.tsv", sep="\t", index=False)
        return recommendations.to_dict("records")

    def resume_patient_from_stage8(
        self,
        *,
        case_id: str,
        output_dir: str,
        note_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume a partially completed case without changing Stages 1–8.

        A valid Stage 9 JSON is reused and its text rendering is repaired when
        absent. Stage 10 is generated only when its canonical TSV is absent.
        """
        out_dir = Path(output_dir)
        if not self.can_resume_from_stage8(out_dir):
            missing = [
                name for name in self._STAGE8_RESUME_FILES
                if not (out_dir / name).is_file()
            ]
            raise FileNotFoundError(
                f"Case {case_id} cannot resume at Stage 9; missing: {', '.join(missing)}"
            )

        s1 = self._read_saved_json(out_dir / "stage1_extraction.json")
        s2 = self._read_saved_json(out_dir / "stage2_patient_evidence.json")
        note_text = ""
        if note_path and Path(note_path).is_file():
            note_text = Path(note_path).read_text(encoding="utf-8")
        note_text = note_text or str(s1.get("note_text", ""))
        if not note_text.strip():
            raise ValueError(
                f"Case {case_id} has no recoverable source note for Stage 9 resume"
            )

        family_history = sanitize_family_history(
            s2.get("family_history", s1.get("family_history", [])),
            note=note_text,
        )
        genes = s2.get("gene_evidence", {}) or {}
        phenotypes = list(s2.get("phenotypes", []))
        for negated in s2.get("negated", []):
            item = {**negated, "present": False}
            key = (item.get("hpo_id"), item.get("attribution"), False)
            existing = {
                (p.get("hpo_id"), p.get("attribution"), p.get("present", True))
                for p in phenotypes if isinstance(p, dict)
            }
            if key not in existing:
                phenotypes.append(item)

        state = PatientCaseState(case_id=case_id, note_path=note_path)
        state.note_text = note_text
        state.phenotype_mentions_text = s1.get("text_phenotypes", []) or []
        state.phenotype_mentions_vision = s1.get("vision_phenotypes", []) or []
        state.demographics = s2.get("demographics", s1.get("demographics", {})) or {}
        state.family_history = family_history
        state.testing = s1.get("testing", []) or []
        state.gene_mentions = genes.get("gene_mentions", s1.get("gene_mentions", [])) or []
        state.vcf_summary = genes.get("vcf_summary", s1.get("vcf_summary", [])) or []
        state.normalized_hpo = [
            NormalizedPhenotype(**item) for item in phenotypes if isinstance(item, dict)
        ]
        state.inheritance_prior = infer_inheritance_prior(family_history)
        state.temporal_view = TemporalView(**(s2.get("temporal_view", {}) or {}))
        state.incongruity = IncongruityInfo(**(s2.get("incongruity", {}) or {}))
        state.ranked = pd.read_csv(out_dir / "stage3_composite_ranking.tsv", sep="\t")

        case_signal_path = out_dir / "stage3_5_case_signal.json"
        if case_signal_path.is_file():
            state.case_signal = self._read_saved_json(case_signal_path)
        elif bool(cfg_get(cfg_get(self.cfg, "case_signal", {}), "enabled", True)):
            logger.info("--- Stage 3.5: Resume from saved Stage 3 ranking ---")
            state.case_signal = build_case_signal_assessment(
                patient_state=state,
                ranked_df=state.ranked,
                hpo=self.hpo,
                config=cfg_get(self.cfg, "case_signal", {}),
                llm=self.text_llm,
            )
            write_json(state.case_signal, case_signal_path)

        final_df = pd.read_csv(out_dir / "stage8_final_fusion.tsv", sep="\t")
        final_df = self._ensure_group_columns(final_df)
        if "final_rank" in final_df.columns:
            final_df = final_df.sort_values("final_rank", kind="stable").reset_index(drop=True)
        audit_results = self._read_saved_json(out_dir / "stage5_audit_results.json")
        pairwise = self._read_saved_json(out_dir / "stage6_pairwise_results_subtype.json")
        reconciled = self._read_saved_json(out_dir / "stage8_reconciled.json")
        frontier_path = out_dir / "stage4_frontier_output.json"
        frontier = self._read_saved_json(frontier_path) if frontier_path.is_file() else {}
        state.audit_results = audit_results
        state.pairwise_results_subtype = pairwise
        state.reconciled = reconciled
        state.frontier_output = frontier

        reconciled_ranking_path = out_dir / "stage8_reconciled_ranking.tsv"
        subtype_reconciled_for_group = (
            pd.read_csv(reconciled_ranking_path, sep="\t")
            if reconciled_ranking_path.is_file() else final_df.copy()
        )
        subtype_reconciled_for_group = self._ensure_group_columns(subtype_reconciled_for_group)
        group_df = collapse_reconciled_groups(subtype_reconciled_for_group)
        group_df.to_csv(out_dir / "stage9a_group_ranking.tsv", sep="\t", index=False)
        pairwise_group_path = out_dir / "stage6_pairwise_results_group.json"
        pairwise_group = (
            self._read_saved_json(pairwise_group_path) if pairwise_group_path.is_file() else []
        )
        state.reranked_group = group_df
        state.pairwise_results_group = pairwise_group

        stage9a_path = out_dir / "stage9a_group_reasoning_report.json"
        group_scorecard = self._valid_saved_stage9(stage9a_path)
        if (
            group_scorecard is not None
            and (group_scorecard.get("reconciled", {}) or {}).get("method")
            != "stage8_reconciled_group_name_collapse"
        ):
            logger.info("Case %s: rebuilding legacy Stage 9A with Stage 8 group collapse", case_id)
            group_scorecard = None
        if group_scorecard is None:
            logger.info("--- Stage 9A: Resume group synthesis from saved ranking ---")
            group_scorecard = build_group_scorecard(
                state, group_df, subtype_reconciled_for_group, audit_results, reconciled, frontier,
                top_k=self.cfg.output.scorecard_top_k,
                kg=self.kg,
                llm=self.text_llm,
                pairwise_group=pairwise_group,
                pairwise_subtype=pairwise,
                synthesis_config=cfg_get(self.cfg, "stage9", {}),
            )
            write_json(group_scorecard, stage9a_path)
        group_scorecard["case_signal"] = state.case_signal
        state.group_scorecard = group_scorecard
        if self.cfg.output.save_scorecard_json:
            write_json(group_scorecard, stage9a_path)
        if self.cfg.output.save_scorecard_txt:
            (out_dir / "stage9a_group_reasoning_report.txt").write_text(
                format_scorecard_text(group_scorecard).rstrip() + "\n", encoding="utf-8"
            )
        if bool(cfg_get(self.cfg.output, "save_scorecard_pdf", True)):
            render_stage9_pdf(
                group_scorecard, out_dir / "stage9a_group_reasoning_report.pdf",
                case_id=case_id, report_level="a",
            )

        stage9_path = out_dir / "stage9_reasoning_report.json"
        scorecard = self._valid_saved_stage9(stage9_path)
        if scorecard is None:
            logger.info("--- Stage 9: Resume from saved Stage 8 checkpoint ---")
            scorecard = build_scorecard(
                state, final_df, audit_results, reconciled, frontier,
                top_k=self.cfg.output.scorecard_top_k,
                kg=self.kg,
                llm=self.text_llm,
                pairwise_subtype=pairwise,
                synthesis_config=cfg_get(self.cfg, "stage9", {}),
            )
            write_json(scorecard, stage9_path)
        else:
            logger.info("Case %s: valid Stage 9 exists; reusing it", case_id)
        scorecard["case_signal"] = state.case_signal
        if self.cfg.output.save_scorecard_txt:
            text_path = out_dir / "stage9_reasoning_report.txt"
            if not text_path.is_file():
                text_path.write_text(
                    format_scorecard_text(scorecard).rstrip() + "\n", encoding="utf-8"
                )
            (out_dir / "stage9b_subtype_reasoning_report.txt").write_text(
                format_scorecard_text(scorecard).rstrip() + "\n", encoding="utf-8"
            )
        if self.cfg.output.save_scorecard_json:
            write_json(scorecard, out_dir / "stage9b_subtype_reasoning_report.json")
        if bool(cfg_get(self.cfg.output, "save_scorecard_pdf", True)):
            render_stage9_pdf(
                scorecard, out_dir / "stage9b_subtype_reasoning_report.pdf",
                case_id=case_id, report_level="b",
            )
        (out_dir / "stage9_reasoning_report.pdf").unlink(missing_ok=True)

        stage10_path = out_dir / "stage10_next_steps.tsv"
        if stage10_path.is_file():
            logger.info("Case %s: Stage 10 exists; reusing it", case_id)
            stage10_table = pd.read_csv(stage10_path, sep="\t")
            next_steps = stage10_table[
                stage10_table.get("action", pd.Series(index=stage10_table.index, dtype=object)).notna()
            ].to_dict("records")
        else:
            logger.info("--- Stage 10: Resume from saved Final Rank ---")
            next_steps = self._save_stage10_from_final_ranking(
                case_id=case_id, final_df=final_df, out_dir=out_dir,
            )

        logger.info("=== Case %s resumed without rerunning Stages 1–8 ===", case_id)
        return {
            "case_id": case_id,
            "top": scorecard.get("top_candidates", [])[:5],
            "output_dir": str(out_dir),
            "reconciled": reconciled,
            "case_signal": state.case_signal,
            "next_steps": next_steps,
        }
    # ===============================================================
    # Stage implementations
    # ===============================================================
    def _stage1_extraction(
        self,
        state: PatientCaseState,
        note_path: str,
        image_path: Optional[str],
        free_hpo_path: Optional[str],
        vision_json_path: Optional[str] = None,
        stage1_json_path: Optional[str] = None,
        genomics_result_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # Text
        loaded_stage1_cache = False
        cache_path = Path(stage1_json_path) if stage1_json_path else None
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                loaded_stage1_cache = True
                state.note_text = cached.get("note_text") or state.note_text
                state.phenotype_mentions_text = cached.get("text_phenotypes", []) or []
                state.demographics = cached.get("demographics", {}) or {}
                state.family_history = cached.get("family_history", []) or []
                state.testing = cached.get("testing", []) or []
                state.gene_mentions = cached.get("gene_mentions", []) or []
                out.update(cached)
                out["stage1_cache_path"] = str(cache_path)

        if note_path and not loaded_stage1_cache:
            # Read note
            note_text = Path(note_path).read_text(encoding="utf-8")
            text_out = run_stage1_text_extractors_batch(
                self.text_llm, [note_text], self.prompt_dir
            )[0]
            phens = text_out["text_phenotypes"]
            demo = text_out["demographics"]
            fam = text_out["family_history"]
            tests = text_out["testing"]
            genes = text_out["gene_mentions"]
            
            state.note_text = note_text
            state.phenotype_mentions_text = phens if isinstance(phens, list) else []
            state.demographics = demo if isinstance(demo, dict) else {}
            state.family_history = fam if isinstance(fam, list) else []
            state.testing = tests if isinstance(tests, list) else []
            state.gene_mentions = genes if isinstance(genes, list) else []

            out["text_phenotypes"] = state.phenotype_mentions_text
            out["demographics"] = state.demographics
            out["family_history"] = state.family_history
            out["testing"] = state.testing
            out["gene_mentions"] = state.gene_mentions
            out["extraction_diagnostics"] = text_out.get("extraction_diagnostics", {})

        if note_path and not state.note_text:
            state.note_text = Path(note_path).read_text(encoding="utf-8")
        # Vision
        if image_path and Path(image_path).exists():
            vphens: List[Dict[str, Any]] = []
            cache_path = Path(vision_json_path) if vision_json_path else None
            if cache_path and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    vphens = cached.get("vision_phenotypes_raw", [])
                elif isinstance(cached, list):
                    vphens = cached
                out["vision_prefetch_path"] = str(cache_path)
            else:
                self._ensure_vision_llm()
                vphens = run_vision_extractor_batch(
                    self.vision_llm, [image_path], self.prompt_dir
                )[0]
                vision_runtime = getattr(self.cfg, "vision", {})
                if getattr(vision_runtime, "unload_after_extraction", True):
                    self._release_vision_llm()
            vfiltered = filter_vision_against_text(vphens, state.phenotype_mentions_text)
            state.phenotype_mentions_vision = vfiltered
            out["vision_phenotypes"] = vfiltered

        # Free HPO list (optional): lines of "HP:xxx" or free text
        if free_hpo_path and Path(free_hpo_path).exists():
            lines = [
                term
                for l in Path(free_hpo_path).read_text(encoding="utf-8").splitlines()
                if l.strip()
                for term in l.strip().split(";")
                if term  # remove empty from trailing ";"
            ]
            free_mentions: List[Dict[str, Any]] = []
            for l in lines:
                free_mentions.append({
                    "mention": l,
                    "attribution": "patient",
                    "onset": None,
                    "evidence": "free HPO list",
                    "source": "free_hpo",
                })
            state.phenotype_mentions_free_hpo = free_mentions
            out["free_hpo_mentions"] = free_mentions

        # VCF/genomics: prefer completed Exomiser/RankVar/RankVar-like results.
        if vcf_path := state.vcf_path:
            p = Path(vcf_path)
            if p.exists():
                genomics_cfg = getattr(self.cfg, "genomics", {})
                analyzer = getattr(genomics_cfg, "vcf_analyzer", "auto")
                results_dir = getattr(genomics_cfg, "results_dir", "")
                result_path = (
                    Path(genomics_result_path)
                    if genomics_result_path
                    else discover_genomics_result(
                        state.case_id,
                        p,
                        analyzer=analyzer,
                        results_dir=results_dir or None,
                    )
                )
                if result_path and Path(result_path).exists():
                    state.vcf_summary = load_genomics_results(result_path, analyzer=analyzer)
                    out["vcf_summary"] = state.vcf_summary
                    out["vcf_result_path"] = str(result_path)
                    out["vcf_note"] = (
                        f"Loaded {len(state.vcf_summary)} normalized genomics rows "
                        f"from {result_path}."
                    )
                else:
                    out["vcf_summary"] = []
                    out["vcf_note"] = (
                        f"VCF present at {p}; no completed Exomiser/RankVar result found. "
                        "Genomics evidence is pending and was not scored."
                    )

        # Deterministic guardrail after both fresh extraction and cache loading:
        # accompaniment or relationship text alone cannot establish that a
        # relative is clinically affected.
        state.family_history, family_audit = sanitize_family_history(
            state.family_history, note=state.note_text, return_audit=True,
        )
        out["family_history"] = state.family_history
        out["family_history_sanitization"] = family_audit
        out["extracted_entities"] = self._build_extracted_entities(out)
        return out

    def _stage1_text_extraction_batch(
        self,
        cases: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Run Stage 1 text extractors for many cases in true vLLM batches."""
        note_cases = [
            case for case in cases
            if case.get("note_path") and Path(case["note_path"]).exists()
        ]
        if not note_cases:
            return {}

        notes = [Path(case["note_path"]).read_text(encoding="utf-8") for case in note_cases]
        text_outs = run_stage1_text_extractors_batch(self.text_llm, notes, self.prompt_dir)

        out: Dict[str, Dict[str, Any]] = {}
        for case, note_text, text_out in zip(note_cases, notes, text_outs):
            family_history, family_audit = sanitize_family_history(
                text_out["family_history"], note=note_text, return_audit=True,
            )
            payload = {
                "note_text": note_text,
                "text_phenotypes": text_out["text_phenotypes"],
                "demographics": text_out["demographics"],
                "family_history": family_history,
                "family_history_sanitization": family_audit,
                "testing": text_out["testing"],
                "gene_mentions": text_out["gene_mentions"],
                "extraction_diagnostics": text_out.get("extraction_diagnostics", {}),
            }
            payload["extracted_entities"] = self._build_extracted_entities(payload)
            out[case["case_id"]] = payload
        return out

    def precompute_stage1_extractions(
        self,
        cases: List[Dict[str, Any]],
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """Batch Stage 1 text extraction and write per-case cache JSON files."""
        to_extract = []
        for case in cases:
            cache_path = Path(case["stage1_json_path"])
            if cache_path.exists() and not overwrite:
                continue
            if case.get("note_path"):
                to_extract.append(case)

        extracted = self._stage1_text_extraction_batch(to_extract)
        written: Dict[str, str] = {}
        for case in to_extract:
            case_id = case["case_id"]
            if case_id not in extracted:
                continue
            cache_path = Path(case["stage1_json_path"])
            write_json(extracted[case_id], cache_path)
            written[case_id] = str(cache_path)
        return written

    def _stage2_normalization(self, state: PatientCaseState) -> None:
        # Combine all phenotype sources
        all_mentions: List[Dict[str, Any]] = []
        for p in state.phenotype_mentions_text:
            q = dict(p)
            q.setdefault("source", "text")
            all_mentions.append(q)
        for p in state.phenotype_mentions_vision:
            q = dict(p)
            q.setdefault("source", "vision")
            all_mentions.append(q)
        for p in state.phenotype_mentions_free_hpo:
            q = dict(p)
            q.setdefault("source", "free_hpo")
            all_mentions.append(q)

        # Normalize
        normalized = self.hpo_normalizer.normalize(all_mentions, include_negated=True)
        # Convert to pydantic models
        state.normalized_hpo = []
        for n in normalized:
            state.normalized_hpo.append(NormalizedPhenotype(
                hpo_id=n["hpo_id"],
                hpo_name=n["hpo_name"],
                attribution=n.get("attribution", "patient"),
                score=float(n.get("score", 0.0)),
                source=n.get("source", "text"),
                mention=n.get("mention", ""),
                onset=n.get("onset"),
                evidence=n.get("evidence"),
                ic=float(n.get("ic", 0.0)),
                present=bool(n.get("present", True)),
            ))
        state.normalized_hpo = self._deduplicate_normalized_hpo(state.normalized_hpo)

        # Temporal view (over present phenotypes)
        present_list = [
            {"mention": n.mention, "hpo_id": n.hpo_id, "hpo_name": n.hpo_name, "onset": n.onset}
            for n in state.normalized_hpo if n.present
        ]
        tv = build_temporal_view(present_list)
        # age from demographics
        demo = state.demographics if isinstance(state.demographics, dict) else {}
        age_info = demo.get("age", {})
        if isinstance(age_info, dict):
            tv["age_at_presentation"] = age_info.get("value") or age_info.get("age_group")
        state.temporal_view = TemporalView(**tv)

        # Inheritance prior
        state.inheritance_prior = infer_inheritance_prior(state.family_history)

        # Incongruity
        state.incongruity = IncongruityInfo(**detect_incongruity(present_list, self.hpo))

    def _save_patient_evidence(self, state: PatientCaseState, out_dir: Path) -> None:
        pe = {
            "case_id": state.case_id,
            "demographics": state.demographics,
            "phenotypes": [n.model_dump() for n in state.normalized_hpo],
            "negated": [n.model_dump() for n in state.normalized_hpo if not n.present],
            "temporal_view": state.temporal_view.model_dump(),
            "inheritance_prior": state.inheritance_prior,
            "incongruity": state.incongruity.model_dump(),
            "gene_evidence": {
                "gene_mentions": state.gene_mentions,
                "vcf_summary": state.vcf_summary,
            },
            "family_history": state.family_history,
        }
        write_json(pe, out_dir / "stage2_patient_evidence.json")
        write_json(state.incongruity.model_dump(), out_dir / "stage2_incongruity.json")

    def _stage3_retrieval_scoring(
        self,
        state: PatientCaseState,
    ):
        present_hpos = [
            {"hpo_id": n.hpo_id, "mention": n.mention, "reliability": "high"}
            for n in state.normalized_hpo if n.present
        ]

        # Retrieval (union of 3 sources)
        hpo_candidates = retrieve_by_hpo(
            present_hpos, self.kg_index, self.hpo,
            expansion_mode=self.cfg.expansion.mode,
            max_depth=self.cfg.expansion.max_expansion_depth,
            low_ic_threshold=getattr(self.cfg.expansion, "low_ic_threshold", 3.5),
            medium_ic_threshold=getattr(self.cfg.expansion, "medium_ic_threshold", 5.0),
        )
        gene_candidates = retrieve_by_gene(
            state.gene_mentions,
            state.vcf_summary,
            self.kg_index,
            evidence_policy=getattr(
                getattr(self.cfg, "scoring", {}),
                "gene_evidence_policy",
                "negative_only",
            ),
        )
        cooc_candidates = retrieve_by_cooccurrence(
            present_hpos, self.kg_index, self.hpo,
        )

        # -- PubCaseFinder (optional, network) --
        pubcase_scores: Dict[str, float] = {}
        pubcase_gene_df = None
        if (
            getattr(self.cfg, "external", None) is not None
            and getattr(self.cfg.external, "pubcase_finder", None) is not None
            and self.cfg.external.pubcase_finder.enabled
            and present_hpos
        ):
            hpo_id_list = [p["hpo_id"] for p in present_hpos if p.get("hpo_id")]
            if hpo_id_list:
                try:
                    disease_df = search_PubCaseFinder(
                        query=hpo_id_list,
                        mode="disease",
                        max_results=self.cfg.external.pubcase_finder.max_results,
                        disease_mapper=self.disease_mapper,
                    )
                    state.public_disease_cases = disease_df
                    if disease_df is not None and not disease_df.empty and "MONDO_ID" in disease_df.columns:
                        # Normalize scores to [0, 1] for use as the cases_score component
                        raw = disease_df["Score"].astype(float).values
                        if raw.max() > 0:
                            norm = raw / raw.max()
                        else:
                            norm = raw
                        for mondo_id, s in zip(disease_df["MONDO_ID"].values, norm):
                            if mondo_id:
                                pubcase_scores[str(mondo_id)] = float(s)

                    pubcase_gene_df = search_PubCaseFinder(
                        query=hpo_id_list,
                        mode="gene",
                        max_results=self.cfg.external.pubcase_finder.max_results,
                        disease_mapper=None,
                    )
                    state.public_gene_cases = pubcase_gene_df
                except Exception as e:
                    logger.warning(f"PubCaseFinder call failed: {e}")

        all_ids = (
            set(hpo_candidates.keys())
            | set(gene_candidates.keys())
            | set(cooc_candidates.keys())
            | set(pubcase_scores.keys())
        )

        if not all_ids:
            all_ids = set(list(self.kg_index.disease_name.keys())[: retrieval_initial_top_k(self.cfg)])

        # Cap by top_k
        all_ids = list(all_ids)
        
        # Score
        df = score_candidates(
            candidate_ids=all_ids,
            patient_state=state,
            kg=self.kg,
            kg_index=self.kg_index,
            hpo=self.hpo,
            cfg=self.cfg,
            hpo_normalizer=self.hpo_normalizer,
            cooccurrence_candidates=cooc_candidates,
            cases_scores=pubcase_scores,
        )

        # Keep full scored table here; downstream stages apply their own top-N cutoffs.
        df["rank"] = df.index + 1
        return df, cooc_candidates

    def _stage4_frontier(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        out_dir: Path,
    ):
        if not self.cfg.frontier.enabled or self.frontier is None:
            logger.info("Frontier stage disabled in config")
            return {}, {}

        patient_evidence = {
            "demographics": state.demographics,
            "phenotypes": [n.model_dump() for n in state.normalized_hpo],
            "temporal_view": state.temporal_view.model_dump(),
            "gene_evidence": {
                "gene_mentions": state.gene_mentions,
                "vcf_summary": state.vcf_summary,
            },
            "family_history": state.family_history,
        }

        incongruity = state.incongruity.model_dump()

        out = run_frontier_consultation(
            self.frontier, patient_evidence, ranked_df, incongruity,
            self.cfg, self.prompt_dir,
        )

        # Build per-disease flags for downstream stages
        flags: Dict[str, Dict[str, Any]] = {}
        for e in out.get("underranked", []):
            if e.get("matched") and e.get("disease_id"):
                flags[e["disease_id"]] = {
                    "flag_type": "underranked",
                    "category": "strong",
                    "lens": e.get("lens", ""),
                    "reasoning": e.get("reasoning", ""),
                }
        for e in out.get("overranked", []):
            if e.get("matched") and e.get("disease_id"):
                flags[e["disease_id"]] = {
                    "flag_type": "overranked",
                    "category": "weak",
                    "lens": e.get("lens", ""),
                    "reasoning": e.get("reasoning", ""),
                }

        write_json(out, out_dir / "stage4_frontier_output.json")
        return out, flags

    def _stage5_audit(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        frontier_flags: Dict[str, Dict[str, Any]],
    ):
        audit_results = run_audit_batch(
            self.text_llm, state, ranked_df, self.kg_index, self.cfg,
            self.prompt_dir, hpo=self.hpo, frontier_flags=frontier_flags,
        )
        ranking_after_audit = apply_audit_multipliers(ranked_df, audit_results)
        return audit_results, ranking_after_audit

    def _stage6_pairwise(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        audit_results: List[Dict[str, Any]],
        frontier_flags: Dict[str, Dict[str, Any]],
        track: str = "subtype",
    ):
        return run_pairwise_batch(
            self.text_llm, state, ranked_df, audit_results,
            self.kg_index, self.kg, self.hpo, self.cfg, self.prompt_dir,
            frontier_flags=frontier_flags, track=track,
        )

    def _aggregate_to_group(
        self,
        subtype_df: pd.DataFrame,
        frontier_flags: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Collapse subtype-level ranking to group-level ranking."""
        if not self.kg_index:
            return pd.DataFrame()

        frontier_flags = frontier_flags or {}
        rows: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for _, r in subtype_df.iterrows():
            gid = r.get("group_id") or r.get("mondo_group_id") or r["disease_id"]
            grouped.setdefault(gid, []).append(r.to_dict())

        for gid, members in grouped.items():
            # best member's adjusted score is the group's initial score
            scores = [float(m.get("adjusted_score", m.get("total_score", 0.0))) for m in members]
            plausibility_priority = {
                "strong": 4,
                "moderate": 3,
                "weak": 2,
                "not_audited": 1,
                "implausible": 0,
            }
            best_plausibility = max(
                (str(m.get("audit_plausibility", "not_audited")) for m in members),
                key=lambda value: plausibility_priority.get(value, 1),
                default="not_audited",
            )
            member_sources = [str(m.get("audit_source", "")) for m in members]
            if "frontier_underranked" in member_sources:
                audit_source = "frontier_underranked"
            elif "frontier_overranked" in member_sources:
                audit_source = "frontier_overranked"
            else:
                audit_source = next((source for source in member_sources if source), "")
            member_ids = [str(m.get("disease_id", "")) for m in members]
            member_flag_types = {
                (frontier_flags.get(member_id) or {}).get("flag_type", "")
                for member_id in member_ids
            }
            if "underranked" in member_flag_types:
                audit_source = "frontier_underranked"
            elif "overranked" in member_flag_types and best_plausibility in {
                "weak",
                "implausible",
                "not_audited",
            }:
                audit_source = "frontier_overranked"
            group_name = self.kg_index.disease_name.get(gid, gid)
            rows.append({
                "disease_id": gid,
                "disease_name": group_name,
                "group_id": gid,
                "group_name": group_name,
                "total_score": max(scores) if scores else 0.0,
                "adjusted_score": max(scores) if scores else 0.0,
                "audit_plausibility": best_plausibility,
                "audit_multiplier": max(float(m.get("audit_multiplier", 1.0)) for m in members),
                "audit_source": audit_source,
                "n_members": len(members),
                "member_ids": member_ids,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("adjusted_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["adjusted_rank"] = df["rank"]
        return df


# Backward-compatible import alias. RareGraph now refers to the knowledge graph;
# new code should import RareMindHost.
RareGraphHost = RareMindHost
