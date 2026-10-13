"""Stage 9 clinical synthesis: three dependency-aware vLLM inference waves.

No ranking, extraction, or Stage 10 logic is changed here. All inputs are
read-only. The full source note is retained unless a request cannot fit the
model context; in that case, an explicitly configured, audited head-and-tail
truncation is applied to the raw note while structured evidence is preserved.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

REPORT_PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts" / "report"

def _read_report_prompt(filename: str) -> str:
    """Resolve against the source checkout, not the notebook's working directory."""
    path = REPORT_PROMPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Stage 9 prompt missing: {path}. Copy configs/prompts/report with the source code.")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Stage 9 prompt is empty: {path}")
    return text


def _render_prompt(filename: str, **values: Any) -> str:
    # Single-pass substitution: braces/dollar signs inside clinical evidence
    # are preserved verbatim and are never interpreted as template variables.
    try:
        return Template(_read_report_prompt(filename)).substitute(values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid Stage 9 template {filename}: {exc}") from exc

NO_FAMILY = "No family history available for this patient."
TOKEN_DEFAULTS = {
    "patient_summary": 2000, "family_history": 1500,
    "candidate_top3": 2500, "candidate_4_to_7": 2000, "candidate_8_plus": 1500,
    "differential_synthesis": 1500, "next_steps": 2000,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return _plain(obj.model_dump())
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "item"):
        return _plain(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _dump(obj: Any) -> str:
    return json.dumps(_plain(obj), ensure_ascii=False, indent=2, default=str)


def _phenotypes(state: Any) -> list[dict]:
    values = _get(state, "normalized_hpo", None)
    if values is None:
        values = _get(state, "phenotypes", [])
    return [_plain(p) for p in (values or [])]


def _present(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def fmt_phenotypes(phenotypes: list[dict], present: bool) -> str:
    rows = []
    for p in phenotypes:
        if _present(p.get("present", True)) != present:
            continue
        rows.append(
            f"{p.get('hpo_name', '')} | {p.get('hpo_id', '')} | "
            f"mention={p.get('mention', '')} | evidence={p.get('evidence', '')} | "
            f"IC={p.get('ic', 'unknown')} | present={str(present).lower()} | "
            f"attribution={p.get('attribution', 'unknown')} | source={p.get('source', 'unknown')}"
        )
    return "\n".join(rows) if rows else "(none documented)"


def fmt_testing(testing: Any) -> str:
    # Preserve nested assay/result/status/provenance fields instead of assuming
    # a single test_name/result schema across historical Stage 1 extractions.
    return _dump(testing) if testing else "(no prior testing documented in supplied evidence)"


def _patient_blocks(state: Any, include_note: bool = True) -> str:
    sections = []
    if include_note:
        sections.append(("CLINICAL NOTE — FULL SOURCE TEXT", _get(state, "note_text", "") or "(not supplied)"))
    sections.extend([
        ("STRUCTURED PHENOTYPES (PRESENT)", fmt_phenotypes(_phenotypes(state), True)),
        ("NEGATED PHENOTYPES", fmt_phenotypes(_phenotypes(state), False)),
        ("PRIOR TESTING", fmt_testing(_get(state, "testing", []))),
        ("DEMOGRAPHICS", _dump(_get(state, "demographics", {}))),
        ("GENE MENTIONS", _dump(_get(state, "gene_mentions", []))),
        ("AVAILABLE GENOMICS RESULTS", _dump(_get(state, "vcf_summary", []))),
        ("RAW TEXT PHENOTYPES", _dump(_get(state, "phenotype_mentions_text", []))),
        ("IMAGE-EXTRACTED FINDINGS (not a new image interpretation)",
         _dump(_get(state, "phenotype_mentions_vision", []))),
    ])
    return "\n\n".join(f"--- {label} ---\n{text}" for label, text in sections)


def kg_entry_for_candidate(disease_id: str, kg: dict) -> dict:
    """Trim only the supplied disease entry, without inventing group evidence."""
    raw = kg.get(disease_id, {})
    if not isinstance(raw, Mapping) or not raw:
        return {"disease_id": disease_id, "entry_status": "not_available"}
    entry = copy.deepcopy(raw)
    result = {key: entry.get(key, {}) for key in (
        "definition", "genes", "inheritance", "diagnosis", "testing",
        "genetic_counseling", "initial_evaluations", "surveillance",
        "evaluation_of_relatives_at_risk",
    )}
    result.update({
        "disease_id": disease_id, "omim_ids": entry.get("omim_ids", []),
        "orpha_ids": entry.get("orpha_ids", []),
        "source": entry.get("source", entry.get("sources", "")),
    })
    phenotypes = entry.get("phenotypes", {})
    if isinstance(phenotypes, list):
        phenotypes = {
            p.get("name", p.get("hpo_name", str(i))): p
            for i, p in enumerate(phenotypes) if isinstance(p, Mapping)
        }
    selected, common = {}, []
    for name, data in (phenotypes.items() if isinstance(phenotypes, Mapping) else []):
        if not isinstance(data, Mapping):
            continue
        freq = str(data.get("freq", data.get("frequency", ""))).lower().replace(" ", "_")
        importance = str(data.get("importance", "")).lower()
        if importance == "characteristic" or freq == "very_common":
            selected[name] = data
        elif freq == "common":
            common.append((name, data))
    order = {"characteristic": 0, "high": 0, "supportive": 1,
             "medium": 1, "incidental": 2, "low": 2}
    common.sort(key=lambda pair: order.get(str(pair[1].get("importance", "")).lower(), 3))
    selected.update(common[:10])
    result["phenotypes"] = selected
    return result


def fmt_pairwise(pairwise: list[dict], disease_id: str) -> str:
    rows = []
    for p in pairwise:
        if disease_id not in {p.get("disease_a_id"), p.get("disease_b_id")}:
            continue
        is_a = p.get("disease_a_id") == disease_id
        other = p.get("disease_b_name" if is_a else "disease_a_name", "unknown")
        winner = p.get("winner", "tie")
        winner_label = (
            "tie" if str(winner).lower() == "tie" else
            "THIS CANDIDATE" if winner == ("A" if is_a else "B") else other
        )
        rows.append(f"vs. {other} | winner: {winner_label} | strength: {p.get('strength', '')} | reasoning: {p.get('reasoning', '')}")
    return "\n".join(rows) if rows else "(no pairwise comparisons available)"


def build_call1_prompt(state: Any) -> str:
    return _render_prompt('patient_summary.txt',
        patient_evidence=_patient_blocks(state),
    )


def build_call2_prompt(state: Any, cards: list[dict], kg: dict) -> str | None:
    family = _get(state, "family_history", [])
    if not family:
        return None
    candidates = [{
        "disease": c["disease_name"], "MONDO": c["disease_id"],
        "inheritance": kg_entry_for_candidate(c["disease_id"], kg).get("inheritance", {}),
        "genes": kg_entry_for_candidate(c["disease_id"], kg).get("genes", {}),
        "family_evidence_score": c.get("family_evidence_score"),
    } for c in cards]
    return _render_prompt('family_history.txt',
        clinical_note=_get(state, 'note_text', '') or '(not supplied)',
        family_history=_dump(family),
        inheritance_prior=_dump(_get(state, 'inheritance_prior', {})),
        gene_mentions=_dump(_get(state, 'gene_mentions', [])),
        candidates=_dump(candidates),
    )


def _audit_for_synthesis(audit: dict) -> dict:
    """Label epistemic status without altering archived audit evidence or scores."""
    upstream = copy.deepcopy(audit)
    missing = upstream.pop("missing_expected_evidence", [])
    contradictions = upstream.pop("contradicting_evidence", [])
    return {
        "upstream_assessment_not_independent_clinical_evidence": upstream,
        "claimed_contradictions_requiring_source_verification": contradictions,
        "undocumented_or_unresolved_features_for_clarification_not_negative_evidence": missing,
    }


def build_candidate_prompt(card: dict, audit: dict, pairwise: list[dict], entry: dict,
                           patient_summary: str, family_summary: str, state: Any,
                           scores: dict, n_candidates: int) -> str:
    rank = card["final_rank"]
    trajectory = {k: card.get(k) for k in (
        "composite_rank", "adjusted_rank", "reranked_rank", "reconciled_rank",
        "final_rank", "total_rank_change", "final_score", "audit_plausibility",
        "final_fusion_original_weight", "final_fusion_reconciled_weight",
    )}
    return _render_prompt('candidate_assessment.txt',
        patient_summary=patient_summary,
        family_summary=family_summary,
        disease_name=card['disease_name'],
        disease_id=card['disease_id'],
        rank=rank,
        n_candidates=n_candidates,
        kg_entry=_dump(entry),
        audit=_dump(_audit_for_synthesis(audit)),
        pairwise=fmt_pairwise(pairwise, card['disease_id']),
        trajectory=_dump(trajectory),
        scores=_dump(scores),
        patient_evidence=_patient_blocks(state),
        family_history=_dump(_get(state, 'family_history', [])),
    )


def group_entry_for_card(card: dict, kg: dict) -> dict:
    return {
        "group_entry": kg_entry_for_candidate(card["disease_id"], kg),
        "representative_subtypes": [
            {
                **representative,
                "knowledge_graph_entry": kg_entry_for_candidate(representative["disease_id"], kg),
            }
            for representative in card.get("representative_subtypes", [])
        ],
    }


def build_group_candidate_prompt(card: dict, pairwise: list[dict], patient_summary: str,
                                 family_summary: str, state: Any, kg: dict,
                                 scores: dict, n_candidates: int) -> str:
    return _render_prompt(
        "group_candidate_assessment.md",
        patient_summary=patient_summary,
        family_summary=family_summary,
        group_name=card["disease_name"],
        group_id=card["disease_id"],
        rank=card["final_rank"],
        n_candidates=n_candidates,
        member_count=card.get("member_count", 0),
        representative_subtypes=_dump(card.get("representative_subtypes", [])),
        group_kg_context=_dump(group_entry_for_card(card, kg)),
        pairwise=fmt_pairwise(pairwise, card["disease_id"]),
        scores=_dump(scores),
        patient_evidence=_patient_blocks(state),
        family_history=_dump(_get(state, "family_history", [])),
    )


def _extract_overall(text: str, fallback: str) -> tuple[str, str]:
    match = re.search(r"OVERALL\s+ASSESSMENT[^\w\n]*\s*(.*)", text, flags=re.I | re.S)
    if match and match.group(1).strip():
        return match.group(1).strip(), "stage9_candidate_assessment"
    return fallback or "No overall assessment was available.", "stage5_fallback"


def build_synthesis_prompt(summary: str, family: str, assessments: list[dict],
                           state: Any, reconciled: dict) -> str:
    return _render_prompt('differential_synthesis.txt',
        patient_summary=summary,
        family_summary=family,
        assessments=_dump(assessments),
        testing=fmt_testing(_get(state, 'testing', [])),
        incongruity=_dump(_get(state, 'incongruity', {})),
        reconciled=_dump(reconciled),
        patient_evidence=_patient_blocks(state),
    )


def build_next_steps_prompt(summary: str, family: str, cards: list[dict],
                            audit_by_id: dict, kg: dict, state: Any) -> str:
    management = []
    for card in cards[:5]:
        group_context = (
            group_entry_for_card(card, kg)
            if card.get("representative_subtypes") is not None else None
        )
        entry = group_context["group_entry"] if group_context else kg_entry_for_candidate(card["disease_id"], kg)
        representative_management = []
        if group_context:
            for representative in group_context["representative_subtypes"]:
                representative_entry = representative.get("knowledge_graph_entry", {})
                representative_management.append({
                    "disease_id": representative.get("disease_id"),
                    "disease_name": representative.get("disease_name"),
                    **{key: representative_entry.get(key, {}) for key in (
                        "testing", "initial_evaluations", "surveillance",
                        "evaluation_of_relatives_at_risk",
                    )},
                })
        management.append({
            "rank": card["final_rank"], "disease_name": card["disease_name"],
            "disease_id": card["disease_id"], "source": entry.get("source", ""),
            **{k: entry.get(k, {}) for k in (
                "testing", "initial_evaluations", "surveillance", "evaluation_of_relatives_at_risk"
            )},
            "representative_subtype_management": representative_management,
            "high_importance_missing_findings": [
                m for m in audit_by_id.get(card["disease_id"], {}).get("missing_expected_evidence", [])
                if isinstance(m, dict) and m.get("importance") == "high"
            ],
        })
    return _render_prompt('next_steps.txt',
        patient_summary=summary,
        family_summary=family,
        management=_dump(management),
        testing=fmt_testing(_get(state, 'testing', [])),
        family_history=_dump(_get(state, 'family_history', [])),
    )


@dataclass(frozen=True)
class Request:
    label: str
    prompt: str
    max_tokens: int


_FULL_NOTE_HEADER = "--- CLINICAL NOTE — FULL SOURCE TEXT ---\n"
_FULL_NOTE_END = "\n\n--- STRUCTURED PHENOTYPES"
_TRUNCATION_MARKER = (
    "\n\n[... middle of source clinical note omitted because the Stage 9 request "
    "exceeded the model context; beginning and end retained ...]\n\n"
)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _decode_head_tail(tokenizer: Any, token_ids: list[int], keep: int) -> str:
    """Keep both ends of a token sequence and mark the omitted middle."""
    keep = max(0, min(int(keep), len(token_ids)))
    if keep >= len(token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    head = int(round(keep * 0.55))
    tail = keep - head
    left = tokenizer.decode(token_ids[:head], skip_special_tokens=True) if head else ""
    right = tokenizer.decode(token_ids[-tail:], skip_special_tokens=True) if tail else ""
    return left.rstrip() + _TRUNCATION_MARKER + right.lstrip()


def _fit_user_prompt_to_context(
    user_prompt: str,
    *,
    tokenizer: Any,
    build_prompt: Any,
    target_full_input_tokens: int,
) -> tuple[str, str, int, int, str]:
    """Fit one request, preferentially truncating only the raw note block.

    Returns ``(user_prompt, rendered_prompt, original_tokens, retained_tokens,
    strategy)``. Binary search uses the fully rendered chat prompt, so chat
    template overhead is included in the limit.
    """
    rendered = build_prompt(user_prompt)
    original_tokens = _token_count(tokenizer, rendered)
    if original_tokens <= target_full_input_tokens:
        return user_prompt, rendered, original_tokens, original_tokens, "none"
    if not hasattr(tokenizer, "decode"):
        raise ValueError(
            "Stage 9 input truncation requires a tokenizer with encode and decode methods"
        )

    note_start = user_prompt.find(_FULL_NOTE_HEADER)
    if note_start >= 0:
        content_start = note_start + len(_FULL_NOTE_HEADER)
        content_end = user_prompt.find(_FULL_NOTE_END, content_start)
        if content_end >= 0:
            prefix = user_prompt[:content_start]
            note = user_prompt[content_start:content_end]
            suffix = user_prompt[content_end:]
            note_ids = tokenizer.encode(note, add_special_tokens=False)
            low, high = 0, len(note_ids)
            best: tuple[str, str, int] | None = None
            while low <= high:
                keep = (low + high) // 2
                candidate_user = prefix + _decode_head_tail(tokenizer, note_ids, keep) + suffix
                candidate_rendered = build_prompt(candidate_user)
                candidate_tokens = _token_count(tokenizer, candidate_rendered)
                if candidate_tokens <= target_full_input_tokens:
                    best = (candidate_user, candidate_rendered, candidate_tokens)
                    low = keep + 1
                else:
                    high = keep - 1
            if best is not None:
                return best[0], best[1], original_tokens, best[2], "clinical_note_head_tail"

    # Defensive fallback for an unexpectedly large nonstandard prompt. This is
    # deliberately secondary to note-only truncation and is always disclosed.
    user_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    low, high = 0, len(user_ids)
    best = None
    while low <= high:
        keep = (low + high) // 2
        candidate_user = _decode_head_tail(tokenizer, user_ids, keep)
        candidate_rendered = build_prompt(candidate_user)
        candidate_tokens = _token_count(tokenizer, candidate_rendered)
        if candidate_tokens <= target_full_input_tokens:
            best = (candidate_user, candidate_rendered, candidate_tokens)
            low = keep + 1
        else:
            high = keep - 1
    if best is None:
        raise ValueError(
            "Stage 9 prompt cannot fit the configured context even after explicit truncation"
        )
    return best[0], best[1], original_tokens, best[2], "prompt_head_tail_fallback"


def _clean_prose(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Never expose an unfinished thinking-only generation as a clinical report.
    if "<think>" in text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


class WaveRunner:
    """Native vLLM batched generate, or concurrent AsyncOpenAI endpoint calls."""

    def __init__(self, llm: Any, config: Any):
        self.llm = llm
        self.config = config or {}
        self.system_prompt = _read_report_prompt("system.txt")
        self.backend = _get(config, "backend", "native")
        self.temperature = float(_get(config, "temperature", 0.1))
        self.model = (_get(config, "model_name", "") or _get(llm, "model_path", "")) \
            if self.backend == "endpoint" else _get(llm, "model_path", "")
        self.calls: list[dict] = []
        self.client = None

    async def __aenter__(self):
        if self.backend == "endpoint":
            from openai import AsyncOpenAI
            base_url = _get(self.config, "base_url", "")
            if not base_url or not self.model:
                raise ValueError("Stage 9 endpoint mode requires base_url and the served model_name")
            self.client = AsyncOpenAI(
                base_url=base_url, api_key=_get(self.config, "api_key", "EMPTY"),
                timeout=float(_get(self.config, "timeout_seconds", 300)),
                max_retries=int(_get(self.config, "max_retries", 2)),
            )
        elif self.backend != "native":
            raise ValueError(f"Unknown Stage 9 backend: {self.backend}")
        elif not all(hasattr(self.llm, name) for name in ("llm", "_build_prompt", "sampling_params")):
            raise ValueError("Stage 9 native mode requires the pipeline's loaded VllmClient")
        return self

    async def __aexit__(self, *args):
        if self.client is not None:
            await self.client.close()

    def _record(
        self,
        request: Request,
        content: Any,
        finish: Any,
        wave: int,
        *,
        effective_max_tokens: int | None = None,
        input_tokens: int | None = None,
        max_context_tokens: int | None = None,
        original_input_tokens: int | None = None,
        input_truncation_strategy: str = "none",
    ) -> str:
        text = _clean_prose(content)
        if not text:
            raise RuntimeError(f"Stage 9 {request.label} returned no clinical text (finish_reason={finish})")
        effective = int(effective_max_tokens or request.max_tokens)
        context_budget_reduced = effective < request.max_tokens
        self.calls.append({
            "label": request.label, "wave": wave,
            # ``max_tokens`` remains the budget actually sent to the model for
            # compatibility with existing reports. The requested value is
            # retained separately whenever context pressure reduces it.
            "max_tokens": effective,
            "requested_max_tokens": request.max_tokens,
            "effective_max_tokens": effective,
            "input_tokens": input_tokens,
            "original_input_tokens": original_input_tokens or input_tokens,
            "input_truncated": input_truncation_strategy != "none",
            "input_truncation_strategy": input_truncation_strategy,
            "input_tokens_removed": max(
                0, int(original_input_tokens or input_tokens or 0) - int(input_tokens or 0)
            ),
            "max_context_tokens": max_context_tokens,
            "context_budget_reduced": context_budget_reduced,
            "finish_reason": finish,
            "possibly_truncated": finish == "length",
        })
        if context_budget_reduced:
            logger.warning(
                "Stage 9 %s preserved the full prompt and reduced its output "
                "budget from %s to %s tokens to fit the %s-token context",
                request.label, request.max_tokens, effective, max_context_tokens,
            )
        if input_truncation_strategy != "none":
            logger.warning(
                "Stage 9 %s truncated an overlong input from %s to %s tokens "
                "using %s; the requested %s-token output budget was preserved",
                request.label, original_input_tokens, input_tokens,
                input_truncation_strategy, effective,
            )
        if finish == "length":
            logger.warning("Stage 9 %s reached its output token limit", request.label)
        return text

    async def run(self, requests: list[Request], wave: int) -> list[str]:
        if not requests:
            return []
        logger.info("Stage 9 wave %s: %s synthesis requests", wave, len(requests))
        if self.client is not None:
            concurrency = int(_get(self.config, "max_concurrency", 14))
            if concurrency < 1:
                raise ValueError("Stage 9 max_concurrency must be positive")
            semaphore = asyncio.Semaphore(concurrency)

            async def call(request):
                async with semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_prompt},
                                  {"role": "user", "content": request.prompt}],
                        max_tokens=request.max_tokens, temperature=self.temperature,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    choice = response.choices[0]
                    return self._record(request, choice.message.content, choice.finish_reason, wave)

            return list(await asyncio.gather(*(call(r) for r in requests)))

        # Use the already-loaded engine. Per-request SamplingParams allow all
        # ten different candidate budgets in ONE generate call, not ten chats.
        cap = int(_get(self.config, "max_batch_size", _get(self.llm, "max_batch_size", 64)))
        if cap < 1:
            raise ValueError("Stage 9 max_batch_size must be positive")
        result = []
        for start in range(0, len(requests), cap):
            chunk = requests[start:start + cap]
            prompts = []
            params = []
            call_limits = []
            for request in chunk:
                sampling = copy.deepcopy(self.llm.sampling_params)
                sampling.temperature = self.temperature
                engine_config = _get(_get(self.llm.llm, "llm_engine"), "model_config")
                max_context = _get(engine_config, "max_model_len")
                tokenizer = _get(self.llm, "tokenizer")
                input_length = None
                original_input_length = None
                truncation_strategy = "none"
                effective_max_tokens = request.max_tokens
                prompt = self.llm._build_prompt(self.system_prompt, request.prompt)
                if max_context and tokenizer is not None:
                    safety_margin = int(_get(self.config, "context_safety_margin", 64))
                    minimum_output = int(_get(self.config, "minimum_output_tokens", 256))
                    truncate_inputs = bool(_get(self.config, "truncate_overlong_inputs", False))
                    preserve_output = bool(_get(self.config, "preserve_requested_output_tokens", True))
                    input_length = _token_count(tokenizer, prompt)
                    original_input_length = input_length
                    if truncate_inputs and preserve_output:
                        target_input = int(max_context) - safety_margin - request.max_tokens
                        if target_input < 1:
                            raise ValueError(
                                f"Stage 9 {request.label} output budget {request.max_tokens} plus "
                                f"safety margin {safety_margin} exceeds context {max_context}"
                            )
                        if input_length > target_input:
                            _, prompt, original_input_length, input_length, truncation_strategy = (
                                _fit_user_prompt_to_context(
                                    request.prompt,
                                    tokenizer=tokenizer,
                                    build_prompt=lambda user: self.llm._build_prompt(
                                        self.system_prompt, user
                                    ),
                                    target_full_input_tokens=target_input,
                                )
                            )
                    available_output = int(max_context) - input_length - safety_margin
                    if available_output < minimum_output:
                        raise ValueError(
                            f"Stage 9 {request.label} has {input_length} input tokens but only "
                            f"{max(0, available_output)} safe output tokens remain in context "
                            f"{max_context} (minimum required: {minimum_output}). The input was "
                            "not silently truncated. Enable stage9.truncate_overlong_inputs or "
                            "use a model with a larger context."
                        )
                    effective_max_tokens = min(request.max_tokens, available_output)
                sampling.max_tokens = effective_max_tokens
                prompts.append(prompt)
                params.append(sampling)
                call_limits.append((
                    effective_max_tokens, input_length, max_context,
                    original_input_length, truncation_strategy,
                ))
            outputs = self.llm.llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(chunk):
                raise RuntimeError("Stage 9 vLLM returned an unexpected number of outputs")
            for request, output, limits in zip(chunk, outputs, call_limits):
                completion = output.outputs[0]
                result.append(self._record(
                    request, completion.text, completion.finish_reason, wave,
                    effective_max_tokens=limits[0], input_tokens=limits[1],
                    max_context_tokens=limits[2], original_input_tokens=limits[3],
                    input_truncation_strategy=limits[4],
                ))
        return result


def _format_transparency_table(cards: list[dict], report_level: str = "subtype") -> str:
    if report_level == "group":
        lines = [
            "Rank | Disease group (MONDO) | Group score | Members | Representative subtypes",
            "-" * 120,
        ]
        for card in cards:
            representatives = "; ".join(
                item.get("disease_name", "") for item in card.get("representative_subtypes", [])
            ) or "none available"
            lines.append(
                f"{card['final_rank']} | {card['disease_name']} ({card['disease_id']}) | "
                f"{card['final_score']:.3f} | {card.get('member_count', 0)} | {representatives}"
            )
        lines.extend(["", "Group scores are relative pipeline ranking values, not diagnostic probabilities."])
        return "\n".join(lines)
    lines = [
        "Rank | Disease (MONDO) | Composite | Audit | Pairwise | Reconciled | Final | Audit rating | Family score",
        "-" * 120,
    ]
    for c in cards:
        lines.append(
            f"{c['final_rank']} | {c['disease_name']} ({c['disease_id']}) | "
            f"{c['composite_rank']} | {c['adjusted_rank']} | {c['reranked_rank']} | "
            f"{c['reconciled_rank']} | {c['final_rank']} | {c['audit_plausibility']} | "
            f"{c.get('family_evidence_score', 0):.2f}"
        )
    lines.extend(["", "Final scores are relative pipeline ranking values, not diagnostic probabilities."])
    return "\n".join(lines)


def _assemble_report(state, cards, summary, family, assessments, synthesis, next_steps, metadata,
                     report_level="subtype"):
    if len(cards) != len(assessments):
        raise ValueError(
            f"Cannot render Stage 9 report: {len(cards)} candidates but "
            f"{len(assessments)} assessments"
        )
    parts = [
        "=" * 70,
        "RareMind GROUP-LEVEL CLINICAL ASSESSMENT" if report_level == "group" else "RareMind CLINICAL ASSESSMENT",
        "=" * 70,
        f"Case: {_get(state, 'case_id', 'unknown')}",
        f"Report generated (UTC): {metadata['generated_at_utc']}",
        "PATIENT SUMMARY", "-" * 70, summary,
    ]
    if metadata["family_history_included"]:
        parts.extend(["", "SECTION 1: FAMILY HISTORY ASSESSMENT", "-" * 70, family])
    parts.extend(["", f"SECTION 2: DIFFERENTIAL ASSESSMENT — TOP {len(cards)} CANDIDATES", "-" * 70])
    for index, (card, assessment) in enumerate(zip(cards, assessments), start=1):
        group = ""
        if card.get("group_name") or card.get("group_id"):
            group = f"Group: {card.get('group_name', '')} ({card.get('group_id', '')})"
        parts.extend([
            "", "=" * 70,
            f"CANDIDATE {index} OF {len(cards)} — FINAL RANK {card['final_rank']}",
            f"{card['disease_name']} ({card['disease_id']})",
            *([group] if group else []),
            "-" * 70,
            assessment["text"],  # complete model output; never sliced for presentation
        ])
    parts.extend([
        "", "SECTION 3: DIFFERENTIAL SYNTHESIS", "-" * 70, synthesis,
        "", "SECTION 4: RECOMMENDED NEXT STEPS", "-" * 70, next_steps,
        "", "PIPELINE TRANSPARENCY", "-" * 70, _format_transparency_table(cards, report_level),
    ])
    if metadata["warnings"]:
        parts.extend(["", "GENERATION QUALITY NOTES", *metadata["warnings"]])
    return "\n".join(parts)


async def _run_synthesis_async(patient_state, top_cards, final_df, audit_results,
                               reconciled, kg, llm, pairwise_subtype, config,
                               report_level="subtype"):
    audit_by_id = {a["disease_id"]: a for a in audit_results}
    pairwise = pairwise_subtype if pairwise_subtype is not None else _get(patient_state, "pairwise_results_subtype", [])
    budgets = {**TOKEN_DEFAULTS, **dict(_get(config, "token_budgets", {}) or {})}
    if any(int(v) < 1 for v in budgets.values()):
        raise ValueError("Stage 9 token budgets must be positive")
    if not top_cards:
        raise ValueError("Stage 9 cannot synthesize a report without ranked candidates")
    family_prompt = build_call2_prompt(patient_state, top_cards, kg)
    async with WaveRunner(llm, config) as runner:
        wave1 = [Request("patient_summary", build_call1_prompt(patient_state), int(budgets["patient_summary"]))]
        if family_prompt:
            wave1.append(Request("family_history", family_prompt, int(budgets["family_history"])))
        first = await runner.run(wave1, wave=1)
        summary, family = first[0], first[1] if family_prompt else NO_FAMILY

        raw_by_id = {str(row["disease_id"]): row for row in final_df.to_dict("records")}
        stage3 = _get(patient_state, "ranked")
        stage3_by_id = {
            str(row["disease_id"]): row for row in stage3.to_dict("records")
        } if isinstance(stage3, pd.DataFrame) else {}
        wave2 = []
        for card in top_cards:
            did, rank = card["disease_id"], card["final_rank"]
            merged = {**stage3_by_id.get(did, {}), **raw_by_id.get(did, {})}
            score_fields = {k: v for k, v in merged.items() if "score" in k or k == "negation_penalty"}
            key = "candidate_top3" if rank <= 3 else "candidate_4_to_7" if rank <= 7 else "candidate_8_plus"
            candidate_prompt = (
                build_group_candidate_prompt(card, pairwise or [], summary, family,
                    patient_state, kg, score_fields, len(top_cards))
                if report_level == "group"
                else build_candidate_prompt(card, audit_by_id.get(did, {}), pairwise or [],
                    kg_entry_for_candidate(did, kg), summary, family, patient_state,
                    score_fields, len(top_cards))
            )
            wave2.append(Request(
                f"candidate:{did}",
                candidate_prompt,
                int(budgets[key]),
            ))
        candidate_texts = await runner.run(wave2, wave=2)
        assessments = []
        overall_inputs = []
        for card, text in zip(top_cards, candidate_texts):
            overall, source = _extract_overall(text, card.get("key_distinguishing_note", ""))
            assessments.append({
                "disease_id": card["disease_id"], "final_rank": card["final_rank"],
                "text": text, "overall_assessment": overall, "overall_assessment_source": source,
            })
            overall_inputs.append({
                "rank": card["final_rank"], "disease_name": card["disease_name"],
                "final_score": card["final_score"], "audit_plausibility": card["audit_plausibility"],
                "overall_assessment": overall, "source": source,
            })
        synthesis, next_steps = await runner.run([
            Request("differential_synthesis", build_synthesis_prompt(summary, family, overall_inputs,
                patient_state, reconciled), int(budgets["differential_synthesis"])),
            Request("next_steps", build_next_steps_prompt(summary, family, top_cards, audit_by_id,
                kg, patient_state), int(budgets["next_steps"])),
        ], wave=3)
        warnings = [f"{c['label']}: output reached token limit; review completeness."
                    for c in runner.calls if c["possibly_truncated"]]
        warnings.extend(
            f"{c['label']}: full input preserved; output budget reduced from "
            f"{c['requested_max_tokens']} to {c['effective_max_tokens']} tokens to fit context."
            for c in runner.calls if c.get("context_budget_reduced")
        )
        warnings.extend(
            f"{c['label']}: overlong input truncated from {c['original_input_tokens']} "
            f"to {c['input_tokens']} tokens using {c['input_truncation_strategy']}; "
            "structured evidence and both ends of the source text were retained."
            for c in runner.calls if c.get("input_truncated")
        )
        warnings.extend(f"{a['disease_id']}: missing OVERALL ASSESSMENT heading; upstream fallback used."
                        for a in assessments if a["overall_assessment_source"] == "stage5_fallback")
        metadata = {
            "status": "complete_with_warnings" if warnings else "complete",
            "model": runner.model, "backend": runner.backend,
            "temperature": runner.temperature, "calls": runner.calls,
            "n_calls": len(runner.calls), "n_waves": 3,
            "family_history_included": bool(family_prompt), "warnings": warnings,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ranking_changed": False,
            "report_level": report_level,
        }
    return {
        "patient_summary": summary, "family_history_assessment": family if family_prompt else None,
        "clinical_assessments": assessments, "differential_synthesis": synthesis,
        "recommended_next_steps": next_steps, "synthesis_metadata": metadata,
        "synthesis_report": _assemble_report(patient_state, top_cards, summary, family,
            assessments, synthesis, next_steps, metadata, report_level),
    }


def run_synthesis(**kwargs) -> dict:
    """Synchronous caller API, including notebooks with an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_synthesis_async(**kwargs))
    # asyncio.run cannot execute on Jupyter's running loop. The one worker runs
    # the complete three-wave operation; native inference is still batched.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run_synthesis_async(**kwargs))).result()
