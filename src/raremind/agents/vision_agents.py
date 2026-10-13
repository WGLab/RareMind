"""Stage 1 vision extraction + concordance filtering with text."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from raremind.core.json_utils import safe_json_load
from raremind.core.utils import read_prompt

logger = logging.getLogger(__name__)


def run_vision_extractor_batch(
    vision_llm: Any,
    image_paths: List[str],
    prompt_dir: str,
) -> List[List[Dict[str, Any]]]:
    prompt_path = Path(prompt_dir) / "extraction" / "vision_phenotypes_extractor.md"
    prompt = read_prompt(prompt_path)

    users = [prompt for _ in image_paths]
    raws = vision_llm.chat_batch(system="", users=users, image_paths=image_paths)

    parsed = []
    for r in raws:
        p = safe_json_load(r, prefer="array")
        parsed.append(p if isinstance(p, list) else [])
    return parsed


def filter_vision_against_text(
    vision_phens: List[Dict[str, Any]],
    text_phens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter vision phenotypes using text as a reference.

    - Concordant with text (same mention detected) -> reliability = high
    - Contradicted by negated text entry -> DROP
    - Only in vision -> reliability = low
    """
    text_mentions_present = {
        (p.get("mention") or "").lower()
        for p in text_phens
        if p.get("attribution") == "patient"
    }
    text_mentions_negated = {
        (p.get("mention") or "").lower()
        for p in text_phens
        if p.get("attribution") == "negated"
    }

    out = []
    for v in vision_phens:
        m = (v.get("mention") or "").lower()
        if not m:
            continue
        if m in text_mentions_negated:
            logger.info(f"Dropping vision phenotype contradicted by text negation: {m}")
            continue
        v_copy = dict(v)
        v_copy["attribution"] = "patient"
        v_copy["reliability"] = "high" if m in text_mentions_present else "low"
        v_copy["source"] = "vision"
        out.append(v_copy)
    return out
