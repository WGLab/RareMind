"""Utilities for running vision extraction before the text model is loaded."""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from raremind.agents.vision_agents import run_vision_extractor_batch
from raremind.llm.vision_api_client import ApiVisionClient
from raremind.llm.vllm_vision_client import VllmVisionClient

logger = logging.getLogger(__name__)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)


def _build_vision_client(cfg: Any):
    vision_cfg = _cfg_get(cfg.models, "vision_llm")
    runtime_cfg = _cfg_get(cfg, "vision", {})
    provider = str(_cfg_get(runtime_cfg, "provider", "local")).lower()

    if provider in {"local", "vllm"}:
        try:
            import torch

            tensor_parallel_size = torch.cuda.device_count() or _cfg_get(
                vision_cfg, "tensor_parallel_size", 1
            )
        except Exception:
            tensor_parallel_size = _cfg_get(vision_cfg, "tensor_parallel_size", 1)
        return VllmVisionClient(
            model_path=_cfg_get(vision_cfg, "model_name"),
            temperature=_cfg_get(vision_cfg, "temperature", 0.2),
            max_tokens=_cfg_get(vision_cfg, "max_tokens", 500),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=_cfg_get(vision_cfg, "gpu_memory_utilization", 0.85),
            max_batch_size=_cfg_get(vision_cfg, "max_batch_size", 4),
        )

    if provider in {"openrouter", "openai"}:
        return ApiVisionClient(
            provider=provider,
            model_name=_cfg_get(runtime_cfg, "model_name", _cfg_get(vision_cfg, "model_name")),
            api_key_env=_cfg_get(runtime_cfg, "api_key_env", "OPENROUTER_API_KEY"),
            api_base_url=_cfg_get(runtime_cfg, "api_base_url", "https://openrouter.ai/api/v1"),
            temperature=_cfg_get(vision_cfg, "temperature", 0.2),
            max_tokens=_cfg_get(vision_cfg, "max_tokens", 500),
            timeout_seconds=_cfg_get(runtime_cfg, "timeout_seconds", 120),
            max_batch_size=_cfg_get(vision_cfg, "max_batch_size", 4),
        )

    raise ValueError(f"Unknown vision provider: {provider}")


def release_vision_client(client: Any) -> None:
    """Best-effort GPU cleanup after vision extraction."""
    for attr in ("llm", "processor", "sampling_params"):
        if hasattr(client, attr):
            try:
                delattr(client, attr)
            except Exception:
                pass
    del client
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def prefetch_vision_for_cases(
    cfg: Any,
    cases: Iterable[Tuple[str, str, str]],
    prompt_dir: str = "configs/prompts",
    overwrite: bool = False,
) -> Dict[str, str]:
    """Run vision extraction for all image cases and write cache JSON files.

    Args:
        cases: tuples of (case_id, image_path, cache_path).
    """
    pending: List[Tuple[str, str, str]] = []
    for case_id, image_path, cache_path in cases:
        img = Path(image_path)
        cache = Path(cache_path)
        if not img.exists():
            continue
        if cache.exists() and not overwrite:
            continue
        pending.append((case_id, str(img), str(cache)))

    if not pending:
        return {}

    client = _build_vision_client(cfg)
    try:
        image_paths = [item[1] for item in pending]
        outputs = run_vision_extractor_batch(client, image_paths, prompt_dir)
        written: Dict[str, str] = {}
        for (case_id, image_path, cache_path), phenotypes in zip(pending, outputs):
            cache = Path(cache_path)
            cache.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "case_id": case_id,
                "image_path": image_path,
                "vision_phenotypes_raw": phenotypes,
            }
            cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written[case_id] = str(cache)
        logger.info(f"Prefetched vision phenotypes for {len(written)} cases")
        return written
    finally:
        release_vision_client(client)
