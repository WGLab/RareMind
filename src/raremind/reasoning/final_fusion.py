"""Final ranking calibration after LLM reranking and reconciliation.

The reconciled rank is intentionally useful, but pairwise LLM decisions can
occasionally over-move candidates. This module blends the deterministic
Original score with the reconciled score so the final rank keeps both signals:

    final_score = w_original * norm(original_score)
                + w_reconciled * norm(reconciled_score)

The default weights are configured in ``configs/default.yaml``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from raremind.core.config import cfg_get


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0
    if hi <= lo:
        return pd.Series([1.0 if hi > 0 else 0.0] * len(values), index=values.index)
    return (values - lo) / (hi - lo)


def _first_existing(df: pd.DataFrame, preferred: str, fallbacks: list[str]) -> str | None:
    for col in [preferred] + fallbacks:
        if col and col in df.columns:
            return col
    return None


def apply_final_fusion(
    reconciled_df: pd.DataFrame,
    cfg: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return final fused ranking and metadata.

    If final ranking is disabled, the function still returns a dataframe with
    ``final_rank``/``final_score`` aliases so downstream scorecards and
    trajectory files have a stable schema.
    """
    final_cfg = cfg_get(cfg, "final_ranking", {})
    enabled = bool(cfg_get(final_cfg, "enabled", True))
    method = str(cfg_get(final_cfg, "method", "score_fusion"))
    original_weight = float(cfg_get(final_cfg, "original_weight", 0.65))
    reconciled_weight = float(cfg_get(final_cfg, "reconciled_weight", 0.35))

    out = reconciled_df.copy() if isinstance(reconciled_df, pd.DataFrame) else pd.DataFrame()
    if out.empty:
        return out, {
            "enabled": enabled,
            "method": method,
            "original_weight": original_weight,
            "reconciled_weight": reconciled_weight,
            "top_subtype": None,
        }

    original_col = _first_existing(
        out,
        str(cfg_get(final_cfg, "score_col_original", "total_score")),
        ["adjusted_score", "final_score_subtype", "reconciled_score"],
    )
    reconciled_col = _first_existing(
        out,
        str(cfg_get(final_cfg, "score_col_reconciled", "reconciled_score")),
        ["final_score_subtype", "total_score"],
    )

    if not enabled or method.lower() in {"disabled", "reconciled"}:
        score_col = reconciled_col or original_col
        out["final_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
        if "reconciled_rank" in out.columns:
            out["final_rank"] = pd.to_numeric(out["reconciled_rank"], errors="coerce").fillna(0).astype(int)
            out = out.sort_values(["final_rank", "disease_id"], ascending=[True, True]).reset_index(drop=True)
        else:
            out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
            out["final_rank"] = out.index + 1
    else:
        if original_col is None:
            original_col = reconciled_col
        if reconciled_col is None:
            reconciled_col = original_col
        if original_col is None or reconciled_col is None:
            out["final_original_score_norm"] = 0.0
            out["final_reconciled_score_norm"] = 0.0
            out["final_score"] = 0.0
        else:
            out["final_original_score_norm"] = _minmax(out[original_col])
            out["final_reconciled_score_norm"] = _minmax(out[reconciled_col])
            denom = max(1e-9, original_weight + reconciled_weight)
            ow = original_weight / denom
            rw = reconciled_weight / denom
            out["final_score"] = (
                ow * out["final_original_score_norm"]
                + rw * out["final_reconciled_score_norm"]
            )
            original_weight = ow
            reconciled_weight = rw

        if "rank" in out.columns:
            out["_final_tiebreak_rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(10**9)
        else:
            out["_final_tiebreak_rank"] = range(1, len(out) + 1)
        out = out.sort_values(
            ["final_score", "_final_tiebreak_rank", "disease_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        out["final_rank"] = out.index + 1
        out = out.drop(columns=["_final_tiebreak_rank"], errors="ignore")

    out["final_fusion_original_weight"] = original_weight
    out["final_fusion_reconciled_weight"] = reconciled_weight
    out["final_fusion_method"] = method if enabled else "disabled"
    top = out.iloc[0].to_dict() if len(out) else None
    return out, {
        "enabled": enabled,
        "method": method if enabled else "disabled",
        "original_weight": original_weight,
        "reconciled_weight": reconciled_weight,
        "score_col_original": original_col,
        "score_col_reconciled": reconciled_col,
        "top_subtype": top,
    }
