"""Stage 0: KG precomputation (one-time at pipeline load).

Produces:
  - disease_hallmarks     : per disease, top-IC hallmark HPO IDs
  - disease_hallmark_names: display names of hallmarks (for prompts)
  - disease_exclusions    : per disease, HPOs curated as ABSENT (polarity=absent)
  - disease_cards         : per disease, compact string card used in prompts
  - disease_branches      : per disease, IC-weighted branch profile (curated phens)
  - disease_all_branches  : per disease, branches of ALL annotated phens
  - pair_frequency        : global (hpo_a, hpo_b) → count_of_diseases table
  - pair_to_diseases      : (hpo_a, hpo_b) → set of disease ids
  - pathognomonic_hpos    : hpo_id → list of diseases (≤3 diseases)
  - characteristic_disease_count : hpo_id → int
  - narrative_texts       : per disease, concatenated narrative text (if available)
  - onset_hpos            : per disease, list of onset-related HPOs
  - hpo_to_diseases       : inverse index (HPO → [diseases annotating it as characteristic])
  - hpo_to_diseases_all   : inverse index for ALL phenotypes (any importance)
  - gene_to_diseases      : gene -> [diseases]
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from raremind.normalize.hpo_ontology import HpoOntology

logger = logging.getLogger(__name__)


# HPO "onset" root term — descendants of this root are all onset modifiers
ONSET_ROOT = "HP:0003674"


IMPORTANCE_WEIGHT = {
    "characteristic": 1.0,
    "supportive": 0.75,
    "incidental": 0.35,
}

FREQUENCY_WEIGHT = {
    "very_common": 1.0,
    "common": 0.9,
    "more_than_half": 0.7,
    "occasional": 0.4,
    "rare": 0.2,
    "": 0.5,
    None: 0.5,
}

HIGH_FREQ = {"very_common", "common", "more_than_half"}
NARRATIVE_SECTION_KEYS = (
    "definition",
    "narrative",
    "description",
    "summary",
    "diagnosis",
    "testing",
    "treatment",
    "surveillance",
    "initial_evaluations",
    "differential",
    "differentials",
    "agents_to_avoid",
    "pregnancy_management",
    "pathways",
    "prognosis_natural_history",
    "management",
    "animal",
)
NARRATIVE_TEXT_KEYS = (
    "description",
    "summary",
    "text",
    "definition",
    "feature",
    "comparison",
    "note",
    "notes",
    "indication",
    "finding",
    "rationale",
    "differential_disease_state",
    "recurrence_risk",
    "expressivity",
    "penetrance",
)


@dataclass
class KGIndex:
    disease_hallmarks: Dict[str, List[str]] = field(default_factory=dict)
    disease_hallmark_names: Dict[str, List[str]] = field(default_factory=dict)
    disease_exclusions: Dict[str, List[str]] = field(default_factory=dict)
    disease_cards: Dict[str, str] = field(default_factory=dict)
    disease_branches: Dict[str, Dict[str, float]] = field(default_factory=dict)
    disease_all_branches: Dict[str, Set[str]] = field(default_factory=dict)
    disease_phenotype_hpos: Dict[str, Set[str]] = field(default_factory=dict)
    disease_characteristic_hpos: Dict[str, Set[str]] = field(default_factory=dict)
    disease_genes: Dict[str, Set[str]] = field(default_factory=dict)
    disease_inheritance: Dict[str, List[str]] = field(default_factory=dict)
    disease_name: Dict[str, str] = field(default_factory=dict)
    disease_narrative: Dict[str, str] = field(default_factory=dict)
    disease_onset_hpos: Dict[str, List[str]] = field(default_factory=dict)
    disease_group: Dict[str, str] = field(default_factory=dict)
    disease_aliases: Dict[str, List[str]] = field(default_factory=dict)

    # Inverse + pair indexes
    hpo_to_diseases: Dict[str, Set[str]] = field(default_factory=dict)
    hpo_to_diseases_all: Dict[str, Set[str]] = field(default_factory=dict)
    gene_to_diseases: Dict[str, Set[str]] = field(default_factory=dict)
    pair_frequency: Dict[Tuple[str, str], int] = field(default_factory=dict)
    pair_to_diseases: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    pathognomonic_hpos: Dict[str, List[str]] = field(default_factory=dict)
    characteristic_disease_count: Dict[str, int] = field(default_factory=dict)


def _iter_phenotypes(phen_block: Any):
    if isinstance(phen_block, dict):
        for name, info in phen_block.items():
            if isinstance(info, dict):
                yield name, info
    elif isinstance(phen_block, list):
        for info in phen_block:
            if isinstance(info, dict):
                yield info.get("name") or info.get("hpo_name") or info.get("hpo", ""), info


def _safe_hpo(info: Dict[str, Any]) -> Optional[str]:
    h = info.get("hpo") or info.get("hpo_id")
    if h and str(h).startswith("HP:"):
        return str(h)
    return None


def _hpo_exists(hid: str, hpo: HpoOntology) -> bool:
    """Return True if the HPO id exists in the loaded ontology."""
    if not hid:
        return False

    # Most likely fast path
    if hasattr(hpo, "id_to_name") and hid in hpo.id_to_name:
        return True

    # Fallback to graph membership if needed
    directed = getattr(hpo, "directed", None)
    if directed is not None:
        try:
            return hid in directed
        except Exception:
            pass

    return False


def _phen_weight(info: Dict[str, Any]) -> float:
    imp = info.get("importance", "incidental")
    freq = info.get("frequency")
    return IMPORTANCE_WEIGHT.get(imp, 0.35) * FREQUENCY_WEIGHT.get(freq, 0.5)


def _extract_disease_name(entry: Dict[str, Any], disease_id: str) -> str:
    meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
    return (
        entry.get("preferred_title")
        or entry.get("name")
        or entry.get("label")
        or meta.get("preferred_title")
        or meta.get("name")
        or meta.get("label")
        or disease_id
    )


def _extract_aliases(entry: Dict[str, Any]) -> List[str]:
    meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
    aliases: List[str] = []
    for block in (
        entry.get("alternative_titles"),
        entry.get("aliases"),
        entry.get("synonyms"),
        meta.get("synonyms"),
        meta.get("aliases"),
    ):
        if isinstance(block, str):
            aliases.extend([x.strip() for x in block.split("|") if x.strip()])
        elif isinstance(block, list):
            aliases.extend([str(x).strip() for x in block if str(x).strip()])
    seen: Set[str] = set()
    out: List[str] = []
    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            out.append(alias)
    return out


def _flatten_text_snippets(value: Any, out: List[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return
    if isinstance(value, list):
        for item in value:
            _flatten_text_snippets(item, out)
        return
    if isinstance(value, dict):
        for key in NARRATIVE_TEXT_KEYS:
            if key in value:
                _flatten_text_snippets(value.get(key), out)
        for nested_key in ("comparison", "details", "evidence"):
            if nested_key in value:
                _flatten_text_snippets(value.get(nested_key), out)
        for key, nested in value.items():
            if key in NARRATIVE_TEXT_KEYS or key in {"comparison", "details", "evidence"}:
                continue
            if isinstance(nested, (dict, list)):
                _flatten_text_snippets(nested, out)


def _extract_narrative(entry: Dict[str, Any]) -> str:
    snippets: List[str] = []
    for key in NARRATIVE_SECTION_KEYS:
        if key in entry:
            _flatten_text_snippets(entry.get(key), snippets)

    seen: Set[str] = set()
    unique: List[str] = []
    for snippet in snippets:
        norm = " ".join(snippet.split()).strip()
        if not norm:
            continue
        key = norm.lower()
        if key not in seen:
            seen.add(key)
            unique.append(norm)
    return "\n".join(unique)


def _extract_genes(g_block: Any) -> Set[str]:
    gset: Set[str] = set()
    if isinstance(g_block, dict):
        gset.update({str(g).upper() for g in g_block.keys() if g})
    elif isinstance(g_block, list):
        for item in g_block:
            if isinstance(item, dict):
                gene = item.get("gene") or item.get("symbol") or item.get("name")
                if gene:
                    gset.add(str(gene).upper())
            elif item:
                gset.add(str(item).upper())
    elif isinstance(g_block, str) and g_block.strip():
        gset.add(g_block.strip().upper())
    return gset


def _flatten_inheritance_modes(block: Any) -> List[str]:
    """Extract inheritance mode strings from KG inheritance variants."""
    modes: List[str] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                modes.append(text)
            return
        if isinstance(value, list):
            for item in value:
                _add(item)
            return
        if isinstance(value, dict):
            for key in ("mode", "label", "name", "inheritance", "mode_of_inheritance"):
                if key in value:
                    _add(value.get(key))
            return

    _add(block)
    seen: Set[str] = set()
    out: List[str] = []
    for mode in modes:
        key = mode.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(mode.strip())
    return out


def precompute_kg_index(
    kg: Dict[str, Dict[str, Any]],
    hpo: HpoOntology,
    hallmark_display_limit: int = 15,
    pathognomonic_max_diseases: int = 3,
    ancestor_depth_for_pairs: int = 2,
) -> KGIndex:
    idx = KGIndex()

    skipped_unknown_hpos = 0
    skipped_unknown_hpos_examples: Set[str] = set()

    # onset descendants (for per-disease onset HPO extraction)
    onset_descendants = (
        hpo.get_descendants(ONSET_ROOT, include_self=True)
        if _hpo_exists(ONSET_ROOT, hpo)
        else set()
    )

    # -----------------------------------------------------------
    # Pass 1: per-disease features
    # -----------------------------------------------------------
    for did, entry in kg.items():
        if not isinstance(entry, dict):
            continue

        name = _extract_disease_name(entry, did)
        idx.disease_name[did] = name
        idx.disease_aliases[did] = _extract_aliases(entry)

        # Narrative
        nar = _extract_narrative(entry)
        if isinstance(nar, str) and nar.strip():
            idx.disease_narrative[did] = nar.strip()

        # Group assignments are applied from data/mondo/hierarchy.json by the host.
        idx.disease_group[did] = ""

        # Genes
        g_block = entry.get("genes", {}) or {}
        gset = _extract_genes(g_block)
        idx.disease_genes[did] = gset
        for g in gset:
            idx.gene_to_diseases.setdefault(g, set()).add(did)

        # Inheritance
        inh = entry.get("inheritance") or entry.get("mode_of_inheritance") or []
        idx.disease_inheritance[did] = _flatten_inheritance_modes(inh)

        # Phenotypes
        hallmarks_ranked: List[Tuple[float, str, str, Dict[str, Any]]] = []
        exclusions: List[str] = []
        all_hpos: Set[str] = set()
        charact_hpos: Set[str] = set()
        branch_profile: Dict[str, float] = defaultdict(float)
        all_branches: Set[str] = set()
        onset_hpos_found: List[str] = []

        for phen_name, info in _iter_phenotypes(entry.get("phenotypes", {})):
            hid = _safe_hpo(info)
            if not hid:
                continue

            if not _hpo_exists(hid, hpo):
                skipped_unknown_hpos += 1
                if len(skipped_unknown_hpos_examples) < 20:
                    skipped_unknown_hpos_examples.add(hid)
                continue

            polarity = info.get("polarity", "present")
            if polarity == "absent":
                exclusions.append(hid)
                continue

            all_hpos.add(hid)

            for b in hpo.get_branches(hid):
                all_branches.add(b)

            is_predef = bool(info.get("is_predefined", False))
            importance = info.get("importance", "incidental")
            freq = info.get("frequency", "")

            if importance == "characteristic":
                charact_hpos.add(hid)
                idx.characteristic_disease_count[hid] = idx.characteristic_disease_count.get(hid, 0) + 1

            # Hallmark = is_predefined + characteristic + high-freq
            if is_predef and importance == "characteristic" and freq in HIGH_FREQ:
                ic = hpo.get_ic(hid)
                hallmarks_ranked.append((ic, hid, phen_name, info))

            # Branch profile (curated is_predefined only)
            if is_predef:
                w = _phen_weight(info)
                ic = hpo.get_ic(hid)
                for b in hpo.get_branches(hid):
                    branch_profile[b] += ic * w

            # Inverse indexes (any phenotype -> disease)
            idx.hpo_to_diseases_all.setdefault(hid, set()).add(did)
            if importance in ("characteristic", "supportive"):
                idx.hpo_to_diseases.setdefault(hid, set()).add(did)

            if hid in onset_descendants:
                onset_hpos_found.append(hid)

        # Fallback: if no hallmarks, use any is_predefined characteristic
        if not hallmarks_ranked:
            for phen_name, info in _iter_phenotypes(entry.get("phenotypes", {})):
                hid = _safe_hpo(info)
                if not hid:
                    continue
                if not _hpo_exists(hid, hpo):
                    continue
                if info.get("polarity") == "absent":
                    continue
                if not info.get("is_predefined", False):
                    continue
                if info.get("importance") != "characteristic":
                    continue
                ic = hpo.get_ic(hid)
                hallmarks_ranked.append((ic, hid, phen_name, info))

        hallmarks_ranked.sort(reverse=True)  # by IC descending
        idx.disease_hallmarks[did] = [h[1] for h in hallmarks_ranked]
        idx.disease_hallmark_names[did] = [h[2] for h in hallmarks_ranked]
        idx.disease_exclusions[did] = exclusions
        idx.disease_phenotype_hpos[did] = all_hpos
        idx.disease_characteristic_hpos[did] = charact_hpos
        idx.disease_branches[did] = dict(branch_profile)
        idx.disease_all_branches[did] = all_branches
        idx.disease_onset_hpos[did] = onset_hpos_found

        # Compact card
        idx.disease_cards[did] = _make_compact_card(
            name=name,
            disease_id=did,
            hallmarks_ranked=hallmarks_ranked[:hallmark_display_limit],
            exclusions=[(hpo.get_name(h), h) for h in exclusions],
            genes=list(gset)[:8],
            inheritance=idx.disease_inheritance[did],
            onset_hpos=[hpo.get_name(h) for h in onset_hpos_found[:5]],
            aliases=idx.disease_aliases.get(did, [])[:6],
        )

    # -----------------------------------------------------------
    # Pass 2: co-occurrence pair frequency
    # -----------------------------------------------------------
    logger.info("Computing co-occurrence pair frequency ...")
    for did, hps in idx.disease_characteristic_hpos.items():
        # Extra safety: keep only HPOs that actually exist in ontology
        valid_hps = [h for h in hps if _hpo_exists(h, hpo)]

        # take top-30 characteristic HPOs by IC for bounded compute
        sorted_hps = sorted(valid_hps, key=lambda h: hpo.get_ic(h), reverse=True)[:30]

        if len(sorted_hps) < 2:
            continue

        # Expand each hpo to itself + ancestors (up to ancestor_depth_for_pairs)
        expansions: Dict[str, Set[str]] = {}
        for h in sorted_hps:
            anc = _limited_ancestors(h, hpo, depth=ancestor_depth_for_pairs)
            expansions[h] = anc | {h}

        # For each pair of distinct hpos (in KG), add one expanded-cross-product entry
        for a, b in combinations(sorted_hps, 2):
            for ea in expansions[a]:
                for eb in expansions[b]:
                    if ea == eb:
                        continue
                    key = tuple(sorted((ea, eb)))
                    idx.pair_frequency[key] = idx.pair_frequency.get(key, 0) + 1
                    idx.pair_to_diseases.setdefault(key, set()).add(did)

    # -----------------------------------------------------------
    # Pass 3: pathognomonic flagging
    # -----------------------------------------------------------
    for hid, diseases in idx.hpo_to_diseases.items():
        if 1 <= len(diseases) <= pathognomonic_max_diseases:
            idx.pathognomonic_hpos[hid] = sorted(diseases)

    if skipped_unknown_hpos > 0:
        logger.warning(
            "Skipped %d phenotype annotations whose HPO IDs were not found in the loaded ontology. "
            "Examples: %s",
            skipped_unknown_hpos,
            ", ".join(sorted(skipped_unknown_hpos_examples)) if skipped_unknown_hpos_examples else "N/A",
        )

    logger.info(
        f"KG precompute done: {len(idx.disease_name)} diseases, "
        f"{len(idx.pair_frequency)} co-occurrence pairs, "
        f"{len(idx.pathognomonic_hpos)} pathognomonic HPOs"
    )
    return idx


def _limited_ancestors(hid: str, hpo: HpoOntology, depth: int) -> Set[str]:
    """BFS up the HPO DAG for at most `depth` levels.

    Safe against KG HPO ids that are missing from the loaded ontology graph.
    """
    if not _hpo_exists(hid, hpo):
        return set()

    frontier = {hid}
    collected: Set[str] = set()

    for _ in range(depth):
        new_frontier = set()
        for node in frontier:
            if not _hpo_exists(node, hpo):
                continue
            try:
                parents = hpo.directed.successors(node)
            except Exception:
                continue
            for p in parents:
                if p not in collected:
                    collected.add(p)
                    new_frontier.add(p)
        if not new_frontier:
            break
        frontier = new_frontier

    return collected


def _make_compact_card(
    name: str,
    disease_id: str,
    hallmarks_ranked,
    exclusions,
    genes,
    inheritance,
    onset_hpos,
    aliases,
) -> str:
    """Produce the ~200-token compact card used in audit/pairwise prompts."""
    lines: List[str] = []
    lines.append(f"Disease: {name} ({disease_id})")
    if aliases:
        lines.append("Aliases: " + ", ".join(aliases))
    if genes:
        lines.append("Genes: " + ", ".join(genes))
    if inheritance:
        lines.append("Inheritance: " + ", ".join(inheritance))
    if onset_hpos:
        lines.append("Typical onset: " + ", ".join(onset_hpos))

    lines.append("Hallmark phenotypes:")
    for ic, hid, phen_name, info in hallmarks_ranked:
        freq = info.get("frequency", "")
        imp = info.get("importance", "")
        lines.append(f"  - {phen_name} ({hid}) [IC={ic:.1f}; {imp}; {freq}]")
    if not hallmarks_ranked:
        lines.append("  (no curated hallmarks in KG)")

    if exclusions:
        lines.append("Typically absent / excluded:")
        for ename, eid in exclusions[:10]:
            lines.append(f"  - {ename} ({eid})")

    return "\n".join(lines)
