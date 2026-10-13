# 🧬 RareMind

### From multimodal patient evidence to ranked rare-disease hypotheses—and clinically actionable next steps

RareMind is a locally deployable, multi-agent rare-disease reasoning pipeline grounded in **RareGraph**, the structured knowledge graph developed for this project. It integrates clinical notes, phenotype terms, images, and genomic evidence; retrieves and adjudicates disease candidates; produces an auditable final ranking; and now converts its leading diagnostic hypotheses into normalized **next-test and next-step recommendations**.

> **Pipeline name:** RareMind  
> **Knowledge graph:** RareGraph  
> **Default final output:** ranked diagnoses **plus** ten normalized clinical actions

---

## ✨ What RareMind does

- 📝 Extracts phenotypes, demographics, family history, prior testing, and gene mentions from clinical notes
- 🖼️ Incorporates phenotype evidence from medical images when available
- 🧬 Accepts structured HPO terms and external genomic-ranking results
- 🔗 Grounds retrieval and reasoning in RareGraph, HPO, MONDO, OMIM, Orphanet, and GeneReviews-derived knowledge
- 🤖 Uses complementary agents for evidence auditing, pairwise adjudication, and group/subtype reconciliation
- 📊 Preserves every major rank transition in an auditable trajectory
- 🩺 Converts the final Top-10 disease groups into normalized next tests, referrals, and initial evaluations

RareMind is designed to move beyond a single Top-K accuracy number. The accompanying clinical-impact analyses evaluate:

1. ⏱️ **Diagnostic lead time and healthcare utilization**
2. 🤝 **Alignment with clinician differential diagnoses**
3. 🛟 **Recovery when the clinician differential misses the eventual diagnosis**
4. 🚫 **Deprioritization of diagnoses explicitly ruled out in the evaluated note**
5. 🧪 **Concordance between recommended and subsequently documented tests**

---

## 🧭 The RareMind pipeline

| Stage | Purpose | Principal output |
|---:|---|---|
| 1 | Multimodal clinical extraction | Phenotypes, demographics, family history, testing, genes, and image evidence |
| 2 | Ontology normalization and post-processing | Normalized HPO evidence, temporal context, inheritance, incongruity |
| 3 | Candidate retrieval and composite scoring | Broad RareGraph-grounded disease ranking |
| 3.5 | Non-gating case-signal assessment | Genetic/rare-disease versus nonspecific signal, eight raw features, and a brief explanation |
| 4 | Conditional frontier consultation | Targeted review of ambiguous, incongruous, or under-supported cases |
| 5 | Evidence audit | Supporting, contradicting, and missing expected evidence |
| 6 | Pairwise adjudication | Subtype- and group-level candidate comparisons |
| 7 | Rank aggregation | Aggregated subtype and disease-group rankings |
| 8 | Group/subtype reconciliation and final fusion | Calibrated global `final_rank` |
| 9A | Group-level clinical synthesis | Disease-group differential with representative subtypes and professional-review PDF |
| 9B | Exact subtype clinical synthesis | Existing exact Top-10 report, retained with a professional-review PDF |
| **10** | **Next-test and next-step synthesis** | **Ten normalized, cross-diagnosis clinical actions with provenance** |

Submitting a patient through `scripts/run_pipeline.py` runs the complete sequence from Stage 1 through Stage 10. Stage 10 is enabled by default.

---

## 🩺 Stage 10: from diagnosis to action

Stage 10 applies the same action-normalization strategy used in the retrospective next-test concordance analysis:

```text
Global final ranking
        ↓
Disease groups represented within final ranks 1–10
        ↓
RareGraph: testing + initial_evaluations
        ↓
Lexical canonicalization
        ↓
BioLORD semantic clustering (cosine similarity ≥ 0.90)
        ↓
Rank by number of supporting disease groups
        ↓
Top 10 next tests / evaluations / referrals
```

Examples of deterministic normalization include:

- `echo` / `echocardiography` → `echocardiogram`
- `6MWT` → `six minute walk test`
- `whole-exome sequencing` → `exome sequencing`
- `brain natriuretic peptide` → `BNP`

Semantically similar actions that remain lexically different are clustered with the already loaded BioLORD model. Each recommendation retains:

- its normalized label and observed aliases;
- the number of distinct disease groups supporting it;
- the best final rank among its supporting groups;
- the RareGraph field(s) that supplied it;
- the supporting group and representative disease IDs and names.

This stage generates recommendations from the model output alone. Reference future tests are used only for retrospective evaluation—not during patient inference.

### Stage 10 configuration

```yaml
next_steps:
  enabled: true
  disease_top_k: 10
  action_top_k: 10
  action_fields:
    - testing
    - initial_evaluations
  cluster_similarity_threshold: 0.90
```

---

## 🚀 Running RareMind

### 1. Prepare a dataset folder

At least one supported modality must be present. Clinical text is the usual starting point.

```text
inputs/
└── demo/
    ├── text/
    │   └── PATIENT_001.txt
    ├── free_hpo/
    │   └── PATIENT_001.txt          # optional
    ├── image/
    │   └── PATIENT_001.png          # optional
    └── vcf/
        └── PATIENT_001.vcf          # optional
```

The filename stem is the case identifier shared across modalities.

### 2. Configure local resources

Edit [`configs/default.yaml`](configs/default.yaml) to point to:

- the RareGraph JSON;
- the MONDO hierarchy and cross-mappings;
- the HPO ontology;
- local text and vision models;
- cache and output locations.

The default BioLORD model is `FremyCompany/BioLORD-2023`.

### 3. Run one patient

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --dataset demo \
  --case_id PATIENT_001
```

### 4. Run every discovered patient

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --dataset demo
```

On a SLURM cluster, the supplied wrapper submits the same complete Stage 1–10
workflow; no separate next-step command is required:

```bash
sbatch -p gpuq --gres=gpu:a100:1 \
  --wrap="bash scripts/run_agents.sh --dataset demo --input_dir inputs --output_dir outputs"
```

Useful overrides:

```bash
# Test a different local text model
python scripts/run_pipeline.py --dataset demo --text_model Qwen/Qwen3-8B

# Process only the first ten discovered cases
python scripts/run_pipeline.py --dataset demo --limit 10

# Recompute Stage 1 caches
python scripts/run_pipeline.py --dataset demo --overwrite_stage1_cache

# Resume an interrupted cohort without regenerating completed work
python scripts/run_pipeline.py --dataset demo --resume

# Control-test fast path: stop after Stage 3 ranking and Stage 3.5 classification
python scripts/run_pipeline.py --dataset control_test --control

# The SLURM wrapper forwards the same control flag
bash scripts/run_agents.sh --dataset control_test --control
```

Resume mode reuses valid Stage 9 and Stage 10 artifacts. When a case has a
complete saved Stage 8 checkpoint but lacks Stage 9 or Stage 10, RareMind runs
only the missing final stage(s). Cases interrupted before Stage 8 restart the
upstream workflow while reusing their Stage 1 prefetch cache.

`--control` runs extraction, normalization, composite ranking, and the Stage 3.5
case-signal classification, then exits before Frontier consultation, audit,
pairwise reranking, reconciliation, Stage 9 synthesis/PDF generation, and Stage
10. It writes `control_run_summary.json` for an explicit completion marker. If
combined with `--resume`, Stage 8 resume checkpoints are intentionally ignored;
Stage 1 caches may still be reused. Missing-genomics background jobs are not
launched in control mode.

> A Linux environment with an NVIDIA GPU is recommended for the local vLLM stages. BioLORD action normalization uses `sentence-transformers` and reuses the embedder already loaded for ontology normalization.

---

## 📦 Per-patient outputs

```text
outputs/<dataset>/<case_id>/
├── stage1_extraction.json
├── stage2_patient_evidence.json
├── stage3_composite_ranking.tsv
├── stage3_5_case_signal.json
├── stage5_audit_results.json
├── stage5_ranking_after_audit.tsv
├── stage6_pairwise_results_subtype.json
├── stage6_pairwise_results_group.json
├── stage7_reranked_subtype.tsv
├── stage7_reranked_group.tsv
├── stage8_reconciled_ranking.tsv
├── stage8_final_fusion.tsv
├── stage8_reconciled.json
├── stage9a_group_ranking.tsv
├── stage9_reasoning_report.json
├── stage9_reasoning_report.txt
├── stage9a_group_reasoning_report.json
├── stage9a_group_reasoning_report.txt
├── stage9a_group_reasoning_report.pdf
├── stage9b_subtype_reasoning_report.json
├── stage9b_subtype_reasoning_report.txt
├── stage9b_subtype_reasoning_report.pdf
├── stage10_next_steps.tsv
└── rank_trajectory.tsv
```

The dataset-level `summary.tsv` includes the Stage 3.5 case-signal class,
category, score and short reason; the leading diagnosis and final rank; the
highest-ranked next step; and the complete Top-10 action list.

### 🩺 Stage 3.5 case-signal assessment

Stage 3.5 computes four score-shape, three HPO-specificity, and one structured
family-history feature after Stage 3. It writes an explicitly uncalibrated
genetic/rare-disease-versus-nonspecific signal and a short explanation. It does
not validate the leading diagnosis, change any rank, skip any case, or stop the
pipeline. Raw features are retained for later cohort calibration. The
provisional v2 defaults place more weight on repeated HPO specificity, reduce
the influence of score shape and ordinary affected-relative history, and use a
75-point genetic-signal threshold. These settings separate the two supplied
synthetic controls but still require validation on confirmed rare-disease and
control cohorts before clinical interpretation.

### 🩺 Stage 9A and 9B clinical synthesis

Stage 9 now produces two clinician-facing reports without changing Stages 1–8:

- **Stage 9A** collapses the complete Stage 8 reconciled subtype ranking by
  `group_name`, keeps each group's best-ranked subtype position, removes
  duplicates, reindexes the resulting groups, and synthesizes the Top 10. It
  shows up to five ranked exact subtypes with their supplied audit and pairwise
  analyses.
- **Stage 9B** preserves the existing exact-subtype Top-10 synthesis. The legacy
  `stage9_reasoning_report.json` and `.txt` filenames remain unchanged for compatibility.

Both tracks reuse the loaded text model through three vLLM inference waves:

1. **Patient summary + family-history assessment** (family call omitted if empty).
2. **One assessment per Top-10 candidate**, submitted together in a native vLLM
   batch with rank-specific output budgets.
3. **Differential synthesis + prioritized next steps**, submitted together.

Each report track uses 14 generation requests for ten candidates with family
history, or 13 without it—not 14 sequential calls. Producing both 9A and 9B
therefore uses 28 requests (or 26 without family history), arranged in six
batched inference waves. No separate model is loaded. The current run's
configured text model is used (Qwen3-8B when configured), not a hard-coded
replacement. Actual latency depends on context length, GPU capacity and batching.

The report distinguishes **documented negations**, **undocumented findings** and
**prior-test constraints**. It uses full source-note text, normalized phenotypes,
available image/genomics evidence, family history, RareGraph entries, audit and
pairwise assessments, and reconciliation information. Missing evidence must not
be invented. Scores remain relative ranking values—not diagnostic probabilities.
These are prompt-level safeguards; clinical fidelity still requires validation.

- `stage9_reasoning_report.json` preserves `top_candidates`, `suggested_workup`,
  `reconciled` and `frontier` for compatibility. New fields include
  `synthesis_report`, `clinical_assessments`, `patient_summary`,
  `family_history_assessment`, `differential_synthesis`,
  `recommended_next_steps` and `synthesis_metadata`.
- `stage9_reasoning_report.txt` presents every candidate's complete generated
  assessment with reader-friendly rank, disease, and section headings, followed
  by a Python-generated pipeline transparency table. Historical JSON remains
  renderable.
- `stage9a_group_reasoning_report.pdf` and
  `stage9b_subtype_reasoning_report.pdf` are styled, paginated review documents
  with the Stage 3.5 case signal, an explicit research-use disclaimer, and a
  pipeline-transparency table. These are the only Stage 9 PDF outputs.
- Stage 9 next steps are an LLM synthesis of supplied top-five KG management
  evidence. They do **not** replace or consume the separate Stage 10 artifact.
- `stage9` in `configs/default.yaml` controls only synthesis sampling/budgets.
  `backend: native` reuses the in-process engine; `backend: endpoint` optionally
  submits concurrent requests to an existing OpenAI-compatible vLLM server.
- Empty/failed generations raise an explicit error rather than masquerading as
  a successful clinical report. Output-limit and missing-assessment warnings are
  recorded in metadata and displayed in the report. Oversized native contexts
  raise an error rather than silently truncating the clinical note.

Stage 9's seven editable prompt templates live in `configs/prompts/report/`.
`scorecard_synthesis.py` loads them for both normal pipeline runs and replay;
copy that directory along with the source code when uploading to your server.

#### Replay Stage 9 for existing Dr. Strong reviewer samples

Use `clinical_impact_analyses/notebooks/13_dr_strong_regenerate_stage9.ipynb`
with the existing master JSON exported by notebook 12. Set `REVIEW_JSON` in
the first cell and confirm the RareMind source, archived result root, KG, and
GPU/model settings. Upload `clinical_impact/stage9_replay.py` alongside it.

The notebook checks every peak/final sample before loading one text model,
reconstructs evidence from saved stages, and calls `build_scorecard` directly.
It never runs the pipeline host or Stages 1–8/10. Saved Final Ranks must agree
with the review packet; ground-truth annotations are not included in prompts.
Each sample receives matching `stage9_reasoning_report.json` and `.txt` files,
leaving its original scorecard and Stage 10 intact. The notebook also displays complete reports and
saves an updated reviewer master JSON plus progress logs for resuming.
Change `OVERWRITE=True` deliberately when changing synthesis settings or KG;
otherwise existing replay reports with matching evidence are reused.

For a fresh reviewer cohort that requires the true diagnosis to be within the
Final-Rank Top-10 in **both** the earliest qualifying peak note and the closest
pre-recommendation note, use
`clinical_impact_analyses/notebooks/14_select_dual_top10_and_complete_stage9.ipynb`.
It selects 30–50 patients independently of Stage 9 availability, prioritizes
exact MONDO pairs and longer lead times, reuses valid reports, and generates
only missing/invalid Stage 9 JSON/TXT reports before exporting review packets.

### Stage 10 output

**`stage10_next_steps.tsv`** is the single canonical Stage 10 artifact. It
contains the normalized Top-10 actions, aliases, supporting disease groups and
diseases, source RareGraph fields, ranking support, normalization metadata, and
the selected source disease groups.

---

## 🔬 Clinical-impact analyses

The [`clinical_impact_analyses`](clinical_impact_analyses) workspace contains the retrospective evaluation notebooks and reusable functions for:

- earliest computationally recoverable diagnostic signal;
- utilization accumulated before formal genetic-testing recommendation;
- clinician-differential alignment and missed-diagnosis recovery;
- note-aligned ruled-out negative controls;
- next-test lexical and BioLORD semantic concordance;
- publication-ready figures and review tables.

These analyses are intentionally separated from patient inference. Cohort labels and future documented tests never enter the RareMind ranking or Stage 10 recommendation process.

---

## 🧱 Repository map

```text
src/raremind/                  Canonical RareMind implementation
├── agents/                    Extraction agents
├── normalize/                 HPO, MONDO, temporal, and BioLORD normalization
├── kg/                        RareGraph loading and indexing
├── retrieval/                 Candidate-generation channels
├── scoring/                   Deterministic composite scoring
├── reasoning/                 Audit, adjudication, fusion, scorecard, Stage 10
└── orchestration/             End-to-end patient runner

scripts/run_pipeline.py        Main RareMind command-line entry point
configs/default.yaml           Runtime and Stage 10 configuration
clinical_impact_analyses/      Retrospective clinical-impact evaluations
```

New code should import the pipeline from `raremind`. A minimal historical
`raregraph` compatibility namespace is retained to avoid abruptly breaking
existing environments; **RareGraph** otherwise refers only to the knowledge
graph.

---

## 🛡️ Intended use

RareMind is a research decision-support system. Its rankings and suggested actions are not medical advice, are not a substitute for clinical judgment, and require review in the context of the complete patient record, prior testing, local practice, and test availability.

For protected health information, use only approved infrastructure and follow institutional privacy, security, and data-governance requirements.

---

## 🌟 In one sentence

**RareMind turns multimodal clinical evidence into an auditable rare-disease differential—and turns that differential into a normalized, traceable plan for what clinicians might evaluate next.**

---

## 📚 Publication and citation

The full RareMind manuscript is currently in preparation. Until it becomes
available, if you use or reference this work, please cite our
[related publication](https://dl.acm.org/doi/abs/10.1007/978-3-032-30813-9_82).
