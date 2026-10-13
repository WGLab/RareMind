# Group-level candidate assessment

## Task

Generate a clinical assessment of one fixed disease group. Preserve the supplied group rank.

## Patient summary

${patient_summary}

## Family-history summary

${family_summary}

## Disease group

- Name: `${group_name}`
- MONDO ID: `${group_id}`
- Group rank: `${rank}`
- Displayed groups: `${n_candidates}`
- Represented members: `${member_count}`

## Ranked representative subtypes and upstream analyses

${representative_subtypes}

Each subtype record may include its Stage 3 rank, audit-adjusted rank, pairwise
rank, Stage 8 reconciled rank, structured audit assessment, and pairwise
comparisons. Use these supplied analyses when explaining the group. Do not
invent evidence for subtypes that is not present here or in the knowledge graph.

## Group and representative-subtype knowledge-graph context

${group_kg_context}

## Group pairwise comparisons

${pairwise}

## Group score fields

These are relative values, not probabilities.

${scores}

${patient_evidence}

## Structured family history

${family_history}

## Required assessment sections

Use exactly these labeled sections:

- `GROUP OVERVIEW:` Define the shared syndrome or mechanism using supplied knowledge-graph evidence only.
- `EVIDENCE SUPPORTING THE GROUP:` Connect documented patient findings and supplied upstream subtype analyses to evidence shared by or representative of the group. Distinguish group-wide evidence from subtype-specific evidence.
- `DOCUMENTED CONTRADICTIONS OR NEGATIVE TESTING:` Include only explicit patient negatives or completed tests. Do not treat undocumented findings as absent.
- `SUBTYPE RESOLUTION:` Name the ranked representative subtypes, including their supplied Stage 8 ranks, and explain which remain plausible and which supplied findings, genes, tests, or examinations could distinguish them. Do not claim that one subtype is established.
- `INFORMATION TO CLARIFY:` Ask only the most discriminating unresolved questions supported by the supplied context.
- `OVERALL ASSESSMENT:` Give a concise patient-specific explanation of why the group deserves its fixed rank. The group score is not a diagnostic probability. Do not change the ranking.

