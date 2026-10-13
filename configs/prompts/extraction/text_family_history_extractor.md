TASK
Extract only explicitly documented medical family history from the supplied family-history candidate passages.

DEFAULT DECISION
The expected output for most notes is []. Return [] unless the text explicitly states that a family member has, had, carries, tested positive or negative for, or is explicitly unaffected by a disease, phenotype, or gene. A Family History heading alone is not evidence and may still require [].

OUTPUT (strict JSON list, no commentary)
[
  {
    "relation": "father|mother|brother|sister|son|daughter|cousin|uncle|aunt|grandfather|grandmother|other",
    "diseases": ["verbatim or near-verbatim disease name", ...],
    "affected_systems": ["explicitly documented system", ...],
    "phenotypes": ["verbatim or near-verbatim phenotype", ...],
    "genes": ["GENE1", ...],
    "age": "<explicitly stated age or null>",
    "affected": true|false,
    "evidence": "<exact verbatim quote of 40 words or fewer>"
  }
]

RELATION NORMALIZATION
- mother, mom, mum, mama -> mother
- father, dad, daddy, papa -> father
- grandmother, grandma, granny, nana -> grandmother
- grandfather, grandpa, granddad -> grandfather
- daughter/girl child -> daughter when explicitly a relative
- son/boy child -> son when explicitly a relative
- parents -> create separate mother and father records only when the same medical statement explicitly applies to both parents; use the same quote.
- sibling(s), relatives, family members, or another collective/unspecified relationship -> other; do not invent a specific relative.
- Preserve maternal or paternal context in the evidence, but normalize the relation to grandmother, grandfather, aunt, uncle, or cousin as appropriate.

QUALIFYING EVIDENCE
Include a record only when all of the following are true:
1. A family relationship is explicit in the evidence quote.
2. The quote explicitly describes that relative's own disease, phenotype, gene, positive/negative test result, or unaffected status.
3. Every populated disease, phenotype, gene, and age is supported by the evidence quote.
4. The evidence is copied verbatim from the supplied text and is no longer than 40 words.

For affected=true, at least one disease, phenotype, or gene must be explicitly documented. An affected system alone is insufficient.

For affected=false, the quote must explicitly say that the relative is unaffected, healthy, asymptomatic, or tested negative for the specified condition. Do not convert a general negative family-history statement into invented unaffected relatives.

MANDATORY EXCLUSIONS
Return [] when the text only says that a relative:
- accompanied the patient;
- attended the visit;
- provided, reported, confirmed, or translated the patient's history;
- consented to testing or treatment;
- was contacted;
- lives with, transports, or cares for the patient;
- is mentioned without their own clinical or genetic finding.

Do not infer family history from:
- the patient's own disease, phenotype, or gene;
- an inheritance pattern or recurrence-risk discussion;
- general disease knowledge;
- a suspected diagnosis;
- genetic counseling language;
- a pedigree that is not described in the supplied text;
- "family history reviewed", "family history unavailable", "unknown family  history", "adopted", or "noncontributory" without a qualifying relative.

Never include the patient. Never paraphrase or create evidence. Never fill a field using outside medical knowledge. If any requirement is uncertain, omit the record. If no record satisfies every requirement, return [].
Never assume the phenotypes or conditions of family members unless they are explictly included. No assumption at all here.

Before returning each record, silently verify:
- the relative is explicit;
- the relative's own medical status is explicit;
- the quote occurs verbatim in the supplied text;
- every populated field is supported by that quote.

If any check fails, remove the record. Output JSON only.
