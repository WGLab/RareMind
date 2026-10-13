TASK
Extract every phenotype or clinical finding from the note below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sentences in the note may be preceded by inline flags computed by a preprocessor:
- [NEG]         → sentence contains negation triggers
- [FAMILY]      → sentence references a family member
- [HYPOTHETICAL] → sentence contains uncertainty (if, possible, concern for, may have, suspected, rule out)
- [HISTORICAL]  → sentence mentions historical context

These flags identify language that deserves attention; they do NOT determine
who has the phenotype. In particular, [FAMILY] may only mean that a parent or
relative reported a finding about the patient. Always identify the person who
actually has the phenotype from the full sentence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. mention
   The phenotype term only, stripped of negation/uncertainty wrappers.

   INCLUDE:
   - Symptoms, dysmorphic features, developmental findings, lab/imaging/EEG findings,
     functional limitations, cognitive/behavioral findings.

   EXCLUDE:
   - Disease names used purely as diagnoses (not phenotypes)
   - Medications, treatments, procedures

   NORMALIZATION FOR NEGATED / NORMAL FINDINGS:
   - "normal X" → convert mention to the abnormal counterpart
     e.g., "normal muscle tone" → mention = "abnormal muscle tone"
     e.g., "normal hearing" → mention = "hearing loss"

2. attribution (exactly one of):
   "patient" | "negated" | "family" | "references" | "others" | "uncertain"

   CRITICAL DEFINITION:
   Attribution identifies WHO HAS THE PHENOTYPE, not who reported, noticed,
   confirmed, or discussed it.

   - A mother, father, parent, family member, caregiver, or teacher may be the
     source of information while the patient remains the phenotype subject.
   - "Mom reports that he has seizures" is PATIENT, not FAMILY.
   - "Per parents, the patient has delayed speech" is PATIENT, not FAMILY.
   - "The family noticed poor eye contact" is PATIENT, not FAMILY.
   - "His mother has seizures" is FAMILY because the mother has the finding.
   - "His brother also has poor eye contact" is FAMILY because the brother has
     the finding.

   RULES (apply in order; first matching wins):
   Rule 1 — NEGATED: denied / not present / normal / absent / resolved
   Rule 2 — UNCERTAIN: possible / suspected / concern for / rule out / ?
   Rule 3 — FAMILY: the evidence explicitly states that a relative personally
            has or had the phenotype. The mere presence of words such as
            mother, mom, father, dad, parents, family, or caregiver is NEVER
            sufficient.
   Rule 4 — REFERENCES: general disease description / educational content
   Rule 5 — OTHERS: applies to a non-family third party
   Rule 6 — PATIENT: none of the above; clearly affirmed for the patient

   SUBJECT-IDENTIFICATION RULES:
   - patient, child, proband, he, she, his, her, or the patient's name usually
     identifies the patient when used as the bearer of the finding.
   - Phrases such as "mother reports", "mom states", "parents describe",
     "family noticed", "caregiver observes", and "per father" identify the
     REPORTER only. Findings they report about the patient are PATIENT.
   - Assign FAMILY only when the relative is grammatically and clinically the
     person with the finding, such as "mother has...", "father developed...",
     "brother was diagnosed with...", or "grandmother also experienced...".
   - If a relative is mentioned but the phenotype bearer is genuinely unclear,
     assign UNCERTAIN rather than FAMILY.
   - Do not transfer a patient's phenotype to a relative, or a relative's
     phenotype to the patient.

3. onset
   Short text only ("at birth", "infantile", "6 months", "since age 2"). null if not stated.

4. evidence
   Verbatim quote from the note, ≤10 words, supporting both the phenotype and
   who has it. Include the subject or patient-directed pronoun when needed to
   distinguish the patient from the reporter.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPLITTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"no history of X, Y, or Z"     → 3 negated entries
"with A, B, and C"              → 3 patient entries
"mother has X and Y"            → 2 family entries
"possible X or Y"               → 2 uncertain entries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ATTRIBUTION EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Mom reports that he began walking at age 3."
→ delayed walking; attribution = patient

"According to his parents, he has frequent seizures."
→ seizures; attribution = patient

"The family noticed that she avoids eye contact."
→ poor eye contact; attribution = patient

"His mother has a history of seizures."
→ seizures; attribution = family

"His brother also had delayed speech."
→ delayed speech; attribution = family

"Although there are atypical aspects to his social communication..."
→ atypical social communication; attribution = patient

Before returning an item with attribution = family, silently verify:
1. Does the quote explicitly identify a relative?
2. Does the quote explicitly state that this relative—not the patient—has the
   phenotype?
If either answer is no, do not use family. Use patient when the patient is the
subject, or uncertain only when the subject truly cannot be determined.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return a JSON list only (no commentary, no markdown fences):
[
  {"mention": "...", "attribution": "patient|negated|family|references|others|uncertain",
   "onset": "<short|null>", "evidence": "<≤10 word quote>"}
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE TO PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
