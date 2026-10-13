# Stage 3.5 case-signal interpretation

RareMind has already computed a fixed, uncalibrated case-signal assessment.

## Fixed assessment

- Prediction: `${prediction}`
- Signal category: `${category}`
- Fixed score: `${score}/100`

## Eight features

${features}

## Component scores

${components}

## Required output

Return one JSON object with exactly these keys:

```json
{
  "summary": "Two concise sentences explaining whether this looks rare-disease/genetic or nonspecific.",
  "reasons": ["Two to four feature-grounded reasons."],
  "caveat": "One sentence stating this is not a diagnostic probability and does not validate the top-ranked disease."
}
```

Do not change the supplied prediction or score. Do not name a diagnosis. Do not
infer undocumented family history. Absence of family history is neutral rather
than exclusionary for simplex, recessive, or de novo disease.

