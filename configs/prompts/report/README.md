# Stage 9 clinical report prompts

Both the main pipeline and the Stage 9 replay notebook load these files via
`src/raremind/reasoning/scorecard_synthesis.py`.

| File | Purpose |
| --- | --- |
| `system.txt` | Shared instructions and evidence safeguards |
| `patient_summary.txt` | Patient evidence summary |
| `family_history.txt` | Family-history assessment, when available |
| `candidate_assessment.txt` | Each fixed-rank candidate's clinical assessment |
| `group_candidate_assessment.md` | Each fixed-rank disease group's assessment and subtype resolution |
| `differential_synthesis.txt` | Overall differential synthesis |
| `next_steps.txt` | Stage 9 clinical next steps (separate from Stage 10) |

Edit the wording here; no Python edits are needed. Preserve `${placeholder}`
names: Python supplies those values from the archived/current evidence. For a
literal dollar sign in a user template, write `$$`. Ordinary JSON braces need
no escaping. The system prompt has no substitution and can use literal dollars.

Files are read during generation without a persistent cache. Paths are resolved
relative to the RareMind source checkout, not the working directory. Upload
this folder with the updated source to the GPU server. Missing, empty, or invalid
templates raise errors instead of silently using old embedded prompts.

To apply edits to already-generated reviewer reports, set `OVERWRITE=True` in
notebook 13. Otherwise it intentionally resumes from the saved reports.
