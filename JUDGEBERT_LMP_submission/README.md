# JUDGEBERT — Legal Meaning Preservation (LMP)

## Overview
JUDGEBERT is a research prototype that compares an original legal sentence
with a simplified version in the same language. It combines multilingual
sentence similarity with rule-based warnings for selected legal-detail changes.

## Current features
- English and Hindi text input in the interactive demo.
- Multilingual sentence embeddings using
  `paraphrase-multilingual-MiniLM-L12-v2`.
- Semantic similarity calculation.
- Warnings for possible differences in negation, obligation/permission terms,
  numbers/dates/currency, and selected legal-party terms.
- Preliminary heuristic score from 1 to 10.
- CSV export of evaluation and demo results.

## Method
1. Encode the original and simplified text.
2. Calculate cosine similarity between their sentence embeddings.
3. Compare selected legal-sensitive terms.
4. Produce a preliminary score and warning flags.

## Important limitations
- The score is heuristic and has not been calibrated against a sufficiently
  large, reliable set of human-rated examples.
- Semantic similarity does not prove legal meaning preservation.
- Rule warnings can include harmless wording changes and may miss other risks.
- English and Hindi performance must be evaluated separately with suitable data.
- This prototype is for research and demonstration, not legal advice.

## Project files
The Google Drive project folder contains the working notebook outputs,
baseline results, score results, summaries, and draft project report.
Copy only files suitable for public release into the GitHub repository.

## Future work
- Create a reliable set of same-language simplification pairs.
- Collect human ratings using a documented rubric.
- Evaluate agreement with human ratings and calibrate the score.
- Expand Hindi evaluation and legal-detail detection.
- Compare with stronger baselines.

## Status
Working prototype; further evaluation and validation are required.
