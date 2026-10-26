# JUDGEBERT: Legal Meaning Preservation (LMP)

**Project report draft**  
**Prepared:** 2026-10-26  
**Languages targeted:** English and Hindi

## 1. Problem statement

Legal documents often contain complex terminology, obligations, exceptions,
conditions, deadlines, and rights. Simplifying legal language can improve
readability, but careless rewriting may alter the original legal meaning.

JUDGEBERT aims to compare an original legal passage with a simplified version
in the same language and provide an initial indication of possible meaning
preservation issues.

## 2. Objectives

- Compare original and simplified text using multilingual sentence embeddings.
- Flag possible changes to negation, legal obligation or permission terms,
  numeric details, and references to legal parties.
- Produce a preliminary 1–10 score for demonstration.
- Support an English/Hindi workflow, subject to appropriate same-language
  evaluation data and further validation.

## 3. Current methodology

The prototype uses the
`paraphrase-multilingual-MiniLM-L12-v2` sentence-transformer model to compute
semantic similarity between the two texts.

A rule-based layer checks for differences in:
- Negation terms.
- Obligation or permission terms such as must, shall, and may.
- Numbers, dates, and currency expressions.
- Selected legal party terms.

The current score combines semantic similarity with manually chosen penalties
for detected differences. These penalties have not been empirically calibrated.
The score is therefore a preliminary heuristic, not a validated legal rating.

## 4. Evaluation sample

The current evaluation file contains **500 pairs**.

### Review-priority counts

- CHECK: 263
- HIGH - check legal details: 226
- LOWER - still verify: 11

### Rule warning counts

- Negation differences: 125
- Obligation/permission differences: 460
- Numbers, dates, or currency differences: 0
- Legal party term differences: 145

These are warning counts, not confirmed legal errors. A rule may flag harmless
rewriting, and a rule may miss important meaning changes.

## 5. Limitations

1. Semantic similarity does not guarantee legal meaning preservation.
2. The score mapping and warning penalties have not been validated against a
   sufficiently large, reliable set of human-rated examples.
3. The evaluation sample is not evidence of performance on all legal documents.
4. The English/Hindi capability requires dedicated evaluation in each language.
5. The current rules cover only selected lexical patterns and may miss changes
   involving exceptions, scope, actors, cross-references, or legal interpretation.
6. The prototype is not legal advice and should not be used to make legal
   decisions without qualified human review.

## 6. Future work

- Obtain a reliable set of same-language original/simplified legal pairs.
- Collect independent human ratings using a documented rubric.
- Evaluate agreement between model outputs and human ratings.
- Calibrate score thresholds and penalties using held-out data.
- Expand Hindi-specific testing and legal terminology coverage.
- Compare the baseline against stronger models and document the results.
- Develop a user interface with clear warnings and exportable results.

## 7. Conclusion

JUDGEBERT currently demonstrates a multilingual semantic similarity baseline,
rule-based legal-detail warnings, and a preliminary scoring workflow. Its
current value is as a research prototype for identifying text pairs that
may need closer review. Further labelled-data collection and evaluation are
required before making claims about legal meaning preservation accuracy.
