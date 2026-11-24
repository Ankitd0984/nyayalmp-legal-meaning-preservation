# NyayaLMP model card

## Model identity
- **Project:** NyayaLMP — An Explainable English–Hindi Legal Meaning Preservation System for Indian Property Law.
- **Artifact:** `models/nyayalmp_tfidf_ridge.joblib`.
- **Model:** TF-IDF word (1–2 grams) and character (2–5 grams) features with Ridge regression.
- **Intended use:** Educational/research prototype inference with mandatory human review.

## Training scope
The deployable artifact is fitted on the full QA-fixed bilingual annotation corpus (360 labelled pairs from 180 source provisions: 180 English and 180 Hindi). This artifact is for application inference; its fit on the training corpus is not held-out test performance.

## Inputs and output
The model receives the original legal passage and candidate simplified passage concatenated with a separator. It returns an experimental regression estimate on the annotated 1–10 scale. The app displays the raw prediction and a clipped 1–10 display value.

## Important limitations
- Annotation was performed by one annotator in the current dataset.
- Hindi statutory source verification is incomplete.
- The model is not calibrated as a probability of legal equivalence.
- It does not replace legal professionals or source-text verification.
- Reported research metrics must come from grouped-provision holdout and leave-one-Act-out experiments, not from this full-corpus artifact.

## Reproducibility
See `src/models/train_export_tfidf.py`, `requirements.txt`, and the research notebook in the repository.
