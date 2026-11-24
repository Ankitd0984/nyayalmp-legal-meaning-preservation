"""Train and export NyayaLMP's deployable TF-IDF + Ridge regression model.

The exported artifact is fitted on the complete QA-fixed 360-row annotation
corpus for application inference. Reported evaluation metrics must continue to
come from the grouped holdout / leave-one-Act-out experiments, not this fit.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

REQUIRED = {
    "Pair_ID", "Source_Provision_ID", "Act", "Language",
    "Original_Legal_Text", "Simplified_Text", "Human_LMP_Score_1_to_10"
}


def make_pair_text(original: str, simplified: str) -> str:
    return f"{str(original).strip()} [SEP] {str(simplified).strip()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="QA-fixed XLSX workbook")
    parser.add_argument("--output", default="models/nyayalmp_tfidf_ridge.joblib")
    args = parser.parse_args()

    df = pd.read_excel(args.data, sheet_name="Human_LMP_Annotation")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in ["Original_Legal_Text", "Simplified_Text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["Human_LMP_Score_1_to_10"] = pd.to_numeric(
        df["Human_LMP_Score_1_to_10"], errors="coerce"
    )
    df = df[
        (df["Original_Legal_Text"] != "")
        & (df["Simplified_Text"] != "")
        & df["Human_LMP_Score_1_to_10"].between(1, 10)
        & df["Source_Provision_ID"].notna()
        & df["Act"].notna()
    ].copy()
    if len(df) < 300:
        raise ValueError(f"Expected >=300 valid rows, found {len(df)}")
    if not {"english", "hindi"}.issubset(set(df["Language"].str.lower())):
        raise ValueError("Both English and Hindi rows are required")

    texts = [make_pair_text(a, b) for a, b in zip(df.Original_Legal_Text, df.Simplified_Text)]
    y = df["Human_LMP_Score_1_to_10"].to_numpy(dtype=float)

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=1,
        max_features=50000, sublinear_tf=True
    )
    X_word = word_vec.fit_transform(texts)
    X_char = char_vec.fit_transform(texts)
    X = hstack([X_word, X_char]).tocsr()
    regressor = Ridge(alpha=1.0)
    regressor.fit(X, y)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": "NyayaLMP TF-IDF word+char n-grams + Ridge",
        "model_version": "1.0.0",
        "task": "experimental human LMP score regression (1-10)",
        "word_vectorizer": word_vec,
        "char_vectorizer": char_vec,
        "regressor": regressor,
        "feature_definition": "Original_Legal_Text + ' [SEP] ' + Simplified_Text; word (1,2) and char (2,5) TF-IDF concatenated",
        "ridge_alpha": 1.0,
        "training_rows": int(len(df)),
        "unique_source_provisions": int(df.Source_Provision_ID.nunique()),
        "language_counts": {str(k): int(v) for k, v in df.Language.value_counts().items()},
        "dataset_file": Path(args.data).name,
        "fit_scope": "full QA-fixed corpus for deployment; not a held-out evaluation model",
        "target_column": "Human_LMP_Score_1_to_10",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
    }
    joblib.dump(artifact, output, compress=3)

    # Reload and run deterministic smoke inference on real examples.
    loaded = joblib.load(output)
    Xw_check = loaded["word_vectorizer"].transform(texts[:3])
    Xc_check = loaded["char_vectorizer"].transform(texts[:3])
    check_preds = loaded["regressor"].predict(hstack([Xw_check, Xc_check]).tocsr())
    if len(check_preds) != 3 or not np.isfinite(check_preds).all():
        raise RuntimeError("Artifact smoke test failed")

    report = {
        "artifact": str(output),
        "artifact_bytes": output.stat().st_size,
        "training_rows": len(df),
        "unique_source_provisions": int(df.Source_Provision_ID.nunique()),
        "language_counts": artifact["language_counts"],
        "smoke_test_predictions_first_3": [float(x) for x in check_preds],
        "note": "Predictions are raw Ridge regression outputs; app display may clip to 1-10. Not legal advice or expert-validated judgments.",
    }
    report_path = output.with_suffix(".metadata.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
