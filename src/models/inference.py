"""Inference helpers for the exported NyayaLMP TF-IDF + Ridge artifact."""
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "nyayalmp_tfidf_ridge.joblib"


def load_regression_artifact(model_path=DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained NyayaLMP model not found at {path}. "
            "Run src/models/train_export_tfidf.py to create it."
        )
    return joblib.load(path)


def predict_lmp_score(original: str, simplified: str, artifact: dict) -> float:
    original = str(original).strip()
    simplified = str(simplified).strip()
    if not original or not simplified:
        raise ValueError("Both original and simplified text are required")
    pair = [f"{original} [SEP] {simplified}"]
    x_word = artifact["word_vectorizer"].transform(pair)
    x_char = artifact["char_vectorizer"].transform(pair)
    features = hstack([x_word, x_char]).tocsr()
    prediction = float(artifact["regressor"].predict(features)[0])
    if not np.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction")
    return prediction
