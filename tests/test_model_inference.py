from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inference import load_regression_artifact, predict_lmp_score


def test_saved_model_predicts_finite_score():
    artifact = load_regression_artifact(ROOT / "models" / "nyayalmp_tfidf_ridge.joblib")
    score = predict_lmp_score(
        "The tenant must pay rent by the fifth day of each month.",
        "The tenant is required to pay rent by day five each month.",
        artifact,
    )
    assert isinstance(score, float)
    assert score == score  # not NaN


def test_empty_input_rejected():
    artifact = load_regression_artifact(ROOT / "models" / "nyayalmp_tfidf_ridge.joblib")
    try:
        predict_lmp_score("", "Hindi or English text", artifact)
    except ValueError:
        return
    raise AssertionError("Expected empty input to raise ValueError")
