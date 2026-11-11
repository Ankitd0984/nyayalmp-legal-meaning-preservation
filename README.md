# JUDGEBERT – Legal Meaning Preservation (LMP)

A prototype system for comparing original legal text with simplified legal text and identifying potential changes in legal meaning.

## 1. Project Overview

JUDGEBERT aims to support the evaluation of whether a simplified legal sentence preserves the meaning of its original version.

The application compares two text passages and provides:
- Semantic similarity using a multilingual MiniLM sentence-transformer.
- Rule-based warnings for potential changes in negation, legal modal terms, conditions, exceptions, and numbers.
- A preliminary LMP indicator.
- A final review status highlighting possible meaning changes.

**Important:** JUDGEBERT is an academic prototype. Its output is not legal advice, legal certification, or an expert-validated legal judgment.

## 2. Supported Languages

- English
- Hindi (experimental; limited evaluation data)

The current Streamlit application uses a multilingual sentence-transformer. Hindi performance has not been sufficiently validated.

## 3. Methodology

### A. Dataset Preparation
- Legal sentence-pair preparation and cleaning.
- Duplicate and incomplete record checks.
- Dataset-derived binary labels: preserved and changed.
- Training, validation, and test split creation.

### B. Baseline Experiments
- TF-IDF with Logistic Regression.
- TF-IDF cosine similarity.
- Multilingual MiniLM cosine similarity.
- MiniLM embeddings with Ridge regression as an experimental comparison.

### C. Current Application
The current application combines:

1. **Semantic similarity:** MiniLM generates sentence embeddings and calculates cosine similarity.
2. **Legal-risk checks:** Rule-based checks identify changes in negation, modal terms, conditions, and numbers.
3. **Hybrid decision logic:** Potentially important rule warnings trigger a review-required status, even when semantic similarity is high.

The Ridge regression experiment is not used for the displayed application indicator because its predictions were not reliably calibrated to the intended 1–10 range.

## 4. Experimental Results

The TF-IDF Logistic Regression baseline achieved approximately 94.37% test accuracy against the existing dataset-derived binary labels.

The comparison experiment reported:

| Method | Test accuracy | Preserved F1 |
|---|---:|---:|
| TF-IDF cosine | 0.7130 | 0.8321 |
| MiniLM cosine | 0.7465 | 0.8548 |
| MiniLM + Ridge | 0.9454 | 0.9642 |

These results measure agreement with existing dataset-derived labels, not human legal-expert judgments. They should not be interpreted as legal meaning-preservation accuracy.

## 5. Application Testing

The current Streamlit prototype was tested using three demonstration cases:

| Test case | Expected behaviour | Observed behaviour |
|---|---|---|
| Equivalent paraphrase | High indicator, no critical warning | Passed |
| Added negation reversing an obligation | Negation warning and review required | Passed |
| `May` changed to `must` | Modal warning and review required | Passed |

These are demonstration tests, not comprehensive validation.

## 6. Technology Stack

- Python 3.11
- Streamlit
- PyTorch
- Sentence-Transformers
- Multilingual MiniLM
- Scikit-learn
- Pandas
- NumPy
- Matplotlib and Seaborn

## 7. Project Structure

```text
Project NLP/
│
├── app/
│   ├── streamlit_app.py
│   └── audit_dataset.py
│
├── data/
├── models/
├── notebooks/
├── reports/
├── results/
├── src/
│
├── JUDGEBERT_LMP_submission/
│   ├── README.md
│   ├── JUDGEBERT_Project_Report_Draft.md
│   └── experiment result CSV files
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/Ankitd0984/judgebert-legal-lmp.git
cd judgebert-legal-lmp
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in the terminal.

The first launch may download the multilingual MiniLM model and therefore requires an internet connection.

## 9. Limitations

- Existing binary labels are dataset-derived weak labels, not human-verified legal judgments.
- The current 1–10 indicator is a heuristic based on semantic similarity and rule-based warnings. It is not a calibrated probability or validated legal score.
- Rule-based checks cover only selected legal-language patterns.
- Complex legal meaning, context, parties, scope, exceptions, and cross-references may not be detected.
- Hindi data is currently insufficient for robust performance claims.
- Human legal review remains necessary.

## 10. Future Improvements

- Build a human-annotated legal meaning-preservation dataset.
- Improve detection of rights, duties, exceptions, conditions, parties, and scope.
- Evaluate English and Hindi separately.
- Conduct broader testing with legal-domain experts.
- Improve score calibration and model validation.

## 11. Academic Project Disclaimer

JUDGEBERT is developed as a B.Tech Computer Science and Engineering academic project. It is intended for research and educational demonstration, not for making legal decisions or replacing qualified legal professionals.