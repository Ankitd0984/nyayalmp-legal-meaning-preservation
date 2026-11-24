# NyayaLMP
### An Explainable English–Hindi Legal Meaning Preservation System for Indian Property Law

**NyayaLMP** is a B.Tech CSE research prototype that compares an original legal passage with a simplified version and surfaces signals that may indicate a change in legal meaning. The name combines *Nyaya* (न्याय, justice) with *LMP* (Legal Meaning Preservation).

The project focuses on English and Hindi text related to Indian property and tenancy law. It is designed to support human review—not to provide legal advice, certify equivalence, or make legal decisions.

---

## What the application does

Paste an original legal passage and its simplified candidate. NyayaLMP displays three distinct signals:

1. **TF-IDF + Ridge estimate** — a supervised regression estimate trained on the annotated bilingual pair corpus. It is shown separately and is not a calibrated probability.
2. **MiniLM semantic similarity** — cosine similarity between multilingual sentence embeddings. Similarity is not the same as legal equivalence.
3. **Rule-based legal-risk checks** — heuristic checks for selected English/Hindi negation, modality/obligation, conditions/exceptions, and numeric expressions.

The interface keeps these signals separate to reduce the risk of interpreting one score as a legal verdict. A triggered rule is a prompt for review; it may be a false positive, and the rules may miss complex changes.

## Research data and model scope

The annotation corpus used for the deployable regression artifact contains **360 labelled pairs from 180 source provisions**, with **180 English and 180 Hindi records**. The current artifact is fitted on the full QA-fixed corpus for application inference.

**Important evaluation distinction:** the full-corpus artifact is not a held-out evaluation model. Do not present its training fit as test performance. Use the grouped-provision holdout and leave-one-Act-out results from the research notebook/report when discussing evaluation.

The dataset currently has important limitations:
- The current corpus was labelled by one annotator.
- Source verification is incomplete, particularly for Hindi statutory text.
- It covers selected Indian property/tenancy provisions, not Indian law generally.
- The model and heuristic checks are experimental and require expert review.

## Interface identity

NyayaLMP's visual identity uses a deep legal-navy foundation, teal accents, and a restrained saffron highlight inspired by Indian civic/legal visual language. The **न्याय** mark refers to justice; it is a project identity, not an official government seal or endorsement.

## Repository structure

```text
NyayaLMP/
├── app/
│   ├── streamlit_app.py
│   └── audit_dataset.py
├── models/
│   ├── nyayalmp_tfidf_ridge.joblib
│   └── nyayalmp_tfidf_ridge.metadata.json
├── src/
│   └── models/
│       ├── inference.py
│       └── train_export_tfidf.py
├── tests/
│   └── test_model_inference.py
├── results/
│   └── experiment result files
├── NyayaLMP_Project_Resources/
├── 01_exploration_ipynb.ipynb
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## Run locally

### 1. Requirements
- Python 3.11 recommended
- Git
- Internet access on first launch to download the multilingual MiniLM encoder, unless it is already cached

### 2. Clone

```bash
git clone https://github.com/Ankitd0984/nyayalmp-legal-meaning-preservation.git
cd nyayalmp-legal-meaning-preservation
```

If you choose a different GitHub repository name, update the clone URL accordingly.

### 3. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Launch the app

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

### 6. Run tests

```bash
python -m pytest -q
```

## Rebuild the regression artifact (optional)

The trained artifact is included in `models/`. To rebuild it from the QA-fixed annotation workbook:

```bash
python src/models/train_export_tfidf.py --data path/to/your_QA_fixed_annotation_workbook.xlsx
```

The script writes `models/nyayalmp_tfidf_ridge.joblib` and its metadata file. The workbook is not bundled in this repository because dataset redistribution/source permissions should be checked separately.

## Suggested GitHub repository settings

- **Repository name:** `nyayalmp-legal-meaning-preservation`
- **Description:** `Explainable English–Hindi legal meaning preservation prototype for Indian property law.`
- **Visibility:** Public only if the included data and legal text excerpts are permitted to be redistributed; otherwise keep the repository private or remove restricted data files.
- Add a repository screenshot to `assets/` after capturing the final app screen.

## Disclaimer

NyayaLMP is an academic research prototype developed for educational demonstration. It is not legal advice, does not certify that a simplification preserves legal effect, and must not be used as a substitute for checking the official legal source or consulting a qualified legal professional.
