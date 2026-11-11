
import re
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="JUDGEBERT | Legal Meaning Preservation",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ JUDGEBERT")
st.subheader("Legal Meaning Preservation Assistant")

st.write(
    "Compare an original legal passage with its simplified version "
    "to identify possible changes in meaning."
)

st.info(
    "Prototype notice: This tool provides preliminary indicators, "
    "not legal advice or expert-validated legal judgments."
)


# --------------------------------------------------
# LOAD SEMANTIC MODEL
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_encoder():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# LEGAL RISK CHECKER
# --------------------------------------------------

NEGATION_TERMS = {
    "not", "no", "never", "neither", "nor",
    "without", "unless", "cannot"
}

MODAL_TERMS = {
    "must", "shall", "may", "might", "can",
    "could", "should", "will", "would", "ought"
}

CONDITION_TERMS = {
    "if", "unless", "provided", "subject",
    "except", "only", "until"
}


def normalize_legal_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"\b(is|are|was|were) required to\b",
        "must",
        text
    )

    text = re.sub(r"\b(has|have) to\b", "must", text)
    text = re.sub(r"\bon or before\b", "by", text)

    return text


def extract_terms(text, vocabulary):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    return sorted(set(words) & vocabulary)


def extract_numbers(text):
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text.lower())


def legal_risk_check(original, simplified):

    original = normalize_legal_text(original)
    simplified = normalize_legal_text(simplified)

    warnings = []

    if extract_terms(original, NEGATION_TERMS) != extract_terms(
        simplified, NEGATION_TERMS
    ):
        warnings.append(
            "NEGATION CHANGE: A right, duty, or prohibition may have changed."
        )

    if extract_terms(original, MODAL_TERMS) != extract_terms(
        simplified, MODAL_TERMS
    ):
        warnings.append(
            "MODAL CHANGE: Permission, obligation, or possibility may have changed."
        )

    if extract_terms(original, CONDITION_TERMS) != extract_terms(
        simplified, CONDITION_TERMS
    ):
        warnings.append(
            "CONDITION CHANGE: A condition or exception may have changed."
        )

    if extract_numbers(original) != extract_numbers(simplified):
        warnings.append(
            "NUMBER CHANGE: A date, amount, deadline, or quantity may have changed."
        )

    if not warnings:
        warnings.append(
            "No change detected by basic rules. Preservation is not guaranteed."
        )

    return warnings


# --------------------------------------------------
# JUDGEBERT ANALYSIS ENGINE
# --------------------------------------------------

def judgebert_analyze(original, simplified, encoder):

    embeddings = encoder.encode(
        [original, simplified],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarity = float(np.dot(embeddings[0], embeddings[1]))

    # Prototype indicator, not a calibrated probability or legal score.
    base_score = 1 + 9 * np.clip(similarity, 0, 1)

    warnings = legal_risk_check(original, simplified)

    critical_keywords = [
        "NEGATION CHANGE",
        "MODAL CHANGE",
        "NUMBER CHANGE",
        "CONDITION CHANGE"
    ]

    critical_warnings = [
        warning for warning in warnings
        if any(keyword in warning for keyword in critical_keywords)
    ]

    if critical_warnings:
        displayed_score = min(base_score, 4.0)
        status = "REVIEW REQUIRED - POSSIBLE LEGAL MEANING CHANGE"

    elif base_score >= 8:
        displayed_score = base_score
        status = "LIKELY PRESERVED - HUMAN REVIEW REQUIRED"

    else:
        displayed_score = base_score
        status = "UNCERTAIN - HUMAN REVIEW REQUIRED"

    return {
        "similarity": similarity,
        "score": float(displayed_score),
        "warnings": warnings,
        "status": status
    }


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Original Legal Text")
    original_text = st.text_area(
        "Enter original text",
        height=250,
        placeholder="Paste the original legal paragraph here..."
    )

with col2:
    st.markdown("### Simplified Legal Text")
    simplified_text = st.text_area(
        "Enter simplified text",
        height=250,
        placeholder="Paste the simplified paragraph here..."
    )

analyze = st.button(
    "Analyze Legal Meaning",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if analyze:

    if not original_text.strip() or not simplified_text.strip():
        st.error("Please enter both original and simplified text.")

    else:
        try:
            with st.spinner("Analyzing legal text..."):

                encoder = load_encoder()

                result = judgebert_analyze(
                    original_text,
                    simplified_text,
                    encoder
                )

            st.divider()
            st.header("Analysis Results")

            score_col, similarity_col = st.columns(2)

            with score_col:
                st.metric(
                    "LMP Prototype Indicator",
                    f"{result['score']:.2f}/10"
                )

            with similarity_col:
                st.metric(
                    "Semantic Similarity",
                    f"{result['similarity']:.4f}"
                )

            st.markdown("### Final Status")

            if "REVIEW REQUIRED" in result["status"]:
                st.error(result["status"])

            elif "UNCERTAIN" in result["status"]:
                st.warning(result["status"])

            else:
                st.success(result["status"])

            st.markdown("### Legal Risk Warnings")

            for warning in result["warnings"]:
                if any(
                    keyword in warning
                    for keyword in [
                        "NEGATION CHANGE",
                        "MODAL CHANGE",
                        "NUMBER CHANGE",
                        "CONDITION CHANGE"
                    ]
                ):
                    st.warning(warning)
                else:
                    st.info(warning)

            st.caption(
                "The indicator is based on semantic similarity and "
                "prototype risk rules. It is not an expert-validated "
                "legal meaning-preservation score."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")


st.divider()

st.caption(
    "JUDGEBERT | B.Tech CSE Project | Legal Meaning Preservation Prototype"
)
