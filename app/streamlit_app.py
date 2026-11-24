
import re
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.models.inference import load_regression_artifact, predict_lmp_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NyayaLMP | Legal Meaning Preservation",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17243b; --muted:#68758b; --teal:#0f8b82; --teal-dark:#08736d; --saffron:#d58a32; --line:#e4eaf1; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5f8fc 0%,#ffffff 42%,#f7fafc 100%); color:var(--ink); }
.block-container { max-width:1220px; padding-top:1.7rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
.hero { position:relative; overflow:hidden; background:radial-gradient(circle at 92% 8%,rgba(213,138,50,.25),transparent 24%),linear-gradient(120deg,#101f38 0%,#183b59 55%,#08766f 100%); padding:2.15rem 2.35rem; border:1px solid rgba(255,255,255,.12); border-radius:24px; color:white; box-shadow:0 18px 45px rgba(20,45,70,.16); margin-bottom:1.2rem; }\n.hero:after { content:'न्याय'; position:absolute; right:2.4rem; bottom:-2.4rem; font-family:serif; font-size:8rem; font-weight:700; color:rgba(255,255,255,.055); pointer-events:none; }\n.brand-mark { display:inline-flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:13px; margin-right:.65rem; background:linear-gradient(145deg,#e4a44e,#c77c2b); color:#14253d; font-family:serif; font-size:1.45rem; font-weight:800; vertical-align:middle; box-shadow:0 5px 16px rgba(0,0,0,.18); }
.hero-kicker { color:#a9eee5; font-size:.78rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
.hero-title { font-family:'Manrope',sans-serif; font-size:2.45rem; line-height:1.12; font-weight:800; color:white; margin:.45rem 0 .55rem; }
.hero-copy { color:#d6e4ee; font-size:1rem; max-width:760px; line-height:1.65; }
.pill { display:inline-block; border:1px solid rgba(255,255,255,.25); background:rgba(255,255,255,.10); color:#eaf7f6; padding:.35rem .7rem; border-radius:99px; font-size:.78rem; margin:.8rem .35rem 0 0; }
.section-label { color:#0c8f86; text-transform:uppercase; font-size:.74rem; font-weight:800; letter-spacing:.13em; margin-bottom:.35rem; }
.helper { color:#718096; font-size:.9rem; margin-top:-.35rem; }
div[data-testid="stTextArea"] textarea { border:1px solid #dce5ee; border-radius:14px; background:#fff; color:#17243b; font-size:.96rem; line-height:1.65; padding:1rem; box-shadow:0 3px 12px rgba(20,45,70,.035); }
div[data-testid="stTextArea"] textarea:focus { border-color:#0f9d91; box-shadow:0 0 0 3px rgba(15,157,145,.12); }
div.stButton > button[kind="primary"] { background:linear-gradient(90deg,#0b9389,#087f79); border:0; border-radius:12px; color:white; font-weight:700; min-height:3.1rem; box-shadow:0 8px 18px rgba(8,127,121,.18); }
div.stButton > button[kind="primary"]:hover { background:linear-gradient(90deg,#087f79,#066c68); border:0; transform:translateY(-1px); }
.metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; min-height:125px; box-shadow:0 5px 18px rgba(28,48,75,.045); }
.metric-name { color:#68758b; font-size:.8rem; font-weight:700; margin-bottom:.45rem; }
.metric-value { color:#14253d; font-family:'Manrope',sans-serif; font-size:1.9rem; font-weight:800; letter-spacing:-.04em; }
.metric-note { color:#8793a4; font-size:.76rem; margin-top:.3rem; }
.result-panel { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.3rem; box-shadow:0 5px 18px rgba(28,48,75,.04); }
.status-good { border-left:5px solid #0e9b76; background:#ecfdf5; padding:1rem 1.1rem; border-radius:12px; color:#12664f; font-weight:700; }
.status-review { border-left:5px solid #e2a12d; background:#fff8e8; padding:1rem 1.1rem; border-radius:12px; color:#80520c; font-weight:700; }
.status-risk { border-left:5px solid #dc6262; background:#fff0f0; padding:1rem 1.1rem; border-radius:12px; color:#9b3434; font-weight:700; }
.warning-item { background:#fff9e9; border:1px solid #f3e4bd; color:#76520f; padding:.8rem 1rem; border-radius:11px; margin:.45rem 0; font-size:.91rem; }
.info-item { background:#eff8ff; border:1px solid #d7eafa; color:#315c7c; padding:.8rem 1rem; border-radius:11px; margin:.45rem 0; font-size:.91rem; }
div[data-testid="stAlert"] { border-radius:12px; }
footer { visibility:hidden; }
</style>
<div class="hero">
  <div class="hero-kicker">न्याय · Explainable legal-language intelligence</div>
  <div class="hero-title"><span class="brand-mark">न्य</span>NyayaLMP</div>
  <div class="hero-copy">An English–Hindi Legal Meaning Preservation workspace for Indian property law. Compare a source provision with its simplified version and inspect separate model signals and legal-risk checks before human review.</div>
  <span class="pill">⚖️ Indian property law</span><span class="pill">English · हिन्दी</span><span class="pill">Meaning-preservation review</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="helper">A research prototype inspired by the idea of nyaya (न्याय, justice) — for preliminary review only, not legal advice or automated legal decisions.</div>', unsafe_allow_html=True)


# --------------------------------------------------
# LOAD SEMANTIC MODEL
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource
def load_encoder():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_lmp_regressor():
    return load_regression_artifact()


# --------------------------------------------------
# LEGAL RISK CHECKER
# --------------------------------------------------

# English and Hindi legal-risk vocabulary. These are heuristic checks,
# not a complete linguistic or legal analysis.
EN_NEGATION_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b", r"\bnor\b",
    r"\bwithout\b", r"\bunless\b", r"\bcannot\b", r"\bcan't\b"
]
HI_NEGATION_PATTERNS = [
    r"नहीं", r"नही", r"मत", r"बिना", r"ना\s+ही", r"न\s+ही"
]

EN_MODAL_PATTERNS = [
    r"\bmust\b", r"\bshall\b", r"\bmay\b", r"\bmight\b", r"\bcan\b",
    r"\bcould\b", r"\bshould\b", r"\bwill\b", r"\bwould\b", r"\bought\b"
]
HI_MODAL_PATTERNS = [
    r"चाहिए", r"अनिवार्य", r"आवश्यक", r"बाध्य", r"करना\s+होगा",
    r"करनी\s+होगी", r"करने\s+होंगे", r"सकता", r"सकती", r"सकते",
    r"अनुमति", r"अधिकार\s+है"
]

EN_CONDITION_PATTERNS = [
    r"\bif\b", r"\bunless\b", r"\bprovided\b", r"\bsubject\b",
    r"\bexcept\b", r"\bonly\b", r"\buntil\b"
]
HI_CONDITION_PATTERNS = [
    r"यदि", r"अगर", r"जब\s+तक", r"बशर्ते", r"केवल", r"सिवाय",
    r"के\s+अलावा", r"अपवाद", r"शर्त"
]

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_legal_text(text):
    text = str(text).lower().translate(DEVANAGARI_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize common English paraphrase variants.
    text = re.sub(
        r"\b(is|are|was|were) required to\b",
        "must",
        text
    )
    text = re.sub(r"\b(has|have) to\b", "must", text)
    text = re.sub(r"\bon or before\b", "by", text)
    return text


def _matched_patterns(text, patterns):
    """Return pattern IDs that occur in text; IDs are stable for comparison."""
    return {
        index for index, pattern in enumerate(patterns)
        if re.search(pattern, text, flags=re.IGNORECASE)
    }


def extract_numbers(text):
    normalized = normalize_legal_text(text)
    return re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", normalized)


def legal_risk_check(original, simplified):
    original = normalize_legal_text(original)
    simplified = normalize_legal_text(simplified)

    warnings = []

    negation_patterns = EN_NEGATION_PATTERNS + HI_NEGATION_PATTERNS
    modal_patterns = EN_MODAL_PATTERNS + HI_MODAL_PATTERNS
    condition_patterns = EN_CONDITION_PATTERNS + HI_CONDITION_PATTERNS

    if _matched_patterns(original, negation_patterns) != _matched_patterns(
        simplified, negation_patterns
    ):
        warnings.append(
            "NEGATION CHANGE: A right, duty, or prohibition may have changed "
            "(English/Hindi heuristic check)."
        )

    if _matched_patterns(original, modal_patterns) != _matched_patterns(
        simplified, modal_patterns
    ):
        warnings.append(
            "MODAL CHANGE: Permission, obligation, or possibility may have changed "
            "(English/Hindi heuristic check)."
        )

    if _matched_patterns(original, condition_patterns) != _matched_patterns(
        simplified, condition_patterns
    ):
        warnings.append(
            "CONDITION CHANGE: A condition or exception may have changed "
            "(English/Hindi heuristic check)."
        )

    if extract_numbers(original) != extract_numbers(simplified):
        warnings.append(
            "NUMBER CHANGE: A date, amount, deadline, or quantity may have changed."
        )

    if not warnings:
        warnings.append(
            "No change detected by basic English/Hindi rules. "
            "Preservation is not guaranteed."
        )

    return warnings


# --------------------------------------------------
# NYAYALMP ANALYSIS ENGINE
# --------------------------------------------------

def nyayalmp_analyze(original, simplified, encoder):

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

st.markdown('<div class="section-label">01 · Compare legal passages</div>', unsafe_allow_html=True)
st.markdown('## Add the original and simplified text')
st.markdown('<div class="helper">Paste both versions below. The analysis keeps model scores and rule-based warnings separate so that each signal can be interpreted independently.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Original legal text**")
    original_text = st.text_area(
        "Original legal text", height=245,
        placeholder="Paste the original legal paragraph here…",
        label_visibility="collapsed", key="original_legal_text"
    )
    st.caption("Source passage · English or Hindi")
with col2:
    st.markdown("**Simplified legal text**")
    simplified_text = st.text_area(
        "Simplified legal text", height=245,
        placeholder="Paste the simplified paragraph here…",
        label_visibility="collapsed", key="simplified_legal_text"
    )
    st.caption("Candidate simplification · English or Hindi")

button_col, hint_col = st.columns([1, 2])
with button_col:
    analyze = st.button("✦  Analyze legal meaning", type="primary", use_container_width=True)
with hint_col:
    st.markdown('<div class="helper" style="padding:.65rem 0">Tip: preserve names, dates, amounts, obligations and exceptions when simplifying.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if analyze:
    if not original_text.strip() or not simplified_text.strip():
        st.error("Please enter both the original and simplified text before analysis.")
    else:
        try:
            with st.spinner("Comparing passages and loading the analysis models…"):
                encoder = load_encoder()
                regression_artifact = load_lmp_regressor()
                result = nyayalmp_analyze(original_text, simplified_text, encoder)
                regression_score_raw = predict_lmp_score(
                    original_text, simplified_text, regression_artifact
                )
                regression_score_display = float(np.clip(regression_score_raw, 1.0, 10.0))

            st.markdown('<div class="section-label">02 · Analysis summary</div>', unsafe_allow_html=True)
            st.markdown("## Review the signals")
            m1, m2, m3 = st.columns(3, gap="medium")
            with m1:
                st.markdown(f"""<div class="metric-card"><div class="metric-name">Rule-adjusted MiniLM indicator</div><div class="metric-value">{result['score']:.2f}<span style="font-size:1rem;color:#8491a3"> / 10</span></div><div class="metric-note">Prototype indicator; may be reduced when a risk rule triggers</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card"><div class="metric-name">MiniLM semantic similarity</div><div class="metric-value">{result['similarity']:.4f}</div><div class="metric-note">Embedding similarity, not a legal-equivalence probability</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card"><div class="metric-name">Trained TF-IDF + Ridge estimate</div><div class="metric-value">{regression_score_display:.2f}<span style="font-size:1rem;color:#8491a3"> / 10</span></div><div class="metric-note">Raw model output: {regression_score_raw:.3f}</div></div>""", unsafe_allow_html=True)

            st.markdown("### Overall review status")
            if "REVIEW REQUIRED" in result["status"]:
                st.markdown(f'<div class="status-risk">⚠️ &nbsp; {result["status"]}</div>', unsafe_allow_html=True)
            elif "UNCERTAIN" in result["status"]:
                st.markdown(f'<div class="status-review">◷ &nbsp; {result["status"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-good">✓ &nbsp; {result["status"]}</div>', unsafe_allow_html=True)

            st.markdown("### Legal risk checks")
            critical = [w for w in result["warnings"] if any(k in w for k in [
                "NEGATION CHANGE", "MODAL CHANGE", "NUMBER CHANGE", "CONDITION CHANGE"
            ])]
            if critical:
                for warning in critical:
                    st.markdown(f'<div class="warning-item">⚑ &nbsp; {warning}</div>', unsafe_allow_html=True)
            else:
                for warning in result["warnings"]:
                    st.markdown(f'<div class="info-item">ⓘ &nbsp; {warning}</div>', unsafe_allow_html=True)

            with st.expander("How to interpret these results", expanded=False):
                st.markdown("""
                - **MiniLM semantic similarity** compares the sentence embeddings of the two passages.
                - **Rule-adjusted MiniLM indicator** is a prototype display indicator and may be capped when a rule warning is triggered.
                - **TF-IDF + Ridge estimate** is a separate experimental regression prediction from the trained model.
                - **Legal risk checks** are heuristic checks for negation, modal language, conditions and numbers. They can produce false positives or miss complex changes.

                None of these values is a calibrated probability or an expert-validated legal judgment. Always have a qualified person review the source and simplified text.
                """)
            st.caption(
                "Research prototype · The TF-IDF + Ridge artifact is fitted on the full annotated corpus for deployment. "
                "Research performance must be reported using the separate grouped holdout and leave-one-Act-out evaluations."
            )
        except Exception as error:
            st.error(f"Analysis failed: {error}")

st.divider()

st.caption(
    "NyayaLMP  ·  English–Hindi Legal Meaning Preservation  ·  B.Tech CSE Research Prototype"
)
