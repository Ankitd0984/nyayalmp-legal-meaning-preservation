
import streamlit as st

st.set_page_config(
    page_title="JUDGEBERT Legal LMP",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ JUDGEBERT")
st.subheader("Legal Meaning Preservation Checker")

st.write(
    "Compare an original legal sentence with a simplified "
    "version to investigate whether legal meaning is preserved."
)

language = st.selectbox("Language", ["English", "Hindi"])

original = st.text_area(
    "Original legal sentence",
    placeholder="Enter the original sentence...",
)

simplified = st.text_area(
    "Simplified legal sentence",
    placeholder="Enter the simplified sentence...",
)

if st.button("Analyze Legal Meaning", type="primary"):
    if not original.strip() or not simplified.strip():
        st.warning("Enter both sentences first.")
    else:
        st.success("Input received successfully.")
        st.write("Language:", language)
        st.write("Original:", original)
        st.write("Simplified:", simplified)
        st.info(
            "Model integration is pending. No LMP score "
            "has been predicted yet."
        )

