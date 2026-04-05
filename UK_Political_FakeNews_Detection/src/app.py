# src/app.py
import streamlit as st
from pipeline import UKFakeNewsPipeline # Import your OOP class!

# --- PAGE CONFIG ---
st.set_page_config(page_title="UK Fake News Detector", page_icon="🇬🇧", layout="wide")

# --- INITIALIZE PIPELINE (Cached so it only loads once) ---
@st.cache_resource
def load_pipeline():
    return UKFakeNewsPipeline()

detector = load_pipeline()

# --- UI LAYOUT ---
st.title("🇬🇧 UK Political Fake News Detector")
st.write("Verify the credibility of online political claims using Dual-Branch Machine Learning.")

# User Input
user_text = st.text_area("Paste Article Text here:", height=200)
stack_choice = st.radio(
    "Choose Inference Stack",
    options=["roberta", "sklearn"],
    horizontal=True,
    help="RoBERTa uses transformer embeddings + fusion; sklearn uses TF-IDF + fusion.",
)

if st.button("Analyze Article"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing semantics and style..."):
            # Execute the OOP pipeline!
            results = detector.predict(user_text, stack=stack_choice)
            
            # Display Results
            st.markdown("---")
            st.caption(f"Stack: {results['stack']}")
            st.subheader(f"Verdict: {results['prediction']}")
            st.progress(results['confidence'])
            st.write(f"Confidence Score: {results['confidence'] * 100:.1f}%")

            branch_scores = results.get("branch_scores", {})
            st.markdown("### Branch Scores")
            st.write(
                {
                    "style_fake_probability": branch_scores.get("style_fake_probability"),
                    "semantic_fake_probability": branch_scores.get("semantic_fake_probability"),
                    "fusion_fake_probability": branch_scores.get("fusion_fake_probability"),
                }
            )
            
            st.markdown("### Branch B: Stylistic Breakdown")
            st.json(results['stylistic_breakdown'])