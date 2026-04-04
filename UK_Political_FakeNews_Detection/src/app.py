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

if st.button("Analyze Article"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing semantics and style..."):
            # Execute the OOP pipeline!
            results = detector.predict(user_text)
            
            # Display Results
            st.markdown("---")
            st.subheader(f"Verdict: {results['prediction']}")
            st.progress(results['confidence'])
            st.write(f"Confidence Score: {results['confidence'] * 100:.1f}%")
            
            st.markdown("### Branch B: Stylistic Breakdown")
            st.json(results['stylistic_breakdown'])