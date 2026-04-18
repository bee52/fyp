# src/app.py
import streamlit as st
from pydantic import ValidationError
from pipeline import UKFakeNewsPipeline # Import your OOP class!


def _as_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _risk_label(value: float | None) -> tuple[str, str]:
    if value is None:
        return "Unavailable", "gray"
    if value < 0.33:
        return "Low", "green"
    if value < 0.66:
        return "Medium", "orange"
    return "High", "red"


def _render_diagnostics_sidebar(results: dict) -> None:
    branch_scores = results.get("branch_scores", {})
    style = results.get("stylistic_breakdown", {})

    st.sidebar.header("Diagnostic Transparency")
    st.sidebar.caption("Branch-level probabilities and stylistic evidence")
    st.sidebar.caption(f"Stack: {results.get('stack', 'unknown')}")

    st.sidebar.subheader("Branch Probabilities")
    branch_map = [
        ("Style Branch", branch_scores.get("style_fake_probability")),
        ("Semantic Branch", branch_scores.get("semantic_fake_probability")),
        ("Fusion Output", branch_scores.get("fusion_fake_probability")),
    ]

    for label, score in branch_map:
        st.sidebar.markdown(f"**{label}: {_as_percent(score)}**")
        st.sidebar.progress(float(score) if score is not None else 0.0)
        risk_text, risk_color = _risk_label(score)
        st.sidebar.caption(f"Risk: :{risk_color}[{risk_text}]")

    st.sidebar.caption("Risk thresholds: Low < 33%, Medium 33-65.9%, High >= 66%")

    st.sidebar.subheader("Stylistic Diagnostics")
    style_metrics = {
        "Word Count": float(style.get("word_count", 0.0)),
        "Shout Ratio": float(style.get("shout_ratio", 0.0)),
        "Exclamation Density": float(style.get("exclamation_density", 0.0)),
        "Question Density": float(style.get("question_density", 0.0)),
        "Lexical Diversity": float(style.get("lexical_diversity", 0.0)),
        "Sentiment": float(style.get("sentiment", 0.0)),
    }

    st.sidebar.metric("Word Count", f"{style_metrics['Word Count']:.0f}")
    st.sidebar.metric("Shout Ratio", _as_percent(style_metrics["Shout Ratio"]))
    st.sidebar.metric("Exclamation Density", f"{style_metrics['Exclamation Density']:.3f}")
    st.sidebar.metric("Question Density", f"{style_metrics['Question Density']:.3f}")
    st.sidebar.metric("Lexical Diversity", f"{style_metrics['Lexical Diversity']:.3f}")
    st.sidebar.metric("Sentiment", f"{style_metrics['Sentiment']:+.3f}")

    st.sidebar.markdown("#### Explore One Feature")
    selected_feature = st.sidebar.selectbox(
        "Choose feature",
        options=list(style_metrics.keys()),
    )

    selected_value = style_metrics[selected_feature]
    if selected_feature in {"Shout Ratio", "Lexical Diversity"}:
        st.sidebar.write(f"{selected_feature}: {_as_percent(selected_value)}")
        st.sidebar.progress(max(0.0, min(1.0, selected_value)))
    elif selected_feature in {"Exclamation Density", "Question Density"}:
        st.sidebar.write(f"{selected_feature}: {selected_value:.3f} per word")
    elif selected_feature == "Sentiment":
        normalized = (selected_value + 1.0) / 2.0
        st.sidebar.write(f"{selected_feature}: {selected_value:+.3f} (-1 to +1)")
        st.sidebar.progress(max(0.0, min(1.0, normalized)))
    else:
        st.sidebar.write(f"{selected_feature}: {selected_value:.0f}")

    with st.sidebar.expander("Raw Diagnostic Payload"):
        st.json(
            {
                "branch_scores": branch_scores,
                "stylistic_breakdown": style,
            }
        )

# --- PAGE CONFIG ---
st.set_page_config(page_title="UK Fake News Detector", page_icon="🇬🇧", layout="wide")

# --- INITIALIZE PIPELINE (Cached so it only loads once) ---
@st.cache_resource
def load_pipeline():
    return UKFakeNewsPipeline()

detector = load_pipeline()

if "latest_results" not in st.session_state:
    st.session_state.latest_results = None

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
            try:
                # Execute the OOP pipeline!
                results = detector.predict(user_text, stack=stack_choice)
                st.session_state.latest_results = results
            except (ValidationError, ValueError, FileNotFoundError, ImportError) as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()
            
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

if st.session_state.latest_results:
    _render_diagnostics_sidebar(st.session_state.latest_results)
else:
    st.sidebar.header("Diagnostic Transparency")
    st.sidebar.info("Run an analysis to view branch probabilities and stylistic diagnostics.")