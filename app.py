import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Import custom processing pipelines
from src.text_cleaner import extract_text_from_pdf, clean_text
from src.skill_extractor import SkillExtractor
from src.scorer import calculate_candidate_match

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening & Candidate Ranking System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Resume Screening & Candidate Ranking System")
st.markdown("Automated candidate ranking using NLP skill extraction and vector similarity scoring.")

# Sidebar Controls
st.sidebar.header("Configuration")
top_n = st.sidebar.slider("Top Candidates to Display", min_value=1, max_value=20, value=5)

# Initialize Skill Extractor Model
@st.cache_resource
def load_extractor():
    return SkillExtractor()

try:
    extractor = load_extractor()
except Exception as e:
    st.error(f"Failed to load NLP pipeline: {e}")
    st.stop()

# Main Workspace Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    job_description_input = st.text_area(
        "Paste Job Description Here",
        height=250,
        placeholder="e.g., Seeking a Data Scientist with strong Python, PyTorch, spaCy, and SQL skills..."
    )

with col2:
    st.subheader("2. Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF Resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

# Execution Pipeline
if st.button("🚀 Process and Rank Candidates", type="primary"):
    if not job_description_input.strip():
        st.warning("Please provide a Job Description before proceeding.")
    elif not uploaded_files:
        st.warning("Please upload at least one candidate resume PDF.")
    else:
        with st.spinner("Extracting text, analyzing skills, and calculating match scores..."):
            
            # Clean and process Job Description
            cleaned_jd = clean_text(job_description_input)
            jd_skills = extractor.extract_skills(cleaned_jd)
            
            candidate_results = []
            
            # Process each uploaded resume
            for file in uploaded_files:
                try:
                    # Extract raw text from PDF buffer
                    raw_text = extract_text_from_pdf(file)
                    cleaned_resume = clean_text(raw_text)
                    
                    # Extract skills
                    candidate_skills = extractor.extract_skills(cleaned_resume)
                    
                    # Calculate overall match score
                    score_metrics = calculate_candidate_match(
                        cleaned_resume, 
                        cleaned_jd, 
                        candidate_skills, 
                        jd_skills
                    )
                    
                    candidate_results.append({
                        "Candidate Name": file.name.replace(".pdf", "").replace("_", " ").title(),
                        "Match Score (%)": round(score_metrics.get("overall_score", 0) * 100, 2),
                        "Matched Skills": ", ".join(score_metrics.get("matched_skills", [])),
                        "Missing Skills": ", ".join(score_metrics.get("missing_skills", [])),
                        "Raw Text": raw_text
                    })
                except Exception as file_err:
                    st.error(f"Error processing {file.name}: {file_err}")

            if candidate_results:
                # Convert results to DataFrame
                df_results = pd.DataFrame(candidate_results)
                df_results = df_results.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
                
                # Store in session state for dynamic filtering
                st.session_state["results_df"] = df_results
                st.session_state["jd_skills"] = jd_skills

# Display Results & Visualizations
if "results_df" in st.session_state and not st.session_state["results_df"].empty:
    df = st.session_state["results_df"]
    
    st.markdown("---")
    st.subheader("📊 Screening Results & Candidate Rankings")
    
    # Overview Table
    st.dataframe(
        df[["Candidate Name", "Match Score (%)", "Matched Skills", "Missing Skills"]].head(top_n),
        use_container_width=True
    )
    
    # Visualization Section
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        fig_bar = px.bar(
            df.head(top_n),
            x="Match Score (%)",
            y="Candidate Name",
            orientation="h",
            title="Top Candidate Match Scores",
            color="Match Score (%)",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart2:
        # Distribution Breakdown
        fig_box = px.box(
            df, 
            y="Match Score (%)", 
            points="all",
            title="Candidate Match Score Distribution"
        )
        st.plotly_chart(fig_box, use_container_width=True)