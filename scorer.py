from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_candidate_match(resume_text: str, job_desc_text: str, resume_skills: list, required_skills: list) -> dict:
    """
    Computes contextual similarity and skill overlap score.
    Returns composite match score, matched skills, and missing skills.
    """
    # 1. TF-IDF Context Similarity
    if resume_text and job_desc_text:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
        context_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    else:
        context_score = 0.0

    # 2. Skill Overlap Calculation
    res_set = set(resume_skills)
    req_set = set(required_skills)
    
    matched_skills = sorted(list(res_set.intersection(req_set)))
    missing_skills = sorted(list(req_set - res_set))

    skill_score = len(matched_skills) / len(req_set) if req_set else 0.0

    # Composite weighted match (60% skill match + 40% TF-IDF context match)
    final_score = round((skill_score * 0.60 + context_score * 0.40) * 100, 2)

    return {
        "final_score": final_score,
        "context_score": round(context_score * 100, 2),
        "skill_score": round(skill_score * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }