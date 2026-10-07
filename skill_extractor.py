import spacy
from spacy.matcher import PhraseMatcher

# Default comprehensive Tech & ML skill taxonomy
DEFAULT_SKILLS = [
    "python", "pytorch", "tensorflow", "fastapi", "scikit-learn", "spacy", "nltk",
    "pandas", "numpy", "sql", "power bi", "ansys", "autodesk fusion 360", "cad",
    "aws", "docker", "git", "rest api", "machine learning", "deep learning",
    "nlp", "sarimax", "time series", "tableau", "c++", "java", "react", "html", "css"
]

class SkillExtractor:
    def __init__(self, skill_list=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        skills = skill_list if skill_list else DEFAULT_SKILLS
        patterns = [self.nlp.make_doc(s) for s in skills]
        self.matcher.add("SKILL_PATTERNS", patterns)

    def extract_skills(self, text: str) -> list[str]:
        """Extract matching skills from preprocessed text."""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        extracted = set()
        for _, start, end in matches:
            extracted.add(doc[start:end].text.lower())
        return sorted(list(extracted))