import re
import pdfplumber

def extract_text_from_pdf(pdf_file) -> str:
    """Extract raw text from an uploaded PDF file handle."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def clean_text(text: str) -> str:
    """Clean and normalize raw text for NLP processing."""
    if not text:
        return ""
    text = re.sub(r'http\S+\s*', ' ', text)                  # Remove URLs
    text = re.sub(r'#\S+', '', text)                         # Remove hashtags
    text = re.sub(r'[^\w\s]', ' ', text)                     # Remove special punctuation
    text = re.sub(r'\s+', ' ', text).strip()                 # Remove extra whitespaces
    return text.lower()