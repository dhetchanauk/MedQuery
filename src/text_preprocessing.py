
import re

def clean_text(text):
    """
    We deliberately avoid removing stopwords or stemming medical text because that could change the meaning of medical statements.
    Basic NLP preprocessing.
    Preserves medical terminology and sentence meaning.
    """

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted special characters,
    # while preserving common medical symbols.
    text = re.sub(r'[^\w\s.,;:()%/+−-]', '', text)

    return text.strip()
