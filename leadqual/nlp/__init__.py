"""Natural-language processing for lead enquiries.

Modules:
    preprocess  - normalisation, sentence splitting, tokenisation, keywords
    extract     - rule/regex entity extraction into qualification fields
    similarity  - TF-IDF intent classification and nearest-neighbour search
"""

from .preprocess import (
    normalise_text,
    sentence_split,
    tokenize,
    content_tokens,
    top_keywords,
    extract_questions,
)
from .extract import extract_fields, analyse_lead
from .similarity import IntentClassifier, find_similar_leads

__all__ = [
    "normalise_text",
    "sentence_split",
    "tokenize",
    "content_tokens",
    "top_keywords",
    "extract_questions",
    "extract_fields",
    "analyse_lead",
    "IntentClassifier",
    "find_similar_leads",
]
