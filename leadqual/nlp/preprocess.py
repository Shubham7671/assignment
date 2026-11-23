"""Text preprocessing: normalisation, segmentation, tokenisation, keywords.

NLTK is used where it is available. Every NLTK call has a pure-Python
fallback so the application still runs on a machine where the corpora were
never downloaded - important for offline marking and CI.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache

from ..exceptions import NLPError

# --------------------------------------------------------------------------
# Optional NLTK wiring
# --------------------------------------------------------------------------
try:
    import nltk
    from nltk.stem import PorterStemmer

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    nltk = None  # type: ignore[assignment]
    PorterStemmer = None  # type: ignore[assignment]
    _NLTK_AVAILABLE = False


#: Used when the NLTK stopword corpus is unavailable.
_FALLBACK_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he her here hers herself him himself his how i i'd
i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she should shouldn't so some such than that the their theirs
them themselves then there these they this those through to too under until up
very was wasn't we were weren't what when where which while who whom why with
won't would wouldn't you your yours yourself yourselves
""".split())

#: Domain words that must survive stopword filtering even if short/common.
_DOMAIN_KEEP = frozenset({
    "mba", "msc", "bsc", "phd", "ba", "ma", "it", "ai", "hr", "ux", "qa",
    "pte", "ielts", "toefl", "gre", "gmat", "cat", "sat",
    "fee", "fees", "emi", "loan", "online", "campus", "evening", "weekend",
    "part", "full", "time", "visa", "intake", "january", "march", "september",
})

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'+\-/]*")
_QUESTION_STARTERS = (
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "can", "could", "do", "does", "did", "is", "are", "am", "will", "would",
    "should", "may", "might", "have", "has", "any chance", "is there",
)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def normalise_text(text: str, *, strip_contacts: bool = False) -> str:
    """Normalise raw message text for analysis.

    Unicode-normalises, collapses whitespace and optionally strips URLs and
    email addresses (which otherwise pollute the keyword list).

    Args:
        text: Raw text.
        strip_contacts: When True, remove URLs and email addresses.

    Returns:
        The cleaned text. Empty input yields an empty string.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        raise NLPError(f"Expected text, got {type(text).__name__}")

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    if strip_contacts:
        cleaned = _URL_RE.sub(" ", cleaned)
        cleaned = _EMAIL_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def sentence_split(text: str) -> list[str]:
    """Split text into sentences, preferring NLTK's Punkt tokenizer.

    Args:
        text: Raw or normalised text.

    Returns:
        A list of non-empty, stripped sentences.
    """
    cleaned = normalise_text(text)
    if not cleaned:
        return []

    if _NLTK_AVAILABLE:
        try:
            sentences = nltk.sent_tokenize(cleaned)
            return [s.strip() for s in sentences if s.strip()]
        except LookupError:
            pass  # punkt not downloaded - fall through
        except Exception:  # pragma: no cover - defensive
            pass

    parts = _SENTENCE_FALLBACK_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """Split text into word tokens.

    Punctuation is dropped; intra-word hyphens, apostrophes, ``+`` and ``/``
    are kept so tokens like ``part-time`` and ``c++`` survive.

    Args:
        text: Raw or normalised text.
        lowercase: Lower-case before tokenising.

    Returns:
        A list of word tokens.
    """
    cleaned = normalise_text(text)
    if not cleaned:
        return []
    if lowercase:
        cleaned = cleaned.lower()
    return _WORD_RE.findall(cleaned.lower() if lowercase else cleaned)


# --------------------------------------------------------------------------
# Stopwords / stemming
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Return the stopword set, from NLTK when available."""
    if _NLTK_AVAILABLE:
        try:
            from nltk.corpus import stopwords

            words = set(stopwords.words("english"))
            return frozenset(words - _DOMAIN_KEEP)
        except LookupError:
            pass
        except Exception:  # pragma: no cover
            pass
    return frozenset(_FALLBACK_STOPWORDS - _DOMAIN_KEEP)


@lru_cache(maxsize=1)
def _get_stemmer():
    """A Porter stemmer instance, or ``None`` when NLTK is absent."""
    if _NLTK_AVAILABLE and PorterStemmer is not None:
        try:
            return PorterStemmer()
        except Exception:  # pragma: no cover
            return None
    return None


def stem(token: str) -> str:
    """Stem a single token (identity when NLTK is unavailable)."""
    stemmer = _get_stemmer()
    if stemmer is None:
        return token
    try:
        return stemmer.stem(token)
    except Exception:  # pragma: no cover
        return token


def content_tokens(text: str, *, min_length: int = 2, do_stem: bool = False) -> list[str]:
    """Tokenise and drop stopwords, very short tokens and bare numbers.

    Args:
        text: Raw or normalised text.
        min_length: Shortest token to keep (domain terms are exempt).
        do_stem: Apply Porter stemming to the surviving tokens.

    Returns:
        The content-bearing tokens, in order.
    """
    stops = get_stopwords()
    tokens: list[str] = []
    for token in tokenize(text):
        if token in _DOMAIN_KEEP:
            tokens.append(stem(token) if do_stem else token)
            continue
        if token in stops or len(token) < min_length or token.isdigit():
            continue
        tokens.append(stem(token) if do_stem else token)
    return tokens


def top_keywords(text: str, limit: int = 12) -> list[str]:
    """Most frequent content words, most frequent first.

    Args:
        text: Raw or normalised text.
        limit: Maximum number of keywords to return.

    Returns:
        Up to ``limit`` keywords.
    """
    tokens = content_tokens(text, min_length=3)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(limit)]


# --------------------------------------------------------------------------
# Question detection
# --------------------------------------------------------------------------
def extract_questions(text: str) -> list[str]:
    """Pull out the questions an enquirer asked.

    Catches both explicit questions (ending in ``?``) and unpunctuated ones
    that open with an interrogative word - common in chat and SMS enquiries.

    Args:
        text: Raw or normalised text.

    Returns:
        Deduplicated question sentences, in order of appearance.
    """
    questions: list[str] = []
    seen: set[str] = set()

    for sentence in sentence_split(text):
        candidate = sentence.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        is_question = candidate.endswith("?") or lowered.startswith(_QUESTION_STARTERS)
        if not is_question:
            continue
        # Only keep interrogative-looking openers when reasonably short,
        # so statements like "Can you see I have attached my CV ..." that
        # run on for a paragraph are not misread as questions.
        if not candidate.endswith("?") and len(candidate.split()) > 20:
            continue
        key = re.sub(r"\W+", "", lowered)
        if key in seen:
            continue
        seen.add(key)
        questions.append(candidate)

    return questions


# --------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------
_POSITIVE_WORDS = frozenset("""
excited keen interested eager great excellent perfect wonderful happy thanks
thank appreciate looking forward ready motivated determined love brilliant
helpful good definitely absolutely yes please
""".split())

_NEGATIVE_WORDS = frozenset("""
worried concerned confused unsure expensive costly afraid struggle struggling
difficult hard problem issue disappointed frustrated unhappy delay delayed
cannot can't unable unfortunately hesitant doubt doubtful
""".split())


@lru_cache(maxsize=1)
def _get_vader():
    """VADER analyser when the lexicon is present, else ``None``."""
    if not _NLTK_AVAILABLE:
        return None
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except LookupError:
        return None
    except Exception:  # pragma: no cover
        return None


def sentiment(text: str) -> tuple[str, float]:
    """Classify the enquirer's tone.

    Uses NLTK VADER when its lexicon is installed, otherwise a small
    lexicon-count heuristic.

    Args:
        text: Raw or normalised text.

    Returns:
        ``(label, score)`` where label is ``positive``/``neutral``/``negative``
        and score is in ``[-1.0, 1.0]``.
    """
    cleaned = normalise_text(text)
    if not cleaned:
        return "neutral", 0.0

    analyser = _get_vader()
    if analyser is not None:
        try:
            score = float(analyser.polarity_scores(cleaned)["compound"])
        except Exception:  # pragma: no cover
            score = 0.0
        else:
            if score >= 0.25:
                return "positive", round(score, 3)
            if score <= -0.25:
                return "negative", round(score, 3)
            return "neutral", round(score, 3)

    tokens = tokenize(cleaned)
    if not tokens:
        return "neutral", 0.0
    positive = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    negative = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if positive == negative == 0:
        return "neutral", 0.0
    score = (positive - negative) / max(positive + negative, 1)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return label, round(float(score), 3)


def nlp_backend_status() -> dict[str, bool]:
    """Report which optional NLP resources resolved - shown in the UI footer."""
    return {
        "nltk_installed": _NLTK_AVAILABLE,
        "punkt_tokenizer": bool(_NLTK_AVAILABLE and sentence_split("A test. Another.")),
        "stopwords_corpus": len(get_stopwords()) > len(_FALLBACK_STOPWORDS) - len(_DOMAIN_KEEP),
        "vader_lexicon": _get_vader() is not None,
        "stemmer": _get_stemmer() is not None,
    }
