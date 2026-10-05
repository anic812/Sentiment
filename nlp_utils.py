"""Shared NLP preprocessing used by the notebook and the standalone prediction pipeline."""
import re
import nltk
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

STOP = set(stopwords.words("english"))
NEGATIONS = {"no", "nor", "not", "never"}
_stemmer, _lemmer = PorterStemmer(), WordNetLemmatizer()


def combine(summary, text):
    """Q20: Review_Summary + Review_Text -> Combined_Review."""
    summary = "" if summary is None or summary != summary else str(summary)
    return f"{summary}. {text}".strip(". ").strip()


def clean(text):
    """Q21: lowercase, drop punctuation/symbols, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Q22: NLTK word tokenisation."""
    return word_tokenize(text)


def remove_numeric(tokens):
    """Q26: drop purely numeric tokens."""
    return [t for t in tokens if not t.isdigit()]


def remove_stop(tokens, keep_negations=False):
    """Q23: stop-word removal (optionally protecting negations)."""
    stop = STOP - NEGATIONS if keep_negations else STOP
    return [t for t in tokens if t not in stop]


def stem(tokens):
    """Q24: Porter stemmer."""
    return [_stemmer.stem(t) for t in tokens]


def lemmatize(tokens):
    """Q25: WordNet lemmatiser (verb pass then noun pass)."""
    return [_lemmer.lemmatize(_lemmer.lemmatize(t, "v"), "n") for t in tokens]


VARIANTS = ("text_clean", "text_nostop", "text_stem", "text_lemma")


def build_variant(combined, variant="text_clean"):
    """Q28: build one of the cleaned-text representations from a Combined_Review string."""
    toks = remove_numeric(tokenize(clean(combined)))
    if variant == "text_clean":
        pass
    elif variant == "text_nostop":
        toks = remove_stop(toks)
    elif variant == "text_stem":
        toks = stem(remove_stop(toks, keep_negations=True))
    elif variant == "text_lemma":
        toks = lemmatize(remove_stop(toks, keep_negations=True))
    else:
        raise ValueError(variant)
    return " ".join(toks)
