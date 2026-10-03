from __future__ import annotations

import argparse
import html
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SPLIT_URLS = {
    "train": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/train-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/test-00000-of-00001.parquet",
}
HTML_TAG_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
WORD_RE = re.compile(r"[a-z]+")


def ensure_nltk_resources() -> None:
    for package, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Could not download the required NLTK resource: {package}")


def download_official_splits() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, url in SPLIT_URLS.items():
        path = DATA_DIR / f"imdb_{split}.parquet"
        if not path.is_file():
            print(f"Downloading official IMDb {split} split...")
            urllib.request.urlretrieve(url, path)


def load_reviews(dataset_csv: Path | None) -> tuple[list[str], list[str], str]:
    if dataset_csv is not None:
        if not dataset_csv.is_file():
            raise FileNotFoundError(f"Dataset CSV does not exist: {dataset_csv}")
        frame = pd.read_csv(dataset_csv)
        if "review" not in frame.columns:
            raise ValueError("Kaggle CSV must contain a 'review' column.")
        sentiments = frame["sentiment"].astype(str).tolist() if "sentiment" in frame else [
            "unknown"
        ] * len(frame)
        return frame["review"].fillna("").astype(str).tolist(), sentiments, "Kaggle CSV"

    download_official_splits()
    frames = [
        pd.read_parquet(DATA_DIR / f"imdb_{split}.parquet", columns=["text", "label"])
        for split in ("train", "test")
    ]
    frame = pd.concat(frames, ignore_index=True)
    labels = frame["label"].map({0: "negative", 1: "positive"}).astype(str).tolist()
    return frame["text"].fillna("").astype(str).tolist(), labels, "Official IMDb parquet splits"


def clean_review(review: str) -> str:
    text = html.unescape(review.lower())
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(review: str) -> list[str]:
    return WORD_RE.findall(review)


def preprocess_tokens(
    review: str, stop_words: set[str], lemmatizer: WordNetLemmatizer
) -> tuple[list[str], list[str]]:
    cleaned_tokens = tokenize(clean_review(review))
    content_tokens = [token for token in cleaned_tokens if token not in stop_words]
    lemmas = [lemmatizer.lemmatize(token) for token in content_tokens]
    return content_tokens, lemmas


def extract_word_features(review: str) -> dict[str, int]:
    """Return normalized, stop-word-filtered lemmatized token frequencies."""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    _, lemmas = preprocess_tokens(review, stop_words, lemmatizer)
    return dict(Counter(lemmas))


def determine_removed_vocab(
    raw_vocab: set[str],
    cleaned_vocab: set[str],
    clean_vocab: set[str],
    stop_words: set[str],
    raw_reviews: Iterable[str],
) -> list[dict[str, str]]:
    tags: set[str] = set()
    urls: set[str] = set()
    for review in raw_reviews:
        tags.update(WORD_RE.findall(" ".join(HTML_TAG_RE.findall(review.lower()))))
        urls.update(WORD_RE.findall(" ".join(URL_RE.findall(review.lower()))))

    removed = sorted(raw_vocab - clean_vocab)
    rows: list[dict[str, str]] = []
    lemmatizer = WordNetLemmatizer()
    for token in removed:
        lemma = lemmatizer.lemmatize(token)
        if lemma != token and lemma in clean_vocab:
            reason = "Inflection merged by lemmatization"
            normalized_form = lemma
        elif token in stop_words:
            reason = "English stop word removed"
            normalized_form = ""
        elif token not in cleaned_vocab and token in tags:
            reason = "HTML tag or attribute removed by HTML-tag regex"
            normalized_form = ""
        elif token not in cleaned_vocab and token in urls:
            reason = "URL token removed by URL regex"
            normalized_form = ""
        elif token not in cleaned_vocab:
            reason = "Removed during markup, URL, or punctuation cleanup"
            normalized_form = ""
        elif token in tags:
            reason = "HTML tag or attribute token"
            normalized_form = ""
        elif token in urls:
            reason = "URL token"
            normalized_form = ""
        else:
            reason = "Vocabulary variant merged or filtered during normalization"
            normalized_form = ""
        rows.append({"token": token, "reason": reason, "normalized_form": normalized_form})
    return rows


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess IMDb reviews and build a BoW matrix.")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        help="Path to Kaggle IMDB Dataset.csv with a review column (and optional sentiment).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()
    reviews, labels, data_source = load_reviews(args.dataset_csv)
    if not reviews:
        raise ValueError("The selected dataset contains no reviews.")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    raw_vocab: set[str] = set()
    cleaned_vocab: set[str] = set()
    no_stop_vocab: set[str] = set()
    lemmatized_vocab: set[str] = set()
    processed_reviews: list[str] = []

    print(f"Preprocessing {len(reviews):,} reviews...")
    for review in reviews:
        raw_vocab.update(WORD_RE.findall(html.unescape(review.lower())))
        cleaned_tokens = tokenize(clean_review(review))
        cleaned_vocab.update(cleaned_tokens)
        content_tokens = [token for token in cleaned_tokens if token not in stop_words]
        no_stop_vocab.update(content_tokens)
        lemmas = [lemmatizer.lemmatize(token) for token in content_tokens]
        lemmatized_vocab.update(lemmas)
        processed_reviews.append(" ".join(lemmas))

    vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z]+\b",
        dtype="int32",
    )
    document_term_matrix = vectorizer.fit_transform(processed_reviews)
    bow_shape = document_term_matrix.shape
    features = vectorizer.get_feature_names_out()
    removed_rows = determine_removed_vocab(
        raw_vocab, cleaned_vocab, lemmatized_vocab, stop_words, reviews
    )

    examples = {
        "This movie was absolutely amazing and entertaining": extract_word_features(
            "This movie was absolutely amazing and entertaining"
        ),
        "The movie was boring and predictable": extract_word_features(
            "The movie was boring and predictable"
        ),
    }
    summary = pd.DataFrame(
        [
            {"measure": "documents", "count": len(reviews)},
            {"measure": "unique raw word tokens before preprocessing", "count": len(raw_vocab)},
            {"measure": "unique tokens after cleaning/tokenization", "count": len(cleaned_vocab)},
            {"measure": "unique tokens after stop-word removal (before lemmatization)", "count": len(no_stop_vocab)},
            {"measure": "unique tokens after lemmatization", "count": len(lemmatized_vocab)},
            {"measure": "unique CountVectorizer features after preprocessing", "count": len(features)},
        ]
    )
    summary.to_csv(RESULTS_DIR / "preprocessing_summary.csv", index=False)
    pd.DataFrame(removed_rows[:500], columns=["token", "reason", "normalized_form"]).to_csv(
        RESULTS_DIR / "vocabulary_changes.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "documents": bow_shape[0],
                "unique_features": bow_shape[1],
                "matrix_rows": bow_shape[0],
                "matrix_columns": bow_shape[1],
                "matrix_shape": str(bow_shape),
            }
        ]
    ).to_csv(RESULTS_DIR / "bow_shape.csv", index=False)
    (RESULTS_DIR / "example_feature_dictionaries.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )

    before_lemmatization = len(no_stop_vocab)
    after_lemmatization = len(lemmatized_vocab)
    vocab_delta = before_lemmatization - after_lemmatization
    removed_frame = pd.DataFrame(removed_rows)
    preferred_reasons = [
        "English stop word removed",
        "HTML tag or attribute removed by HTML-tag regex",
        "URL token removed by URL regex",
        "Removed during markup, URL, or punctuation cleanup",
        "Inflection merged by lemmatization",
    ]
    example_parts = []
    for reason in preferred_reasons:
        matches = removed_frame[removed_frame["reason"] == reason].head(2)
        if not matches.empty:
            example_parts.append(matches)
    removed_examples = (
        pd.concat(example_parts, ignore_index=True)
        if example_parts
        else removed_frame.head(10)
    )
    example_json = json.dumps(examples, indent=2, ensure_ascii=False)
    report = f"""# IMDb Review Preprocessing and Bag-of-Words Report

## Dataset

- Source used: {data_source}
- Documents: {len(reviews):,}
- Labels available: {len(set(labels)) > 1}

## (a)–(c) Preprocessing pipeline

For each review, the pipeline lowercases the text, decodes HTML entities,
removes HTML tags and URLs using regular expressions, replaces characters
other than English letters and whitespace with spaces, collapses whitespace,
and tokenizes with a regular expression. English stop words are removed and
remaining tokens are lemmatized with NLTK WordNetLemmatizer.

Unique token counts measured after stop-word removal but before lemmatization:
**{before_lemmatization:,}**. Unique token counts after lemmatization:
**{after_lemmatization:,}**. Lemmatization reduced the unique vocabulary by
**{vocab_delta:,}** types. Multiple inflected forms that map to one lemma are
counted as one token type.

### Corpus count summary

{markdown_table(summary)}

## (d)–(e) Word-frequency feature dictionaries

`extract_word_features(review)` returns a dictionary of each normalized,
stop-word-filtered, lemmatized token and its frequency. This hand-written
function uses `Counter` and does not use CountVectorizer.

```json
{example_json}
```

## (f) CountVectorizer Bag-of-Words matrix

CountVectorizer is used here only for the separate corpus-level BoW
representation, as requested:

- Number of documents: **{bow_shape[0]:,}**
- Number of unique features: **{bow_shape[1]:,}**
- Document-term matrix shape: **{bow_shape}**

The matrix has one row per review and one column per preprocessed vocabulary
feature; entries are token counts.

## (g) Vocabulary changes

The raw vocabulary is collected from lowercase source text before HTML, URL,
punctuation, or stop-word cleanup. The final vocabulary consists of
lemmatized, stop-word-filtered terms. Terms absent as exact forms from the
final vocabulary are labeled as stop words, markup/URL/punctuation removals,
or inflectional forms merged by lemmatization. The CSV contains up to 500
examples, the reason, and the normalized form when lemmatization merged it.

{markdown_table(removed_examples)}

HTML tag and URL patterns are removed before tokenization, punctuation and
non-letter characters are replaced with spaces, and common function words
(such as articles, pronouns, and auxiliaries) are deleted using the English
stop list. Lemmatization also merges inflectional variants rather than
discarding their meaning. Thus the final vocabulary is smaller and more
consistent than the source-text vocabulary.
"""
    (RESULTS_DIR / "assignment_report.md").write_text(report, encoding="utf-8")
    print(f"BoW matrix shape: {bow_shape}; unique features: {len(features):,}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
