# IMDb Review Preprocessing and Bag-of-Words

This project builds a full text-preprocessing pipeline for the IMDb 50K movie
review dataset. It covers lowercasing, regex cleaning, tokenization, stop-word
removal, lemmatization, vocabulary comparisons, a hand-written word-frequency
feature extractor, and a CountVectorizer document-term matrix.

## Run

Install dependencies and execute:

```powershell
py -m pip install -r requirements.txt
py assignment.py
```

To use the Kaggle CSV (`IMDB Dataset.csv`, columns `review` and `sentiment`),
download it from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
and pass its path:

```powershell
py assignment.py --dataset-csv "C:\path\to\IMDB Dataset.csv"
```

Without a Kaggle download, the script fetches the official 25,000-review train
and test parquet files from the Stanford IMDb dataset mirror and combines them.
This is the same 50,000-review IMDb corpus and split used in the previous
assignments. Dataset files are stored in `data/` and excluded from Git.

## Outputs

The script writes to `results/`:

- `assignment_report.md`: preprocessing summary, example feature dictionaries,
  vocabulary change discussion, and BoW shape
- `preprocessing_summary.csv`: corpus counts
- `vocabulary_changes.csv`: examples of tokens that disappear and their reason
- `example_feature_dictionaries.json`: output for the two supplied reviews
- `bow_shape.csv`: document count, feature count, and matrix dimensions

## Notes

- HTML tags and URLs are removed with regular expressions; HTML entities are
  decoded before tokenization.
- Tokens are lowercased, alphabetic, filtered against the NLTK English stop
  list, and lemmatized using WordNet's default noun lemmatizer.
- Unique-word counts before and after lemmatization are measured after cleaning,
  tokenization, and stop-word removal so the comparison isolates lemmatization.
- `extract_word_features(review)` uses a Python frequency dictionary and does
  not use CountVectorizer. CountVectorizer is used separately only for the
  requested corpus-level Bag-of-Words matrix.

Dataset citation: Maas, A. L. et al. (2011), *Learning Word Vectors for
Sentiment Analysis*, ACL. Kaggle dataset page:
[IMDb 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
