# IMDb Review Preprocessing and Bag-of-Words Report

## Dataset

- Source used: Official IMDb parquet splits
- Documents: 50,000
- Labels available: True

## (a)–(c) Preprocessing pipeline

For each review, the pipeline lowercases the text, decodes HTML entities,
removes HTML tags and URLs using regular expressions, replaces characters
other than English letters and whitespace with spaces, collapses whitespace,
and tokenizes with a regular expression. English stop words are removed and
remaining tokens are lemmatized with NLTK WordNetLemmatizer.

Unique token counts measured after stop-word removal but before lemmatization:
**99,140**. Unique token counts after lemmatization:
**89,690**. Lemmatization reduced the unique vocabulary by
**9,450** types. Multiple inflected forms that map to one lemma are
counted as one token type.

### Corpus count summary

| measure                                                      |   count |
|:-------------------------------------------------------------|--------:|
| documents                                                    |   50000 |
| unique raw word tokens before preprocessing                  |   99426 |
| unique tokens after cleaning/tokenization                    |   99293 |
| unique tokens after stop-word removal (before lemmatization) |   99140 |
| unique tokens after lemmatization                            |   89690 |
| unique CountVectorizer features after preprocessing          |   89690 |

## (d)–(e) Word-frequency feature dictionaries

`extract_word_features(review)` returns a dictionary of each normalized,
stop-word-filtered, lemmatized token and its frequency. This hand-written
function uses `Counter` and does not use CountVectorizer.

```json
{
  "This movie was absolutely amazing and entertaining": {
    "movie": 1,
    "absolutely": 1,
    "amazing": 1,
    "entertaining": 1
  },
  "The movie was boring and predictable": {
    "movie": 1,
    "boring": 1,
    "predictable": 1
  }
}
```

## (f) CountVectorizer Bag-of-Words matrix

CountVectorizer is used here only for the separate corpus-level BoW
representation, as requested:

- Number of documents: **50,000**
- Number of unique features: **89,690**
- Document-term matrix shape: **(50000, 89690)**

The matrix has one row per review and one column per preprocessed vocabulary
feature; entries are token counts.

## (g) Vocabulary changes

The raw vocabulary is collected from lowercase source text before HTML, URL,
punctuation, or stop-word cleanup. The final vocabulary consists of
lemmatized, stop-word-filtered terms. Terms absent as exact forms from the
final vocabulary are labeled as stop words, markup/URL/punctuation removals,
or inflectional forms merged by lemmatization. The CSV contains up to 500
examples, the reason, and the normalized form when lemmatization merged it.

| token           | reason                                             | normalized_form   |
|:----------------|:---------------------------------------------------|:------------------|
| a               | English stop word removed                          |                   |
| about           | English stop word removed                          |                   |
| answerman       | HTML tag or attribute removed by HTML-tag regex    |                   |
| apps            | HTML tag or attribute removed by HTML-tag regex    |                   |
| aetv            | URL token removed by URL regex                     |                   |
| afrovideo       | URL token removed by URL regex                     |                   |
| eeewww          | Removed during markup, URL, or punctuation cleanup |                   |
| slllloooowwwwww | Removed during markup, URL, or punctuation cleanup |                   |
| aardvarks       | Inflection merged by lemmatization                 | aardvark          |
| abandons        | Inflection merged by lemmatization                 | abandon           |

HTML tag and URL patterns are removed before tokenization, punctuation and
non-letter characters are replaced with spaces, and common function words
(such as articles, pronouns, and auxiliaries) are deleted using the English
stop list. Lemmatization also merges inflectional variants rather than
discarding their meaning. Thus the final vocabulary is smaller and more
consistent than the source-text vocabulary.
