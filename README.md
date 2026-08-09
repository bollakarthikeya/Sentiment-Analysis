# Sentiment Analysis on Amazon Fine Food Reviews

## Overview

This project performs binary sentiment classification on the **Amazon Fine Food Reviews** dataset, comparing whether a review's **summary** or full **text** is more effective for identifying positive and negative sentiment.

**Dataset**
- Original reviews: **568,454**
- Users: **256,059**
- Products: **74,258**
- Period: **October 1999–October 2012**
- Positive: scores **4–5**
- Negative: scores **1–2**
- Neutral/mixed: score **3** excluded

## Pipeline

1. Remove nonessential metadata (`ProductId`, `UserId`, `ProfileName`, helpfulness fields, `Time`).
2. Convert ratings into positive/negative labels.
3. Clean text:
   - Remove numbers and punctuation
   - Convert to lowercase
   - Tokenize
   - Stem words
   - Experiment with stopword removal
4. Balance the dataset to **164,074 reviews**:
   - 82,037 positive
   - 82,037 negative
5. Build a bag-of-words vocabulary and generate **length-normalized TF-IDF** feature vectors.
6. Train and evaluate classifiers using **5-fold cross-validation**.

## Models

- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (linear kernel)

## Stopword Experiments

Three preprocessing variants were evaluated:

- Keep all stopwords
- Remove the standard **127-word NLTK** stopword list
- Remove a customized **120-word** list

The customized list retains seven sentiment-bearing words that were removed from the standard stopword list:

`but`, `above`, `below`, `no`, `not`, `too`, `very`

## Results

| Model | Summary: No Removal | Summary: NLTK | Summary: Custom | Text: No Removal | Text: NLTK | Text: Custom |
|---|---:|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 87.101% | 84.899% | 87.123% | 88.780% | 88.333% | 88.490% |
| Logistic Regression | 89.604% | 85.752% | 89.422% | 90.942% | 90.432% | 90.767% |
| Support Vector Machine | **90.372%** | 86.558% | 90.073% | **93.134%** | 92.622% | 93.003% |

## Key Findings

- **Full review text outperformed review summaries** for sentiment classification.
- **Linear SVM achieved the highest accuracy: 93.134%** on full review text without stopword removal
- Multinomial Naive Bayes produced the lowest accuracy among the three models.
- Removing the complete NLTK stopword list generally reduced accuracy.
- Retaining sentiment-bearing words such as `not`, `no`, `but`, `too`, and `very` produced results close to keeping all stopwords

## Implementation

The original project used:

- **Python 2.7**
- **NLTK 3.1** — tokenization, stemming, stopword and special-character processing
- **scikit-learn** — classifiers and model evaluation
- **pandas** — data handling
- **NumPy / SciPy** — numerical support
