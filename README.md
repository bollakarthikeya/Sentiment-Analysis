# Sentiment Analysis on Amazon Fine Food Reviews

## Overview

This project performs binary sentiment classification on the **Amazon Fine Food Reviews** dataset (https://snap.stanford.edu/data/web-FineFoods.html), comparing whether a review's **summary** or full **text** is more effective for identifying positive and negative sentiment.

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
- Remove the stopwords provided in standard **127-word NLTK** stopword list
- Create a customized **120-word** list and remove the stopwords 

The customized list retains seven sentiment-bearing words that were removed from the standard stopword list:

`but`, `above`, `below`, `no`, `not`, `too`, `very`

## Results

- **Review Text** outperformed **Review Summaries** for sentiment classification
- **Linear SVM achieved the highest accuracy: 93.134%** on Review Text without removing any stopwords
- Multinomial Naive Bayes produced the lowest accuracy among the three models
- Retaining sentiment-bearing words such as `not`, `no`, `but`, `too`, and `very` produced results close to retaining all stopwords

## Implementation

- **Python 2.7**
- **NLTK 3.1** — tokenization, stemming, stopword and special-character processing
- **scikit-learn** — classifiers and model evaluation
- **pandas** — data handling
- **NumPy / SciPy** — mathematical and data computation
