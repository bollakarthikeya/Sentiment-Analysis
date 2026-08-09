
# -*- coding: utf8

# reading data

# remove numbering, convert to lower case and remove punctuation 

import numpy
import pandas
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import *
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split

reviews = pandas.read_csv("Reviews.csv")
df = pandas.DataFrame(reviews)

df.drop('Id', axis = 1, inplace=True)
df.drop('ProductId', axis = 1, inplace=True)
df.drop('UserId', axis = 1, inplace=True)
df.drop('ProfileName', axis = 1, inplace=True)
df.drop('HelpfulnessNumerator', axis = 1, inplace=True)
df.drop('HelpfulnessDenominator', axis = 1, inplace=True)
df.drop('Time', axis = 1, inplace=True)

df = df[df.Score != 3]
scores = df['Score'].tolist()
summary = df['Summary'].tolist()
text = df['Text'].tolist()

for i in range(len(scores)):
	if scores[i] == 1 or scores[i] == 2:
		scores[i] = "negative"
	elif scores[i] == 4 or scores[i] == 5:
		scores[i] = "positive"

df = pandas.DataFrame({'Score':scores, 'Summary':summary, 'Text':text})
# select negative reviews
df_neg = df.loc[df['Score'] == "negative"]
# shuffle them
df_neg = df_neg.iloc[numpy.random.permutation(len(df_neg))]
# select positive reviews
df_pos = df.loc[df['Score'] == "positive"]
# shuffle them
df_pos = df_pos.iloc[numpy.random.permutation(len(df_pos))]
# select equal no of positives as there are negatives
df_pos = df_pos.head(len(df_neg))
# create final dataframe with both positive and negative reviews
df = pandas.concat([df_pos, df_neg])
# shuffle them
df = df.iloc[numpy.random.permutation(len(df))]

punct = set(string.punctuation)
num = ["1","2","3","4","5","6","7","8","9","0"]

scores = df['Score'].tolist()
summary = df['Summary'].tolist()
text = df['Text'].tolist()

# Processing review summary
for i in range(len(summary)):
	# changing to lower case
	temp = str(summary[i])
	summary[i] = temp.lower()
	# removing punctuation
	summary[i] = "".join(x for x in summary[i] if x not in punct)	
	# removing numbers
	summary[i] = "".join(x for x in summary[i] if x not in num)	

# processing review description (text)
for i in range(len(text)):
	# changing to lower case
	temp = str(text[i])
	text[i] = temp.lower()
	# removing punctuation
	text[i] = "".join(x for x in text[i] if x not in punct)	
	# removing numbers
	text[i] = "".join(x for x in text[i] if x not in num)	

df = pandas.DataFrame({'Score':scores, 'Summary':summary, 'Text':text})

df.to_csv('Reviews_modified.csv', index = False)



