
# Performing tokenization and stemming with stopword removal

import pandas
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import *
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split

reviews = pandas.read_csv("Reviews_modified.csv")
df = pandas.DataFrame(reviews)

scores = df['Score'].tolist()
summary = df['Summary'].tolist()
text = df['Text'].tolist()

# Processing review summary
for i in range(len(summary)):
	# tokenizing
	tokens = word_tokenize(str(summary[i]).decode("utf8"))
	# stop word removal
	stop = stopwords.words('english')
	tokens_filtered = [x for x in tokens if x not in stop]
	# stemming
	porter = PorterStemmer()
	stems = []
	for k in tokens_filtered:
                stems.append(porter.stem(k))
        stemmedSent = ""
        for l in range(len(stems)):
                stemmedSent = stemmedSent + " " + stems[l].encode("utf8")
	summary[i] = stemmedSent

# processing review description (text)
for i in range(len(text)):
	# tokenizing
	tokens = word_tokenize(str(text[i]).decode("utf8"))
	# stop word removal
	stop = stopwords.words('english')
	tokens_filtered = [x for x in tokens if x not in stop]
	# stemming
	porter = PorterStemmer()
	stems = []
	for k in tokens_filtered:
                stems.append(porter.stem(k))
        stemmedSent = ""
        for l in range(len(stems)):
                stemmedSent = stemmedSent + " " + stems[l].encode("utf8")
	text[i] = stemmedSent

df = pandas.DataFrame({'Score':scores, 'Summary':summary, 'Text':text})

df.to_csv('with_stopwords.csv', index = False)






























