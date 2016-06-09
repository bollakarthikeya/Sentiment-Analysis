
# https://www.youtube.com/watch?v=0tVN19WtPvA
# https://www.youtube.com/watch?v=V3RJGWaYqxQ&nohtml5=False
# https://www.youtube.com/watch?v=h4cG8jLGmKg&nohtml5=False

# performing classification using review_summary

import pandas
import string
import nltk
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import *
from sklearn.cross_validation import KFold
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import precision_recall_fscore_support
import numpy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, classification_report
from prettytable import PrettyTable
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Multinomial Naive Bayes
nb = MultinomialNB()

# Logistic Regression
lreg = LogisticRegression()

# SVM
svc = LinearSVC()

# perform TF.IDF conversion
tfidf = TfidfVectorizer()

t0 = PrettyTable(['','Multinomial Naive Bayes','Logistic Regression','Support Vector Machines'])
t1 = PrettyTable(['','Multinomial Naive Bayes','Logistic Regression','Support Vector Machines'])
t2 = PrettyTable(['','Multinomial Naive Bayes','Logistic Regression','Support Vector Machines'])
t3 = PrettyTable(['','Multinomial Naive Bayes','Logistic Regression','Support Vector Machines'])

# ######################################################################################################################
# 			 			WITHOUT STOPWORD REMOVAL
# ######################################################################################################################
reviews = pandas.read_csv("without_stopwords.csv")
df = pandas.DataFrame(reviews)
df.dropna(inplace = True)

# fetch 'score' and 'summary' of the review
X = df[['Summary']]
y = df[['Score']]

print "============================================================================="
print "\t\tWithout stopword removal"
print "\t\t~~~~~~~~~~~~~~~~~~~~~~~~"
print ""

mnb_accuracy = []
lreg_accuracy = []
svm_accuracy = []

mnb_precision = []
lreg_precision = []
svm_precision = []

mnb_recall = []
lreg_recall = []
svm_recall = []

mnb_fscore = []
lreg_fscore = []
svm_fscore = []

kf = KFold(len(X), n_folds = 5)

for train_index, test_index in kf:

	X_train, X_test = X[train_index[0] : train_index[-1]], X[test_index[0] : test_index[-1]]
	y_train, y_test = y[train_index[0] : train_index[-1]], y[test_index[0] : test_index[-1]]

	X_train = numpy.ravel(X_train)
	X_test = numpy.ravel(X_test)
	y_train = numpy.ravel(y_train)
	y_test = numpy.ravel(y_test)
	
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)	# dont put tfidf.fit_transform() over here, as dimensions will change and there will be mismatch	

	nb.fit(X_train_tfidf, y_train)
	y_pred = nb.predict(X_test_tfidf)
	print "Multinomial Naive Bayes"
	print "~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	mnb_accuracy.append(accuracy)		# write accuracy
	mnb_precision.append(float(temp[17]))	# write precision
	mnb_recall.append(float(temp[18]))	# write recall
	mnb_fscore.append(float(temp[19]))	# write the f1-score which is at 19th index of variable temp. write the value for 5 folds and take avg

	lreg.fit(X_train_tfidf, y_train)
	y_pred = lreg.predict(X_test_tfidf)
	print "Logistic Regression"
	print "~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	lreg_accuracy.append(accuracy)
	lreg_precision.append(float(temp[17]))
	lreg_recall.append(float(temp[18]))
	lreg_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	svc.fit(X_train_tfidf, y_train)
	y_pred = svc.predict(X_test_tfidf)
	print "Support Vector Machines (Linear kernel)"
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	svm_accuracy.append(accuracy)
	svm_precision.append(float(temp[17]))
	svm_recall.append(float(temp[18]))
	svm_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	print "\n"

# accuracy scores
mnb_avg = numpy.average(mnb_accuracy)
lreg_avg = numpy.average(lreg_accuracy)
svm_avg = numpy.average(svm_accuracy)

t0.add_row(['not removing stopwords', mnb_avg, lreg_avg, svm_avg])

# precision scores
mnb_avg = numpy.average(mnb_precision)
lreg_avg = numpy.average(lreg_precision)
svm_avg = numpy.average(svm_precision)

t1.add_row(['not removing stopwords', mnb_avg, lreg_avg, svm_avg])

# recall scores
mnb_avg = numpy.average(mnb_recall)
lreg_avg = numpy.average(lreg_recall)
svm_avg = numpy.average(svm_recall)

t2.add_row(['not removing stopwords', mnb_avg, lreg_avg, svm_avg])

# f1 measure scores
mnb_avg = numpy.average(mnb_fscore)
lreg_avg = numpy.average(lreg_fscore)
svm_avg = numpy.average(svm_fscore)

t3.add_row(['not removing stopwords', mnb_avg, lreg_avg, svm_avg])

print "============================================================================="

# ######################################################################################################################
# 			 			WITH STOPWORD REMOVAL
# ######################################################################################################################
reviews = pandas.read_csv("with_stopwords.csv")
df = pandas.DataFrame(reviews)
df.dropna(inplace = True)

# fetch 'score' and 'summary' of the review
X = df[['Summary']]
y = df[['Score']]

print "============================================================================="
print "\t\tWith stopword removal"
print "\t\t~~~~~~~~~~~~~~~~~~~~~~~~"
print ""

mnb_accuracy = []
lreg_accuracy = []
svm_accuracy = []

mnb_precision = []
lreg_precision = []
svm_precision = []

mnb_recall = []
lreg_recall = []
svm_recall = []

mnb_fscore = []
lreg_fscore = []
svm_fscore = []

kf = KFold(len(X), n_folds = 5)

for train_index, test_index in kf:

	X_train, X_test = X[train_index[0] : train_index[-1]], X[test_index[0] : test_index[-1]]
	y_train, y_test = y[train_index[0] : train_index[-1]], y[test_index[0] : test_index[-1]]

	X_train = numpy.ravel(X_train)
	X_test = numpy.ravel(X_test)
	y_train = numpy.ravel(y_train)
	y_test = numpy.ravel(y_test)
	
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)	# dont put tfidf.fit_transform() over here, as dimensions will change and there will be mismatch	

	nb.fit(X_train_tfidf, y_train)
	y_pred = nb.predict(X_test_tfidf)
	print "Multinomial Naive Bayes"
	print "~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	mnb_accuracy.append(accuracy)
	mnb_precision.append(float(temp[17]))
	mnb_recall.append(float(temp[18]))
	mnb_fscore.append(float(temp[19]))	# write the f1-score which is at 19th index of variable temp. write the value for 5 folds and take avg

	lreg.fit(X_train_tfidf, y_train)
	y_pred = lreg.predict(X_test_tfidf)
	print "Logistic Regression"
	print "~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	lreg_accuracy.append(accuracy)
	lreg_precision.append(float(temp[17]))
	lreg_recall.append(float(temp[18]))
	lreg_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	svc.fit(X_train_tfidf, y_train)
	y_pred = svc.predict(X_test_tfidf)
	print "Support Vector Machines (Linear kernel)"
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	svm_accuracy.append(accuracy)
	svm_precision.append(float(temp[17]))
	svm_recall.append(float(temp[18]))
	svm_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	print "\n"

# accuracy scores
mnb_avg = numpy.average(mnb_accuracy)
lreg_avg = numpy.average(lreg_accuracy)
svm_avg = numpy.average(svm_accuracy)

t0.add_row(['removing stopwords', mnb_avg, lreg_avg, svm_avg])

# precision scores
mnb_avg = numpy.average(mnb_precision)
lreg_avg = numpy.average(lreg_precision)
svm_avg = numpy.average(svm_precision)

t1.add_row(['removing stopwords', mnb_avg, lreg_avg, svm_avg])

# recall scores
mnb_avg = numpy.average(mnb_recall)
lreg_avg = numpy.average(lreg_recall)
svm_avg = numpy.average(svm_recall)

t2.add_row(['removing stopwords', mnb_avg, lreg_avg, svm_avg])

# f1 measure scores
mnb_avg = numpy.average(mnb_fscore)
lreg_avg = numpy.average(lreg_fscore)
svm_avg = numpy.average(svm_fscore)

t3.add_row(['removing stopwords', mnb_avg, lreg_avg, svm_avg])

print "============================================================================="

# ######################################################################################################################
# 			 			CUSTOMIZED STOPWORD REMOVAL
# ######################################################################################################################
reviews = pandas.read_csv("custom_stopwords.csv")
df = pandas.DataFrame(reviews)
df.dropna(inplace = True)

# fetch 'score' and 'summary' of the review
X = df[['Summary']]
y = df[['Score']]

print "============================================================================="
print "\t\tCustomized stopword removal"
print "\t\t~~~~~~~~~~~~~~~~~~~~~~~~"
print ""

mnb_accuracy = []
lreg_accuracy = []
svm_accuracy = []

mnb_precision = []
lreg_precision = []
svm_precision = []

mnb_recall = []
lreg_recall = []
svm_recall = []

mnb_fscore = []
lreg_fscore = []
svm_fscore = []

kf = KFold(len(X), n_folds = 5)

for train_index, test_index in kf:

	X_train, X_test = X[train_index[0] : train_index[-1]], X[test_index[0] : test_index[-1]]
	y_train, y_test = y[train_index[0] : train_index[-1]], y[test_index[0] : test_index[-1]]

	X_train = numpy.ravel(X_train)
	X_test = numpy.ravel(X_test)
	y_train = numpy.ravel(y_train)
	y_test = numpy.ravel(y_test)
	
	X_train_tfidf = tfidf.fit_transform(X_train)
	X_test_tfidf = tfidf.transform(X_test)	# dont put tfidf.fit_transform() over here, as dimensions will change and there will be mismatch	

	nb.fit(X_train_tfidf, y_train)
	y_pred = nb.predict(X_test_tfidf)
	print "Multinomial Naive Bayes"
	print "~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	mnb_accuracy.append(accuracy)
	mnb_precision.append(float(temp[17]))
	mnb_recall.append(float(temp[18]))
	mnb_fscore.append(float(temp[19]))	# write the f1-score which is at 19th index of variable temp. write the value for 5 folds and take avg

	lreg.fit(X_train_tfidf, y_train)
	y_pred = lreg.predict(X_test_tfidf)
	print "Logistic Regression"
	print "~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	lreg_accuracy.append(accuracy)
	lreg_precision.append(float(temp[17]))
	lreg_recall.append(float(temp[18]))
	lreg_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	svc.fit(X_train_tfidf, y_train)
	y_pred = svc.predict(X_test_tfidf)
	print "Support Vector Machines (Linear kernel)"
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	report = classification_report(y_test, y_pred)
	print report
	accuracy = accuracy_score(y_test, y_pred)
	print "Accuracy: ",accuracy, "\n"
	temp = report.split()
	svm_accuracy.append(accuracy)
	svm_precision.append(float(temp[17]))
	svm_recall.append(float(temp[18]))
	svm_fscore.append(float(temp[19]))	# write the f1-score during each iteration

	print "\n"

# accuracy scores
mnb_avg = numpy.average(mnb_accuracy)
lreg_avg = numpy.average(lreg_accuracy)
svm_avg = numpy.average(svm_accuracy)

t0.add_row(['custom stopwords', mnb_avg, lreg_avg, svm_avg])

# precision scores
mnb_avg = numpy.average(mnb_precision)
lreg_avg = numpy.average(lreg_precision)
svm_avg = numpy.average(svm_precision)

t1.add_row(['custom stopwords', mnb_avg, lreg_avg, svm_avg])

# recall scores
mnb_avg = numpy.average(mnb_recall)
lreg_avg = numpy.average(lreg_recall)
svm_avg = numpy.average(svm_recall)

t2.add_row(['custom stopwords', mnb_avg, lreg_avg, svm_avg])

# f1 measure scores
mnb_avg = numpy.average(mnb_fscore)
lreg_avg = numpy.average(lreg_fscore)
svm_avg = numpy.average(svm_fscore)

t3.add_row(['custom stopwords', mnb_avg, lreg_avg, svm_avg])

print "============================================================================="

print "ACCURACY\n"
print t0
print "PRECISION\n"
print t1
print "RECALL\n"
print t2
print "F1-MEASURE\n"
print t3

fp = open("review_summary_stats.txt","w")
fp.write("\nACCURACY\n")
fp.write("=====================\n")
fp.write(str(t0))
fp.write("\nPRECISION\n")
fp.write("=====================\n")
fp.write(str(t1))
fp.write("\nRECALL\n")
fp.write("=====================\n")
fp.write(str(t2))
fp.write("\nF1 SCORE\n")
fp.write("=====================\n")
fp.write(str(t3))
fp.close








