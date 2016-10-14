# import metrics
import re
import json
import codecs
import string
import numpy as np
import sklearn.feature_selection as fs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline
from sklearn import naive_bayes
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.cross_validation import KFold, LeaveOneOut, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.pipeline import FeatureUnion
# from sklearn.feature_selection import SelectKBest



def get_preprocessor(suffix=''):
    def preprocess(unicode_text):
        return unicode(unicode_text.strip().lower() + suffix)
    return preprocess

def load_data(filename='sm-vs-all-lyrics.txt'):
    lyrics, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            lyr, label = line.split("\t")
            lyrics.append(lyr)
            y.append(int(label))
    lyrics, y = np.array(lyrics), np.array(y, dtype=np.int)

    for i in range(len(lyrics)):
    	lyrics[i] = lyrics[i].replace("_", " _ ")
    return lyrics, y


def preprocess_data(X, n, suffix='', binarize=True):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1),
                                 preprocessor=get_preprocessor(suffix))
    X = vectorizer.fit_transform(X)
    X = Binarizer(copy=False).fit_transform(X) if binarize else X
    return X

def compute_lexical_density(lyrics):
	'''unique tokens/ tokens''' 
	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	lyrics = lyrics.translate(replace_punctuation)
	lyrics = lyrics.decode("utf8")
	lyrics = lyrics.split()
	unique_w = []
	for word in lyrics:
		word = word.lower()
		if word not in unique_w:
			unique_w.append(word)
	return float(len(unique_w))/float(len(lyrics))
	
def compute_lexical_richness(lyrics):
	'''unique stems / stems'''
	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	lyrics = lyrics.translate(replace_punctuation)
	lyrics =lyrics.decode("utf8")
	# print lyrics.encode("utf8") #debug
	lyrics = lyrics.split()
	stemmer = SnowballStemmer("english")
	stems = []
	unique_s = []
	for word in lyrics:
		stems.append(stemmer.stem(word))
	for stem in stems:
		if stem not in unique_s:
			unique_s.append(stem)
	# print len(unique_s)," ", len(stems) #debug
	return float(len(unique_s))/float(len(stems))


def extract_lexical_features(lyrics, fname):
	'''extract lexical features from lyrics
	and write them to file fname 
	'''
	with open(fname, "w") as f:
		print lyrics[0]
		for lyric in lyrics:
			# print lyric, "\n"
			ld = compute_lexical_density(lyric)
			lr = compute_lexical_richness(lyric)
			f.write(str(ld) + "," + str(lr) + "\n")



def get_best_features(X, y, vectorizer):
	'''get names of best features in X from vectorizer'''
	fnames = vectorizer.get_feature_names()
	b = fs.SelectKBest(fs.f_classif, k=20) #k is number of features.
	X_n = b.fit_transform(X, y)
	index_v =  b.get_support()

	print "best unigrams:"
	for i in range(len(index_v)):
		if index_v[i] == True:
			print fnames[i]

if __name__ == '__main__':
	f_lyrics = "sm-vs-all-lyrics.txt"
	f_lexft = "lexical-features.txt"

	X, y = load_data(f_lyrics)

	# extract_lexical_features(X, f_lexft)

	'''get best unigram features with ANOVA'''
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
	X_new = vectorizer.fit_transform(X)
	get_best_features(X_new, y, vectorizer)


	'''cross-validation block'''
	skf = StratifiedKFold(y, n_folds=10)
	X_new = preprocess_data(X, n=3, suffix="", binarize=True)
	clf = LinearSVC()

	accuracy, recall, precision, f1 = [], [], [], []
	for train_index, test_index in skf:
		X_train, X_test = X_new[train_index], X_new[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print "fitting the classifier"
		clf.fit(X_train, y_train)

		print "predicting"
		y_pred = clf.predict(X_test)
		print classification_report(y_test, y_pred)
		accuracy.append(accuracy_score(y_test, y_pred))
		precision.append(precision_score(y_test, y_pred, pos_label=1, average='binary'))
		recall.append(recall_score(y_test, y_pred, pos_label=1, average='binary'))
		f1.append(f1_score(y_test, y_pred, pos_label=1, average='binary'))

	print "accuracy mean ", np.mean(accuracy), " accuracy std ", np.std(accuracy)
	print "precision mean ", np.mean(precision), " and std ", np.std(precision)
	print "recall mean ", np.mean(recall), " and std ", np.std(recall)
	print "f1 mean ", np.mean(f1), " and std ", np.std(f1)