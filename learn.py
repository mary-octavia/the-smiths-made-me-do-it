# import metrics
import re
import json
import codecs
import string
import copy as cp
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
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
# from sklearn.feature_selection import SelectKBest


def get_preprocessor(suffix=''):
    def preprocess(unicode_text):
        return unicode(unicode_text.strip().lower() + suffix)
    return preprocess


def preprocess_data(X, n, suffix='', binarize=True):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1),
                                 preprocessor=get_preprocessor(suffix))
    X = vectorizer.fit_transform(X)
    X = Binarizer(copy=False).fit_transform(X) if binarize else X
    return X


def load_data(filename='sm-vs-all-lyrics.txt'):
    lyrics, y = [], []

    with open(filename, 'r') as f:
        for line in f:
            aux = line.split("\t")
            if len(aux) != 2:
            	print "aux", aux
            else:
            	lyr, label = aux[0], aux[1]
            	lyrics.append(preprocess_lyric(lyr))
            	y.append(int(label))

    lyrics, y = np.array(lyrics), np.array(y, dtype=np.int)

    for i in range(len(lyrics)):
    	lyrics[i] = lyrics[i].replace("_", " _ ")
    return lyrics, y


def preprocess_lyric(lyric):
	new_lyric = cp.deepcopy(lyric)
	punct = (string.punctuation).replace("_", "")
	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	new_lyric = new_lyric.translate(replace_punctuation)
	new_lyric = new_lyric.decode("utf8")
	return new_lyric


class get_lex(BaseEstimator, TransformerMixin):
    '''transformer that gets lexical features for each 
    lyric in lyrics'''

    def __init__(self, lex):
    	self.lex = lex


    def fit(self, X, y=None):
    	return self

    def compute_lexical_density(self, lyrics):
		'''unique tokens/ tokens''' 
		# new_lyrics = preprocess_lyrics(lyrics)
		unique_w = []
		for word in lyrics:
			word = word.lower()
			if word not in unique_w:
				unique_w.append(word)
		# print "lyrics", lyrics
		# print "lyrics.split()", lyrics.split()
		return float(len(unique_w))/float(len(lyrics.split()))

    def compute_lexical_richness(self, lyrics):
		'''unique stems / stems'''
		# new_lyrics = preprocess_lyrics(lyrics)
		stemmer = SnowballStemmer("english")
		stems, unique_s = [], []

		for word in lyrics:
			stems.append(stemmer.stem(word))

		for stem in stems:
			if stem not in unique_s:
				unique_s.append(stem)
		# print len(unique_s)," ", len(stems) #debug
		return float(len(unique_s))/float(len(stems))


    def transform(self, X, y=None):
		print "entered extract lexical features"
		# new_X = preprocess_lyrics(X)
		new_X = cp.deepcopy(X)
		lexic = []
		for i in range(len(new_X)):
			if self.lex == 'dens':
				lexic.append(self.compute_lexical_density(new_X[i]))
			elif self.lex == 'rich':
				lexic.append(self.compute_lexical_richness(new_X[i]))
			else:
				print "error: command not supported"
				return
		print len(lexic)
		print "lexical features extracted"
		lexic = np.array(lexic)
		lexic = lexic.reshape(-1,1)

		return lexic

# def write_to_file(X, fname):

# 	# nx = cp.deepcopy(X)
# 	with open(fname, 'w') as f:
# 		for i in range(len(X)):
# 			f.write(X[i] + "\n")


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

	'''get best unigram features with ANOVA'''
	# vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
	# X_new = vectorizer.fit_transform(X)
	# get_best_features(X_new, y, vectorizer)


	'''cross-validation block'''
	skf = StratifiedKFold(y, n_folds=10)
	clf = LinearSVC(class_weight='balanced')

	bow_pipe = Pipeline([
                            ('bow-vectorizer', CountVectorizer(analyzer='word', ngram_range=(1,1))),
                            ('binarizer', Binarizer(copy=False))
                           ])

	ngram_pipe = Pipeline([
                            ('bow-vectorizer', CountVectorizer(analyzer='char', ngram_range=(4,4))),
                            ('binarizer', Binarizer(copy=False))
                           ])

	feature_union = FeatureUnion([
								# ('lex_dens', clone(get_lex('dens'))),
								# ('lex_rich', clone(get_lex('rich'))),
								('bow', bow_pipe),
								('ngram', ngram_pipe)
                                ])

	X_new = feature_union.fit_transform(X)
	print "tshape", X_new.shape
	# write_to_file(X_new, "lexic-ft")


	accuracy, recall, precision, f1 = [], [], [], []
	for train_index, test_index in skf:
		X_train, X_test = X_new[train_index], X_new[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print "fitting the classifier"
		clf.fit(X_train, y_train)

		print "predicting"
		y_pred = clf.predict(X_test)
		#debug------
		count_pred, count_tst = 0, 0
		print len(y_pred) == len(y_test), "len y_pred", len(y_pred)
		for i in range(len(y_pred)):
			if y_pred[i] == 1:
				count_pred += 1
		print "count_pred", count_pred
		print classification_report(y_test, y_pred)
		#debug-----
		accuracy.append(accuracy_score(y_test, y_pred))
		precision.append(precision_score(y_test, y_pred))
		recall.append(recall_score(y_test, y_pred))
		f1.append(f1_score(y_test, y_pred))

	print "accuracy mean ", np.mean(accuracy), " accuracy std ", np.std(accuracy)
	print "precision mean ", np.mean(precision), " and std ", np.std(precision)
	print "recall mean ", np.mean(recall), " and std ", np.std(recall)
	print "f1 mean ", np.mean(f1), " and std ", np.std(f1)