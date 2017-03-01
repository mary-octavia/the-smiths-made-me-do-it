import os
import sys
import codecs
import string
import numpy as np
from glob import glob
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords

flabels = "labels-all.txt"


def load_data(fin=flabels):
	wpt = WordPunctTokenizer()
	X, vocab = [], []
	st = stopwords.words('french')
	with codecs.open(fin, "r", encoding="utf-8") as f:
		for line in f: # read each line
			line = line.replace("\n", "") # process newline character
			line = ((line.encode("utf8")).translate(None, string.punctuation)).decode("utf8")
			line = ((line.encode("utf8")).translate(None, "1234567890")).decode("utf8")
			line = " ".join(wpt.tokenize(line))
			for word in line.split():
				if word not in vocab and word not in st:
					vocab.append(word)
			X.append(line)

	print "len X:", len(X)
	print "len vocab:", len(vocab)
	print "vocab:", vocab
	return X, vocab

# labels = load_data()

def write_matrix_to_file(mat, fout):

	print "entered write_matrix_to_file"
	with open(fout, 'w') as f:
		for i in range(len(mat)):
			f.write(" ".join([str(i) for i in mat[i]]))
			if i != len(mat) - 1:
				f.write("\n")
	print "exited write_matrix_to_file"


def create_occ_matrix(labels, vocab, binarize=False):
	'''compute feature occurence matrix 
	from labels, using the features in vocab
	'''
	docs_vect = np.zeros((len(labels), len(vocab)), dtype=np.int32)
	for i in range(len(labels)):
		for j in range(len(vocab)):
			if binarize == True:
				if vocab[j] in labels[i]: # feature present in document
					docs_vect[i][j] = 1
			else:
				docs_vect[i][j] = labels[i].count(vocab[j])

	print("len docs_vect:" +str(len(docs_vect)))
	print("shape docs_vect:"+ str(docs_vect.shape))
	return docs_vect
# create_feature_matrix(docs)

def rank_distance(a, b):
	'''calculates the rank distance between two vectors
	'''
	a_ranks = list(np.argsort(a))
	b_ranks = list(np.argsort(b))
	rank_dist = 0
	for i in range(len(a)):
		rank_dist = rank_dist + abs(a_ranks.index(i) - b_ranks.index(i))
	return rank_dist


def create_rank_matrix(docs_m):
	rank_m = np.zeros((len(docs_m), len(docs_m)), dtype=np.int32)

	for i in range(len(docs_m)):
		for j in range(i+1, len(docs_m)):
			rank = rank_distance(docs_m[i], docs_m[j])
			rank_m[i,j] = rank
			rank_m[j,i] = rank
	print "rank matrix shape:", rank_m.shape

	write_matrix_to_file(rank_m, "rank_mat.txt")
	return rank_m


