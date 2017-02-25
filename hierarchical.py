import codecs
import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy.spatial.distance as ssd
from itertools import cycle
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, RandomizedPCA, NMF
from sklearn.preprocessing import scale, Normalizer, Binarizer 
from sklearn.pipeline import Pipeline, make_pipeline
from time import time
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import random

'''beautify matplotlib
'''
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 25}

matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2)

# from reader import get_preprocessor

br_ft = "NLP4J-unigram-features/overuse/BR-test-WORD1-lsvc-OVERUSE.csv"
mo_ft = "NLP4J-unigram-features/overuse/MO-test-WORD1-lsvc-OVERUSE.csv"
pt_ft = "NLP4J-unigram-features/overuse/PT-test-WORD1-lsvc-OVERUSE.csv"

br_ft_nne = "NLP4J-filtered-NE-unigram-features/overuse/BR-test-WORD1-lsvc-OVERUSE.csv"
mo_ft_nne = "NLP4J-filtered-NE-unigram-features/overuse/MO-test-WORD1-lsvc-OVERUSE.csv"
pt_ft_nne = "NLP4J-filtered-NE-unigram-features/overuse/PT-test-WORD1-lsvc-OVERUSE.csv"
# f_raw_tok = "all-articles-raw-tokenized.txt"
f_raw_tok_ts = "all-articles-raw-tokenized-ts.txt"


def get_random_points3(n, order):
	'''randomly gets n/order indexes from n, which is an array
	representing 3 classes'''
	clen = int(n/3)
	idx1 = np.linspace(0, clen-1, clen, dtype=int)
	idx2 = np.linspace(clen, clen*2-1, clen*2, dtype=int)
	idx3 = np.linspace(clen*2, clen*3-1, clen*3, dtype=int)

	num =  int(clen/order)        # set the number to select here.
	lstr = random.sample(idx1, num)
	lstr = lstr + random.sample(idx2, num)
	lstr = lstr + random.sample(idx3, num)
	print "len lstr", len(lstr)
	return lstr


def get_random_points2(y, order):
	'''gets random points from y with 2 imbalanced classes
	y = list; 
	y[i] = 0 or 1 representing one of the two classes
	'''
	pclen = y.sum()
	idx1 = np.linspace(0, pclen-1, pclen, dtype=int)
	idx2 = np.linspace(pclen, len(y)-1, , dtype=int)



def load_data(fin=f_raw_tok_ts, merge=2):
# def load_data(fin, merge=2):
	'''
	'''
	docs, y = [], []
	with codecs.open(fin, "r", encoding="utf-8") as f:
		for line in f:
			line = line.replace("\n", "")
			docs.append(line)

	# y = 18000 *[1] + 18000*[2] + 18000*[3]
	# y = 2000 *[1] + 2000*[2] + 2000*[3]
	if merge == 1: # merge articles into one part each
		y = ["BR"] + ["MO"] + ["PT"]
		br = " ".join(docs[:2000])
		mo = " ".join(docs[2000:4000])
		pt = " ".join(docs[4000:])
		docs = [br] + [mo] + [pt]
		return docs, y
	elif merge == 2: # merge articles into two parts 
		# y = 2*[1] + 2*[2] + 2*[3]
		y = 2*["BR"] + 2*["MO"] + 2*["PT"]
		br1 = " ".join(docs[:1000])
		br2 = " ".join(docs[1000:2000])
		mo1 = " ".join(docs[2000:3000])
		mo2 = " ".join(docs[3000:4000])
		pt1 = " ".join(docs[4000:5000])
		pt2 = " ".join(docs[5000:])
		docs = [br1] + [br2] + [mo1] + [mo2] + [pt1] + [pt2]
		return docs, y
	else:#extract random samples 
		y = 2000 * ["BR"] + 2000 * ["MO"] + 2000 * ["PT"]
		n_docs, n_y = [],[]
		idx = get_random_points3(len(docs), order=120)
		for i in idx:
			n_docs.append(docs[i])
			n_y.append(y[i])
		n_docs, n_y = np.array(n_docs), np.array(n_y)
		return n_docs, n_y


def create_feature_vectors(docs, fin=br_ft):
	ct = []
	with codecs.open(fin, "r", encoding="utf-8") as f:
		for line in f:
			line = line.replace("\n", "")
			aux = line.split(",")
			if len(aux)>=9:
				aux = aux[2].replace('"', '')
				aux = aux.replace("=", "")
				ct.append(aux)
	# ct = ct[1:101]
	ct = ct[1:]
	print("len ct", len(ct))

	docs_vect = np.zeros((len(docs), len(ct)), dtype=np.int32)
	for i in range(len(docs)):
		for j in range(len(ct)):
			if ct[j] in docs[i]: # feature present in document
				docs_vect[i][j] = docs_vect[i][j] + 1

	aux = docs_vect.tolist()
	for i in range(len(aux)):
		if aux[i] < 0:
			print i

	print("len docs vect:" +str(len(docs_vect)))
	print("shape docs_vect:"+ str(docs_vect.shape))
	return docs_vect


def plot_agglomerative(X, n_labels, y):
	# Compute clustering
	print("Compute unstructured hierarchical clustering...")
	st = time()
	ward = AgglomerativeClustering(n_clusters=n_labels, linkage='ward').fit(X)
	elapsed_time = time() - st
	# label = ward.labels_
	# label = 13 * [1] + 13 * [2] + 13 * [3]
	# print "label ", label, len(label)
	# print("Elapsed time: %.2fs" % elapsed_time)
	# print("Number of points: %i" % label.size)

	# Plot result
	# fig = plt.figure()
	# ax = p3.Axes3D(fig)
	# ax.view_init(7, -80)
	# for l in np.unique(label):
	#     ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
	#               'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)), label=labels)
	# plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)
	# plt.savefig("dendogram.png")
	plot_agg_dendrogram(ward, leaf_rotation=0., leaf_font_size=15., labels=labels)


def plot_agg_dendrogram(model, **kwargs):

	print("entered plot_dendrogram")
	# Children of hierarchical clustering
	children = model.children_

	# Distances between each pair of children
	# Since we don't have this information, we can use a uniform one for plotting
	distance = np.arange(children.shape[0])

	# The number of observations contained in each cluster level
	no_of_observations = np.arange(2, children.shape[0]+2)

	# Create linkage matrix and then plot the dendrogram
	linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

	# c, coph_dists = cophenet(linkage_matrix, distance)
	# print("coph dist skl " + str(c))

	# Plot the corresponding dendrogram
	plt.figure(figsize=(25, 10))
	dendrogram(linkage_matrix, **kwargs)

	# plt.title('Dendrogram on merged datasets')
	plt.title('Dendrogram of '+str(len(labels))+' random articles')
	# plt.savefig("dend.png", format="png", dpi=500)
	plt.savefig("dend.pdf", format="pdf", dpi=500, bbox_inches='tight')


def get_sqform():

	# convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(distMatrix) # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j


def hierarchical(X, lb):
	'''using scipy
	'''
	# generate the linkage matrix
	print("entered hierarchical")
	Y = pdist(X)
	Z = linkage(Y, 'average')

	c, coph_dists = cophenet(Z, pdist(X))
	print("coph dist" + str(c))

	plt.figure(figsize=(25, 10))
	if len(lb) == 3:
		plt.title('Dendrogram on merged datasets')
	else:
		plt.title('Dendrogram of '+str(len(labels))+' random articles')
	# plt.xlabel('sample index')
	# plt.ylabel('distance')
	dendrogram(Z,
	    leaf_rotation=0.,  # rotates the x axis labels
	    leaf_font_size=50.,  # font size for the x axis labels
	    show_contracted=True,
	    labels=lb
	)
	# plt.savefig("scipyhierdend.png", format="png", dpi=500)
	plt.savefig("scipyhierdend.pdf", format="pdf", dpi=500)


# if __name__ == '__main__':
# 	docs, labels = load_data(merge=1)
# 	n_labels = len(set(labels))
# 	data_v = create_feature_vectors(docs)
# 	plot_agglomerative(data_v, 3, labels)
# 	hierarchical(data_v, labels)