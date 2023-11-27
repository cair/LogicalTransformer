import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix, csr_matrix, dok_array
import sys
import pickle

print(sys.version)

#target_word = 'awful' #'frightening'#'comedy'#'romance'#"scary"
#target_word='romance'
#target_word='comedy'
#target_word = 'brilliant'
#target_word = 'frightening'
target_word = 'king'

min_frequency = 5
prior = 250
threshold = 1.0
profile_words = 10000

NUM_WORDS=25000
INDEX_FROM=2 

f = open("/Users/oleg/Dropbox/Research/Datasets/wiki-english-20171001")
sentences = f.read().split("\n")
f.close()

vectorizer_X = CountVectorizer(min_df=min_frequency, max_features=NUM_WORDS, binary=True)
X = vectorizer_X.fit_transform(sentences)

f_vectorizer_X = open("wiki_vectorizer_X.pickle", "wb")
pickle.dump(vectorizer_X, f_vectorizer_X, protocol=4)
f_vectorizer_X.close()

f_X = open("wiki_X.pickle", "wb")
pickle.dump(X, f_X, protocol=4)
f_X.close()

#print("Loading Vectorizer")
#f_vectorizer_X = open("wiki_vectorizer_X.pickle", "rb")
#vectorizer_X = pickle.load(f_vectorizer_X)
#f_vectorizer_X.close()

#print("Loading Data")
#f_X = open("wiki_X.pickle", "rb")
#X = pickle.load(f_X)
#f_X.close()

X_csc = csc_matrix(X)

feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
target_id = vectorizer_X.vocabulary_[target_word]

word_count = np.array(X.sum(axis=0)).reshape(-1)
word_p = word_count / word_count.sum()
print(word_count.shape)

sorted_ids = np.argsort(-1*word_count)

f = open("Wiki_Bayesian_profile_%d_%d_%.2f_%d.vec" % (NUM_WORDS, prior, threshold, profile_words), "w+")
f.write("%d %d\n" % (X.shape[1], profile_words))

for i in range(number_of_features):
	if i%100 == 0:
		print(i)

	word_score = np.array(X[(X_csc.getcol(i)==1).toarray()[:,0]].sum(axis=0)).reshape(-1)
	word_score = word_score / word_score.sum()
	word_score = np.log(word_score / word_p)
	word_score *= word_count / (prior + word_count)
	word_score = np.where(word_score >= threshold, word_score, 0)
	
	word_score = word_score[sorted_ids[:profile_words]]

	f.write(feature_names[i] + " ")
	np.savetxt(f, word_score.reshape((1,-1)), delimiter=' ', fmt='%.2f')

f.close()

