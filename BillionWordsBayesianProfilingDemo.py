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

min_frequency = 1
prior = 250

NUM_WORDS=10000
INDEX_FROM=2 

# Data obtained from https://www.kaggle.com/c/billion-word-imputation

f = open("train_v2.txt")
sentences = f.read().split("\n")
f.close()

vectorizer_X = CountVectorizer(min_df=min_frequency, max_features=NUM_WORDS, binary=True)
X = vectorizer_X.fit_transform(sentences)

f_vectorizer_X = open("vectorizer_X.pickle", "wb")
pickle.dump(vectorizer_X, f_vectorizer_X, protocol=4)
f_vectorizer_X.close()

f_X = open("X.pickle", "wb")
pickle.dump(X, f_X, protocol=4)
f_X.close()

#print("Loading Vectorizer")
#f_vectorizer_X = open("vectorizer_X.pickle", "rb")
#vectorizer_X = pickle.load(f_vectorizer_X)
#f_vectorizer_X.close()

#print("Loading Data")
#f_X = open("X.pickle", "rb")
#X = pickle.load(f_X)
#f_X.close()

X_csc = csc_matrix(X)

feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
target_id = vectorizer_X.vocabulary_[target_word]

word_count = np.array(X.sum(axis=0)).reshape(-1)
print(word_count.shape)
word_p = word_count / word_count.sum()

word_profile = dok_array((number_of_features, number_of_features), dtype=np.float32)
for i in range(number_of_features):
	if i%10 == 0:
		print(i)

	word_score = np.array(X[(X_csc.getcol(i)==1).toarray()[:,0]].sum(axis=0)).reshape(-1)
	word_score = word_score / word_score.sum()
	word_score = np.log(word_score / word_p)
	word_score *= word_count / (prior + word_count)
	word_score = np.where(word_score > 0.01, word_score, 0)
	sorted_ids = np.argsort(-word_score)[1:]
	word_profile[i, sorted_ids] = -1*np.sort(-word_score)[1:]

	#word_score = word_score / word_p
	#word_score *= word_count / (prior + word_count)
	#sorted_ids = np.argsort(-word_score)[1:]
	#word_profile[i, sorted_ids] = np.where(np.log(-1*np.sort(-word_score)[1:]) > 0.01, np.log(-1*np.sort(-word_score)[1:]), 0)
word_profile = word_profile.tocsr()

print(feature_names[target_id])
print(feature_names[word_profile.getrow(target_id).indices])

pickle.dump(feature_names, open("words.p", "wb"))
pickle.dump(word_profile, open("word_profile.p", "wb"))


