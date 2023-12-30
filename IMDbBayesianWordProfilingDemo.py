import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, dok_array
import sys
import pickle

print(sys.version)

#target_word = 'awful' #'frightening'#'comedy'#'romance'#"scary"
#target_word='romance'
#target_word='comedy'
#target_word = 'brilliant'
#target_word = 'frightening'
target_word = 'lousy'
#target_word = 'car'

min_frequency = 5
prior = 250
threshold = 0.5

NUM_WORDS=10000
INDEX_FROM=2 

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

id_to_word = {value:key for key,value in word_to_id.items()}

training_documents = []
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id].lower())
	
	training_documents.append(terms)

testing_documents = []
for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id].lower())
	
	testing_documents.append(terms)

def tokenizer(s):
	return s

vectorizer_X = CountVectorizer(
	tokenizer=tokenizer,
	lowercase=False,
	min_df=min_frequency,
	max_features=NUM_WORDS,
	binary=True)

X_train = vectorizer_X.fit_transform(training_documents).toarray()
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
target_id = vectorizer_X.vocabulary_[target_word]

print("Number of features:", number_of_features)

X_test = vectorizer_X.transform(testing_documents).toarray()

word_count = X_train.sum(axis=0)
word_p = word_count / word_count.sum()

word_profile = dok_array((number_of_features, number_of_features), dtype=np.float32)
for i in range(number_of_features):
	if i%10 == 0:
		print(i)

	word_score = X_train[X_train[:,i]==1].sum(axis=0)
	word_score = word_score / word_score.sum()
	word_score = word_score / word_p
	word_score *= word_count / (prior + word_count)
	sorted_ids = np.argsort(-word_score)[1:]
	word_profile[i, sorted_ids] = np.where(np.log(-1*np.sort(-word_score)[1:]) > threshold, np.log(-1*np.sort(-word_score)[1:]), 0)
word_profile = word_profile.tocsr()

print(feature_names[target_id])
print(feature_names[word_profile.getrow(target_id).indices])
print(word_profile.getrow(target_id))

pickle.dump(feature_names, open("words.p", "wb"))
pickle.dump(word_profile, open("word_profile.p", "wb"))


