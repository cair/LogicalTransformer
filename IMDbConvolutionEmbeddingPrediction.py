import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

from collections import deque

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PyCoalescedTsetlinMachineCUDA.tm import MultiOutputTsetlinMachine

epochs = 25

window_size = 1

batches = 100

hypervector_size = 512
bits = 256

clauses = 1000
T = 1000
s = 40.0

NUM_WORDS=1000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x, train_y = train
test_x, test_y = test

#train_x = train_x[0:1000]
#train_y = train_y[0:1000]

#test_x = test_x[0:1000]
#test_y = test_y[0:1000]

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

# Read from file instead, otherwise the same

print("Retrieving embeddings...")

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = {}
for i in range(NUM_WORDS+INDEX_FROM):
	encoding[i] = np.random.choice(indexes, size=(bits), replace=False)

# encoding = {}
# f = open("/data/near-lossless-binarization/binary_vectors_1024.vec", "r")
# line = f.readline()
# line = f.readline().strip()
# while line:
# 	entries = line.split(" ")
# 	if entries[0] in word_to_id:
# 		values = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8))
# 		encoding[word_to_id[entries[0]]] = np.unpackbits(np.fromstring(" ".join(entries[1:]), dtype=np.int64, sep=' ').view(np.uint8)).nonzero()
# 	line = f.readline().strip()
# f.close()
	
print("Producing bit representation...")

window = deque([])
number_of_training_examples = 0
for e in range(train_y.shape[0]):
	for word_id in train_x[e]:
		if word_id in encoding:
			if len(window) == window_size:
				number_of_training_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_training_examples)
X_train = np.zeros((number_of_training_examples, window_size, 1, hypervector_size), dtype=np.uint32)
Y_train = np.zeros((number_of_training_examples, hypervector_size), dtype=np.uint32)
window = deque([])
training_example_id = 0
for e in range(train_y.shape[0]):	
	for word_id in train_x[e]:
		if word_id in encoding:
			if len(window) == window_size:
				for i in range(window_size):
					X_train[training_example_id, i, 0][encoding[window[i]]] = 1
				Y_train[training_example_id][encoding[word_id]] = 1
				training_example_id += 1
				window.pop()
			window.appendleft(word_id)

window = deque([])
number_of_testing_examples = 0
for e in range(test_y.shape[0]):
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == window_size:
				number_of_testing_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_testing_examples)
X_test = np.zeros((number_of_testing_examples, window_size, 1, hypervector_size), dtype=np.uint32)
Y_test = np.zeros((number_of_testing_examples, hypervector_size), dtype=np.uint32)
window = deque([])
testing_example_id = 0
for e in range(test_y.shape[0]):
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == window_size:
				for i in range(window_size):
					X_test[testing_example_id, i, 0][encoding[window[i]]] = 1
				Y_test[testing_example_id][encoding[word_id]] = 1
				testing_example_id += 1
				window.pop()
			window.appendleft(word_id)

X_train = X_train.reshape(number_of_training_examples, -1)
X_test = X_test.reshape(number_of_testing_examples, -1)

batch_size_train = Y_train.shape[0] // batches
batch_size_test = Y_test.shape[0] // batches

tm = MultiOutputTsetlinMachine(clauses, T, s)
for i in range(epochs):
	for batch in range(batches):
		print("Fit")
		start_training = time()
		tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train[batch*batch_size_train:(batch+1)*batch_size_train], epochs=1, incremental=True)
		stop_training = time()

		print("Predict Test")
		start_testing = time()
		Y_test_predicted = tm.predict(X_test[batch*batch_size_test:(batch+1)*batch_size_test])
		result_test = 100*(Y_test_predicted == Y_test[batch*batch_size_test:(batch+1)*batch_size_test]).mean()
		f1_test = f1_score(Y_test[batch*batch_size_test:(batch+1)*batch_size_test], Y_test_predicted, average='macro')
		stop_testing = time()

		print("Predict Train")
		Y_train_predicted = tm.predict(X_train[batch*batch_size_train:(batch+1)*batch_size_train])
		result_train = 100*(Y_train_predicted == Y_train[batch*batch_size_train:(batch+1)*batch_size_train]).mean()
		f1_train = f1_score(Y_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train_predicted, average='macro')

		print("#%d/%d Accuracy Test: %.2f%% Accuracy Train: %.2f%% Training: %.2fs Testing: %.2fs" % (batch+1, i+1, result_test, result_train, stop_training-start_training, stop_testing-start_testing))