import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, top_k_accuracy_score
import argparse

from collections import deque

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PyCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

parser = argparse.ArgumentParser()
parser.add_argument("--num_clauses", default=10000, type=int)
parser.add_argument("--T", default=1000, type=int)
parser.add_argument("--s", default=10.0, type=float)
parser.add_argument("--hypervector_size", default=512, type=int)
parser.add_argument("--convolution_window", default=1, type=int)
parser.add_argument("--bits", default=256, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batches", default=100, type=int)
parser.add_argument("--skip", default=25, type=int)
parser.add_argument("--window_size", default=2, type=int)
parser.add_argument("--number_of_examples", default=5000, type=int)
parser.add_argument("--imdb_num_words", default=1000, type=int)
parser.add_argument("--imdb_index_from", default=2, type=int)

args = parser.parse_args()

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)

train_x, train_y = train
test_x, test_y = test

#train_x = train_x[0:1000]
#train_y = train_y[0:1000]

#test_x = test_x[0:1000]
#test_y = test_y[0:1000]

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+args.imdb_index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}

# Read from file instead, otherwise the same

print("Retrieving embeddings...")

indexes = np.arange(args.hypervector_size, dtype=np.uint32)
encoding = {}
for i in range(args.imdb_num_words+args.imdb_index_from):
	encoding[i] = np.random.choice(indexes, size=(args.bits), replace=False)

# encoding = {}
# f = open("/data/near-lossless-binarization/binary_vectors_512.vec", "r")
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
			if len(window) == args.window_size:
				if word_id > args.skip:
					number_of_training_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_training_examples)
X_train = np.zeros((number_of_training_examples, args.window_size, 1, args.hypervector_size), dtype=np.uint32)
Y_train = np.zeros(number_of_training_examples, dtype=np.uint32)
window = deque([])
training_example_id = 0
for e in range(train_y.shape[0]):	
	for word_id in train_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id > args.skip:
					for i in range(args.window_size):
						X_train[training_example_id, i, 0][encoding[window[i]]] = 1
					Y_train[training_example_id] = word_id
					training_example_id += 1
				window.pop()
			window.appendleft(word_id)

window = deque([])
number_of_testing_examples = 0
for e in range(test_y.shape[0]):
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id > args.skip:
					number_of_testing_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_testing_examples)
X_test = np.zeros((number_of_testing_examples, args.window_size, 1, args.hypervector_size), dtype=np.uint32)
Y_test = np.zeros(number_of_testing_examples, dtype=np.uint32)
window = deque([])
testing_example_id = 0
for e in range(test_y.shape[0]):	
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id > args.skip:
					for i in range(args.window_size):
						X_test[testing_example_id, i, 0][encoding[window[i]]] = 1
					Y_test[testing_example_id] = word_id
					testing_example_id += 1
				window.pop()
			window.appendleft(word_id)

batch_size_train = Y_train.shape[0] // args.batches
batch_size_test = Y_test.shape[0] // args.batches

tm = MultiClassConvolutionalTsetlinMachine2D(args.num_clauses, args.T, args.s, (args.convolution_window, 1))
for i in range(args.epochs):
	for batch in range(args.batches):
		print("Fit")
		start_training = time()
		tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train[batch*batch_size_train:(batch+1)*batch_size_train], epochs=1, incremental=True)
		stop_training = time()

		print("Predict Test")
		start_testing = time()
		Y_test_score = tm.score(X_test[batch*batch_size_test:(batch+1)*batch_size_test])
		print(Y_test_score.shape)
		Y_test_predicted = np.argmax(Y_test_score, axis=0)
		result_test = 100*(Y_test_predicted == Y_test[batch*batch_size_test:(batch+1)*batch_size_test]).mean()
		f1_test = 100*f1_score(Y_test[batch*batch_size_test:(batch+1)*batch_size_test], Y_test_predicted, average='macro')
		top_k_accuracy_test = 100*top_k_accuracy_score(Y_test[batch*batch_size_test:(batch+1)*batch_size_test], Y_test_score, k=10)
		stop_testing = time()

		print("Predict Train")
		Y_train_score = tm.score(X_train[batch*batch_size_train:(batch+1)*batch_size_train])
		Y_train_predicted = np.argmax(Y_train_score, axis=0)
		result_train = 100*(Y_train_predicted == Y_train[batch*batch_size_train:(batch+1)*batch_size_train]).mean()
		f1_train = 100*f1_score(Y_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train_predicted, average='macro')
		top_k_accuracy_train = 100*top_k_accuracy_score(Y_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train_score, k=10)

		print("#%d/%d F1 Test: %.2f%% F1 Train: %.2f%% Top-k Test: %.2f%% Top-k Train: %.2f%% Training: %.2fs Testing: %.2fs" % (batch+1, i+1, f1_test, f1_train, top_k_accuracy_test, top_k_accuracy_train, stop_training-start_training, stop_testing-start_testing))