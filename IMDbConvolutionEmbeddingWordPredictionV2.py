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

profile_size = 50

def plot_precision_recall_curve(scores, labels):
    max_score = scores.max(axis=1)
    max_score_index = scores.argmax(axis=1)
    sorted_index = np.argsort(-1*max_score)

    precision = []
    class_sum = []

    correct = 0.0
    total = 0.0
    for i in sorted_index:
        if max_score_index[i] == labels[i]:
            correct += 1
        total += 1

        if total > 100:
        	precision.append(correct/total)
        	class_sum.append(max_score[i])

    plt.plot(class_sum, precision)

    plt.grid()
    plt.xlabel("Max Class Sum")
    plt.ylabel("Accuracy")
    plt.savefig('Figure.pdf')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--num_clauses", default=10000, type=int)
parser.add_argument("--T", default=1000, type=int)
parser.add_argument("--s", default=10.0, type=float)
parser.add_argument("--target_tokens", default=['bad', 'nice', 'car'], nargs='+', type=str)
parser.add_argument("--hypervector_size", default=512, type=int)
parser.add_argument("--convolution_window", default=1, type=int)
parser.add_argument("--bits", default=256, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batches", default=100, type=int)
parser.add_argument("--skip", default=25, type=int)
parser.add_argument("--window_size", default=2, type=int)
parser.add_argument("--imdb_num_words", default=1000, type=int)
parser.add_argument("--imdb_index_from", default=2, type=int)
parser.add_argument("--number_of_examples", default=5000, type=int)

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
				if word_id >= args.skip + args.imdb_index_from:
					number_of_training_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_training_examples)
X_train = np.zeros((number_of_training_examples, args.window_size, 1, args.hypervector_size), dtype=np.uint32)
focus_token_train = np.zeros(number_of_training_examples, dtype=np.uint32)
window = deque([])
training_example_id = 0
for e in range(train_y.shape[0]):	
	for word_id in train_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id >= args.skip + args.imdb_index_from:
					for i in range(args.window_size):
						X_train[training_example_id, i, 0][encoding[window[i]]] = 1
					focus_token_train[training_example_id] = word_id - args.skip - args.imdb_index_from
					training_example_id += 1
				window.pop()
			window.appendleft(word_id)

window = deque([])
number_of_testing_examples = 0
for e in range(test_y.shape[0]):
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id >= args.skip + args.imdb_index_from:
					number_of_testing_examples += 1
				window.pop()
			window.appendleft(word_id)

print(number_of_testing_examples)
X_test = np.zeros((number_of_testing_examples, args.window_size, 1, args.hypervector_size), dtype=np.uint32)
focust_token_test = np.zeros(number_of_testing_examples, dtype=np.uint32)
window = deque([])
testing_example_id = 0
for e in range(test_y.shape[0]):	
	for word_id in test_x[e]:
		if word_id in encoding:
			if len(window) == args.window_size:
				if word_id >= args.skip + args.imdb_index_from:
					for i in range(args.window_size):
						X_test[testing_example_id, i, 0][encoding[window[i]]] = 1
					focus_token_test[testing_example_id] = word_id - args.skip - args.imdb_index_from
					testing_example_id += 1
				window.pop()
			window.appendleft(word_id)

# Train one TM per token
for j in range(len(args.target_tokens)):
	print("\nTraining token model for '%s'\n" % (args.target_tokens[j]))

	tm = MultiClassConvolutionalTsetlinMachine2D(args.num_clauses, args.T, args.s, (args.convolution_window, 1))
	
	Y_train = (training_focus_token_ids == word_to_id[args.target_tokens[j]]) # Creates training target, i.e., target token present/absent

	# Creates random training samples for balancing and speedup
	present_p = 0.5#(Y_train.sum()/Y_train.shape[0])*1.0 # Probability of sampling an example where the target token is present

	X_train_0 = X_train[Y_train==0] # Gets those training examples where the target token is absent
	X_train_1 = X_train[Y_train==1] # Gets those training examples where the target token is present

	# Sample 'number_of_examples' exmples
	X_train_balanced = np.zeros((args.number_of_examples, X_train.shape[1]), dtype=np.uint32)
	Y_train_balanced = np.zeros(args.number_of_examples, dtype=np.uint32)
	for epoch in range(args.epochs):
		for k in range(args.number_of_examples):
			if np.random.rand() <= present_p:
				X_train_balanced[k,:] = X_train_1[np.random.randint(X_train_1.shape[0]),:].toarray()
				Y_train_balanced[k] = 1
			else:
				X_train_balanced[k,:] = X_train_0[np.random.randint(X_train_0.shape[0]),:].toarray()
				Y_train_balanced[k] = 0

		tm.fit(X_train_balanced, Y_train_balanced, incremental=True)

	# Create test data for token prediction

	Y_test = (testing_focus_token_ids == word_to_id[args.target_tokens[j]])

	X_test_0 = X_test[Y_test==0]
	X_test_1 = X_test[Y_test==1]

	X_test_balanced = np.zeros((args.number_of_examples, X_test.shape[1]), dtype=np.uint32)
	Y_test_balanced = np.zeros(args.number_of_examples, dtype=np.uint32)

	for k in range(args.number_of_examples):
		if (np.random.rand() <= present_p):
			X_test_balanced[k,:] = X_test_1[np.random.randint(X_test_1.shape[0]),:].toarray()
			Y_test_balanced[k] = 1
		else:
			X_test_balanced[k,:] = X_test_0[np.random.randint(X_test_0.shape[0]),:].toarray()
			Y_test_balanced[k] = 0

	(Y_test_balanced_predicted, Y_test_balanced_predicted_scores) = tm.predict(X_test_balanced, return_class_sums=True)

	print("Token: '%s' Accuracy: %.2f%% Precision: %.2f%% Recall: %.2f%%" % (args.target_tokens[j], 100*accuracy_score(Y_test_balanced, Y_test_balanced_predicted), 100*precision_score(Y_test_balanced, Y_test_balanced_predicted), 100*recall_score(Y_test_balanced, Y_test_balanced_predicted)))

	sorted_indexes = np.argsort(-1*Y_test_balanced_predicted_scores[:,1])
	print ("Example Prediction (Class Sum: %d)" % (Y_test_balanced_predicted_scores[sorted_indexes[0],1]), end=' ')
	for k in range(X_test_balanced.shape[1]):
		if X_test_balanced[sorted_indexes[0], k] == 1:
			print(feature_names[k], end=' ')
	print("->", args.target_tokens[j])

	print("\nPositive Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print(feature_names[k], end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print("¬" + feature_names[k - X_train.shape[1]], end=' ')

	print()
	print("\nNegative Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print(feature_names[k], end=' ')
	print()

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print("¬" + feature_names[k - X_train.shape[1]], end=' ')
	print()

	plot_precision_recall_curve(Y_test_balanced_predicted_scores, Y_test_balanced)