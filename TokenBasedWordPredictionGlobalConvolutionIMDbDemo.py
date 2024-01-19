from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import argparse
import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.ticker as mticker
from collections import deque
from scipy.sparse import csr_matrix

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
parser.add_argument("--num_clauses", default=100, type=int)
parser.add_argument("--T", default=1000, type=int)
parser.add_argument("--s", default=1.0, type=float)
parser.add_argument("--device", default="GPU", type=str)
parser.add_argument("--target_tokens", default=['bad', 'nice', 'car'], nargs='+', type=str)
parser.add_argument("--weighted_clauses", default=True, type=bool)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--context_size", default=3, type=int)
parser.add_argument("--convolution_size", default=1, type=int)
parser.add_argument("--number_of_examples", default=5000, type=int)
parser.add_argument("--imdb-num-words", default=10000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
args = parser.parse_args()

position_bits = args.context_size*2 - args.convolution_size

#### Retrieves and prepares the IMDb dataset ####

print("Preparing dataset")

train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value: key for key, value in word_to_id.items()}

print("Preparing dataset.... Done!")

#### Sliding window-based approach for producing the training and test data centered around each token #####

print("Producing token-centered training data...")

window = deque([])
number_of_training_examples = 0
for e in range(train_y.shape[0]):	
	for word_id in train_x[e]:
		if len(window) == args.context_size*2+1:
			number_of_training_examples += 1
			window.popleft()
		window.append(word_id)

X_train_data = np.zeros((number_of_training_examples * args.context_size*2), dtype=np.uint32)
X_train_indices = np.zeros((number_of_training_examples * args.context_size*2), dtype=np.uint32)
X_train_indptr = np.zeros((number_of_training_examples + 1), dtype=np.uint32)
training_focus_token_ids = np.zeros(number_of_training_examples, dtype=np.uint32)
X_train_indptr[0] = 0

window = deque([])
pos = 0
training_example_id = 0
for e in range(train_y.shape[0]):	
	for word_id in train_x[e]:
		if len(window) == args.context_size*2+1:
			training_focus_token_ids[training_example_id] = window[args.context_size]
			for i in range(args.context_size*2+1):
				if i < args.context_size:
					X_train_data[pos] = 1
					X_train_indices[pos] = i*(args.imdb_num_words + args.imdb_index_from) + window[i] 
					pos += 1
				elif i > args.context_size:
					X_train_data[pos] = 1
					X_train_indices[pos] = (i-1)*(args.imdb_num_words + args.imdb_index_from) + window[i]
					pos += 1
			X_train_indptr[training_example_id+1] = pos
			training_example_id += 1
			window.popleft()
		window.append(word_id)

X_train = csr_matrix((X_train_data, X_train_indices, X_train_indptr), (number_of_training_examples, 2*args.context_size*(args.imdb_num_words + args.imdb_index_from)))

print("Producing token-centered training data... Done")

print("Producing token-centered testing data...")

window = deque([])
number_of_testing_examples = 0
for e in range(test_y.shape[0]):	
	for word_id in test_x[e]:
		if len(window) == args.context_size*2+1:
			number_of_testing_examples += 1
			window.popleft()
		window.append(word_id)

X_test_data = np.zeros((number_of_testing_examples * args.context_size*2), dtype=np.uint32)
X_test_indices = np.zeros((number_of_testing_examples * args.context_size*2), dtype=np.uint32)
X_test_indptr = np.zeros((number_of_testing_examples + 1), dtype=np.uint32)
testing_focus_token_ids = np.zeros(number_of_testing_examples, dtype=np.uint32)
X_test_indptr[0] = 0

window = deque([])
pos = 0
testing_example_id = 0
for e in range(test_y.shape[0]):	
	for word_id in test_x[e]:
		if len(window) == args.context_size*2+1:
			testing_focus_token_ids[testing_example_id] = window[args.context_size]
			for i in range(args.context_size*2+1):
				if i < args.context_size:
					X_test_data[pos] = 1
					X_test_indices[pos] = i*(args.imdb_num_words + args.imdb_index_from) + window[i]
					pos += 1
				elif i > args.context_size:
					X_test_data[pos] = 1
					X_test_indices[pos] = (i-1)*(args.imdb_num_words + args.imdb_index_from) + window[i]
					pos += 1
			X_test_indptr[testing_example_id+1] = pos
			testing_example_id += 1
			window.popleft()
		window.append(word_id)

X_test = csr_matrix((X_test_data, X_test_indices, X_test_indptr), (number_of_testing_examples, 2*args.context_size*(args.imdb_num_words + args.imdb_index_from)))

print("Producing token-centered testing data... Done")

# Train one TM per token
for j in range(len(args.target_tokens)):
	print("\nTraining token model for '%s'\n" % (args.target_tokens[j]))

	tm = TMClassifier(args.num_clauses, args.T, args.s, patch_dim=(args.convolution_size, 1), weighted_clauses=args.weighted_clauses, max_included_literals=32)

	Y_train = (training_focus_token_ids == word_to_id[args.target_tokens[j]]) # Creates training target, i.e., target token present/absent

	# Creates random training samples for balancing and speedup
	present_p = 0.5#(Y_train.sum()/Y_train.shape[0])*1.0 # Probability of sampling an example where the target token is present

	X_train_0 = X_train[Y_train==0] # Gets those training examples where the target token is absent
	X_train_1 = X_train[Y_train==1] # Gets those training examples where the target token is present

	# Sample 'number_of_examples' exmples
	X_train_balanced = np.zeros((args.number_of_examples, 2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from), dtype=np.uint32)
	Y_train_balanced = np.zeros(args.number_of_examples, dtype=np.uint32)
	for epoch in range(args.epochs):
		for k in range(args.number_of_examples):
			if np.random.rand() <= present_p:
				X_train_balanced[k,:] = X_train_1[np.random.randint(X_train_1.shape[0]),:].toarray().reshape((2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from))
				Y_train_balanced[k] = 1
			else:
				X_train_balanced[k,:] = X_train_0[np.random.randint(X_train_0.shape[0]),:].toarray().reshape((2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from))
				Y_train_balanced[k] = 0

		tm.fit(X_train_balanced, Y_train_balanced)

	# Create test data for token prediction

	Y_test = (testing_focus_token_ids == word_to_id[args.target_tokens[j]])

	X_test_0 = X_test[Y_test==0]
	X_test_1 = X_test[Y_test==1]

	X_test_balanced = np.zeros((args.number_of_examples, 2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from), dtype=np.uint32)
	Y_test_balanced = np.zeros(args.number_of_examples, dtype=np.uint32)

	for k in range(args.number_of_examples):
		if (np.random.rand() <= present_p):
			X_test_balanced[k,:] = X_test_1[np.random.randint(X_test_1.shape[0]),:].toarray().reshape((2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from))
			Y_test_balanced[k] = 1
		else:
			X_test_balanced[k,:] = X_test_0[np.random.randint(X_test_0.shape[0]),:].toarray().reshape((2*args.context_size, 1, args.imdb_num_words + args.imdb_index_from))
			Y_test_balanced[k] = 0

	(Y_test_balanced_predicted, Y_test_balanced_predicted_scores) = tm.predict(X_test_balanced, return_class_sums=True)

	print("Token: '%s' Accuracy: %.2f%% Precision: %.2f%% Recall: %.2f%%" % (args.target_tokens[j], 100*accuracy_score(Y_test_balanced, Y_test_balanced_predicted), 100*precision_score(Y_test_balanced, Y_test_balanced_predicted), 100*recall_score(Y_test_balanced, Y_test_balanced_predicted)))

	sorted_indexes = np.argsort(-1*Y_test_balanced_predicted_scores[:,1])
	print ("Example Prediction (Class Sum: %d)" % (Y_test_balanced_predicted_scores[sorted_indexes[0],1]), end=' ')
	for k in range(args.context_size):
		print(id_to_word[np.flatnonzero(X_test_balanced[sorted_indexes[0], k, 0])[0]], end=' ')
	print(">", args.target_tokens[j], "<", end=' ')
	for k in range(args.context_size):
		print(id_to_word[np.flatnonzero(X_test_balanced[sorted_indexes[0], args.context_size + k, 0])[0]], end=' ')

	print("\nPositive Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k >= position_bits:
			window_id = (k - position_bits) // X_train_balanced.shape[3]
			window_offset = window_id * X_train_balanced.shape[3]

			print(str(window_id) + ":" + id_to_word[k - position_bits - window_offset], end=' ')
		else:
			print(k, end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		number_of_convolution_window_features = position_bits + (X_train_balanced.shape[3]*args.convolution_size)

		if k >= number_of_convolution_window_features + position_bits:
			window_id = (k - position_bits - number_of_convolution_window_features) // X_train_balanced.shape[3]
			window_offset = number_of_convolution_window_features + position_bits + window_id * X_train_balanced.shape[3]

			print("¬" + str(window_id) + ":" + id_to_word[k - window_offset], end=' ')
		else:
			print("¬" + str(k - number_of_convolution_window_features), end=' ')

	print()
	print("\nNegative Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k >= position_bits:
			window_id = (k - position_bits) // X_train_balanced.shape[3]
			window_offset = window_id * X_train_balanced.shape[3]

			print(str(window_id) + ":" + id_to_word[k - position_bits - window_offset], end=' ')
		else:
			print(k, end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		number_of_convolution_window_features = position_bits + (X_train_balanced.shape[3]*args.convolution_size)

		if k >= number_of_convolution_window_features + position_bits:
			window_id = (k - position_bits - number_of_convolution_window_features) // X_train_balanced.shape[3]
			window_offset = number_of_convolution_window_features + position_bits + window_id * X_train_balanced.shape[3]

			print("¬" + str(window_id) + ":" + id_to_word[k - window_offset], end=' ')
		else:
			print("¬" + str(k - number_of_convolution_window_features), end=' ')

	plot_precision_recall_curve(Y_test_balanced_predicted_scores, Y_test_balanced)