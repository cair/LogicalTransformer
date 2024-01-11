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
parser.add_argument("--context_size", default=5, type=int)
parser.add_argument("--number_of_examples", default=5000, type=int)
parser.add_argument("--imdb-num-words", default=10000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
args = parser.parse_args()

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

print("Producing token-centered datasets...")

training_focus_token_ids = [] # Here the id of the token in the centre of the sliding window is stored
training_documents = [] # Here the words surrounding the center token is stored. The size of the context window is set by the argument 'context size'

# First, iterate through each IMDB training document
for i in range(train_y.shape[0]):
	# Then iterate through each token in the current document
	focus_token_position = 0 # Tracks the position of the token in focus inside the document
	for focus_token_id in train_x[i]:
		focus_token = id_to_word[focus_token_id].lower()
		if focus_token not in args.target_tokens:
			focus_token_position += 1 # Skip those tokens not targeted
			continue

		# Find the tokens surrounding the focus token
		tokens = []
		context_token_position = 0 # Keeps track of position of context token in document
		for context_token_id in train_x[i]:
			# Skip tokens outside the context window
			if np.abs(context_token_position - focus_token_position) <= args.context_size and context_token_position != focus_token_position:
				tokens.append(str(context_token_position - focus_token_position) + ":" + id_to_word[context_token_id].lower()) # Concatenate context token with relative position in window

			context_token_position += 1 

		training_focus_token_ids.append(focus_token_id)	# Stores center-token
		training_documents.append(tokens) # Stores context tokens

		focus_token_position += 1

testing_focus_token_ids = []
testing_documents = []

# Iterate through each IMDB testing document
for i in range(test_y.shape[0]):
	# Then iterate through each token in the current document

	focus_token_position = 0  # Tracks the position of the token in focus inside the document
	for focus_token_id in test_x[i]:
		focus_token = id_to_word[focus_token_id].lower()
		if focus_token not in args.target_tokens:
			focus_token_position += 1 # Skip those tokens not targeted
			continue

		# Find the tokens surrounding the focus token
		tokens = []
		context_token_position = 0 # Keeps track of position of context token in document
		for context_token_id in test_x[i]:
			if np.abs(context_token_position - focus_token_position) <= args.context_size and context_token_position != focus_token_position:
				tokens.append(str(context_token_position - focus_token_position) + ":" + id_to_word[context_token_id].lower()) # Concatenate context token with relative position in window

			context_token_position += 1

		testing_focus_token_ids.append(focus_token_id) # Stores center-token
		testing_documents.append(tokens) # Stores context tokens

		focus_token_position += 1

print("Producing token-centered datasets... Done")

print("Producing bit representation...")

vectorizer_X = CountVectorizer(
    tokenizer=lambda s: s,
    token_pattern=None,
    lowercase=False,
    binary=True
)

X_train = vectorizer_X.fit_transform(training_documents).astype(np.uint32)
feature_names = vectorizer_X.get_feature_names_out()

X_test = vectorizer_X.transform(testing_documents).astype(np.uint32)

training_focus_token_ids = np.array(training_focus_token_ids, dtype=np.uint32)
testing_focus_token_ids = np.array(testing_focus_token_ids, dtype=np.uint32)

# Train one TM per token
for j in range(len(args.target_tokens)):
	print("\nTraining token model for '%s'\n" % (args.target_tokens[j]))

	tm = TMClassifier(args.num_clauses, args.T, args.s, weighted_clauses=args.weighted_clauses, max_included_literals=32)

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

		tm.fit(X_train_balanced, Y_train_balanced)

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