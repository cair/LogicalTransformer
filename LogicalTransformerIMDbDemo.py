# Copyright (c) 2024 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import argparse
import logging
import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.ticker as mticker

examples = 5000
context_size = 5
profile_size = 50

def plot_precision_recall_curve(scores, labels):
    max_score = scores.max(axis=1)
    max_score_index = scores.argmax(axis=1)
    sorted_index = np.argsort(-1*max_score)

    precision = []
    recall = []
    class_sum = []

    correct = 0.0
    total = 0.0
    for i in sorted_index:
        if max_score_index[i] == Y_test[i]:
            correct += 1
        total += 1

        if total > 100:
        	precision.append(correct/total)
	        recall.append(total/sorted_index.shape[0])
        	class_sum.append(max_score[i]) 

        if total % 10 == 0:
            print("%d %.2f %.2f" % (max_score[i], total/sorted_index.shape[0], correct/total))

    plt.plot(class_sum, precision)
    #plt.plot(recall, precision)

    plt.grid()
    plt.xlabel("Max Class Sum")
    plt.ylabel("Accuracy")
    plt.savefig('Figure.pdf')
    plt.show()

#positive_sample_p = 0.1

_LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--num_clauses", default=10000, type=int)
parser.add_argument("--T", default=8000, type=int)
parser.add_argument("--s", default=2.0, type=float)
parser.add_argument("--device", default="GPU", type=str)
parser.add_argument("--weighted_clauses", default=True, type=bool)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--clause_drop_p", default=0.75, type=float)
parser.add_argument("--max-ngram", default=1, type=int)
parser.add_argument("--features", default=100, type=int)
parser.add_argument("--imdb-num-words", default=10000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
args = parser.parse_args()

_LOGGER.info("Preparing dataset...")
logging.basicConfig(level=logging.INFO)

train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
_LOGGER.info("Preparing dataset... Done!")

id_to_word = {value: key for key, value in word_to_id.items()}

tokens = {}
for i in range(train_y.shape[0]):
	for token_id in train_x[i]:
		tokens[id_to_word[token_id]] = True

tms = []
for target_token in ['bad']:#tokens.keys():
	_LOGGER.info("Producing self-attention datasets...")

	training_documents = []
	training_class = []
	testing_documents = []
	testing_class = []

	for i in range(train_y.shape[0]):
		focus_token_position = 0
		for focus_token_id in train_x[i]:
			focus_token = id_to_word[focus_token_id]
			if focus_token != target_token:
				focus_token_position += 1
				continue

			terms = []
			context_token_position = 0
			for context_token_id in train_x[i]:
				if np.abs(context_token_position - focus_token_position) <= context_size and context_token_position != focus_token_position:
					terms.append(str(context_token_position - focus_token_position) + ":" + id_to_word[context_token_id].lower())

					#if id_to_word[context_token_id].lower() != target_token:
					#	terms.append(id_to_word[context_token_id].lower())

				context_token_position += 1

			if train_y[i] == 1:
				print(terms)
			training_documents.append(terms)
			training_class.append(train_y[i])
			focus_token_position += 1

			if len(training_documents) >= examples:
				break
		
		if len(training_documents) >= examples:
			break

	positive = 0
	negative = 0
	for i in range(test_y.shape[0]):
		focus_token_position = 0
		for focus_token_id in test_x[i]:
			focus_token = id_to_word[focus_token_id]
			if focus_token != target_token:
				focus_token_position += 1
				continue

			terms = []
			context_token_position = 0
			for context_token_id in test_x[i]:
				if np.abs(context_token_position - focus_token_position) <= context_size and context_token_position != focus_token_position:
					terms.append(str(context_token_position - focus_token_position) + ":" + id_to_word[context_token_id].lower())

					#if id_to_word[context_token_id].lower() != target_token:
					#	terms.append(id_to_word[context_token_id].lower())

				context_token_position += 1

			if test_y[i] == 1 and word_to_id[target_token] in test_x[i]:
				positive += 1
				print(terms)

			if test_y[i] == 0 and word_to_id[target_token] in test_x[i]:
				negative += 1
				print("***", terms)

			testing_documents.append(terms)
			testing_class.append(test_y[i])
			focus_token_position += 1

			if len(testing_documents) >= examples:
				break
		
		if len(testing_documents) > examples:
			break

	if positive + negative > 0:
		print(positive/(positive+negative), (positive + negative)/test_y.shape[0])

	_LOGGER.info("Producing self-attention datasets... Done")

	_LOGGER.info("Producing bit representation...")

	vectorizer_X = CountVectorizer(
	    tokenizer=lambda s: s,
	    token_pattern=None,
	    ngram_range=(1, args.max_ngram),
	    lowercase=False,
	    binary=True
	)

	X_train = vectorizer_X.fit_transform(training_documents).toarray().astype(np.uint32)
	feature_names = vectorizer_X.get_feature_names_out()
	Y_train = np.array(training_class).astype(np.uint32)

	print("******", X_train.shape)

	X_test = vectorizer_X.transform(testing_documents).toarray().astype(np.uint32)
	Y_test = np.array(testing_class).astype(np.uint32)

	_LOGGER.info("Producing bit representation... Done!")

	# This is where you create a local perspective for each token (self-attention)

	tm = TMClassifier(1000, 10000, 1.0, weighted_clauses=True, max_included_literals=32)

	tm.fit(X_train, Y_train)
	tm.fit(X_train, Y_train)

	(Y_test_predicted, Y_test_predicted_scores)  = tm.predict(X_test, return_class_sums=True)

	sorted_indexes = np.argsort(-1*Y_test_predicted_scores[:,0])

	print(sorted_indexes.shape)
	print("Score:", Y_test_predicted_scores[sorted_indexes[0:10], 0], Y_test[sorted_indexes[0:10]])
	for k in range(X_test.shape[1]):
		if X_test[sorted_indexes[0], k] == 1:
			print(feature_names[k], end=' ')
	print()

	plot_precision_recall_curve(Y_test_predicted_scores, Y_test)

	print("\tToken: %s AUC: %.2f%% AUC: %.2f%% Precision: %.2f Recall: %.2f Count: %d Precision: %.2f Recall: %.2f Count: %d" % (target_token, 100*roc_auc_score(Y_test, Y_test_predicted_scores[:,0]), 100*roc_auc_score(Y_test, Y_test_predicted_scores[:,1]), 100*precision_score(Y_test, Y_test_predicted), 100*recall_score(Y_test, Y_test_predicted), Y_test.sum(), 100*precision_score(1-Y_test, 1-Y_test_predicted), 100*recall_score(1-Y_test, 1-Y_test_predicted), (1-Y_test).sum()))

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

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print("¬" + feature_names[k - X_train.shape[1]], end=' ')
	print()

	tms.append(tm)