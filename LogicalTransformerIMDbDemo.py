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
from sklearn.metrics import recall_score, precision_score, accuracy_score

examples = 5000
positive_sample_p = 0.1

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
parser.add_argument("--imdb-num-words", default=5000, type=int)
parser.add_argument("--imdb-index-from", default=2, type=int)
args = parser.parse_args()

_LOGGER.info("Preparing dataset")
logging.basicConfig(level=logging.INFO)

train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
train_x, train_y = train
test_x, test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
_LOGGER.info("Preparing dataset.... Done!")

_LOGGER.info("Producing bit representation...")

id_to_word = {value: key for key, value in word_to_id.items()}

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

vectorizer_X = CountVectorizer(
    tokenizer=lambda s: s,
    token_pattern=None,
    ngram_range=(1, args.max_ngram),
    lowercase=False,
    binary=True
)

X_train = vectorizer_X.fit_transform(training_documents)
feature_names = vectorizer_X.get_feature_names_out()
Y_train = train_y.astype(np.uint32)

X_test = vectorizer_X.transform(testing_documents)
Y_test = test_y.astype(np.uint32)
_LOGGER.info("Producing bit representation... Done!")

_LOGGER.info("Selecting Features....")

SKB = SelectKBest(chi2, k=args.features)
SKB.fit(X_train, Y_train)

selected_features = SKB.get_support(indices=True)
X_train = SKB.transform(X_train).toarray().astype(np.uint32)
X_test = SKB.transform(X_test).toarray().astype(np.uint32)

_LOGGER.info("Selecting Features.... Done!")

# Train one TM per token (feature) per class, producing a class specific token model

tms = {}
for i in range(2):
	tms[i] = []
	print("\nTraining token models for class", i, "\n")
	for j in range(args.features):
		# This is where you create a local perspective for each token (self-attention)

		tm = TMClassifier(10, 100, 1.1)

		# Extract prediction target from column 'j' in X_train, only looking at training examples from class 'i'
		Y_train_token = X_train[:,j][Y_train==i].reshape(-1)

		# Remove prediction target from column 'j' in X_train
		cols = np.arange(X_train.shape[1]) != j
		X_train_token = X_train[:,cols][Y_train==i]

		X_train_token_0 = X_train_token[Y_train_token==0]
		X_train_token_1 = X_train_token[Y_train_token==1]

		X_train_token_balanced = np.zeros((examples, X_train_token.shape[1]), dtype=np.uint32)
		Y_train_token_balanced = np.zeros(examples, dtype=np.uint32)
		for epoch in range(1):	
			for k in range(examples):
				if np.random.rand() <= positive_sample_p:
					X_train_token_balanced[k,:] = X_train_token_1[np.random.randint(X_train_token_1.shape[0]),:]
					Y_train_token_balanced[k] = 1
				else:
					X_train_token_balanced[k,:] = X_train_token_0[np.random.randint(X_train_token_0.shape[0]),:]
					Y_train_token_balanced[k] = 0

			tm.fit(X_train_token_balanced, Y_train_token_balanced)

		# Create test data for token prediction
		Y_test_token = X_test[:,j][Y_test==i].reshape(-1)
		cols = np.arange(X_test.shape[1]) != j
		X_test_token = X_test[:,cols][Y_test==i]

		X_test_token_0 = X_test_token[Y_test_token==0]
		X_test_token_1 = X_test_token[Y_test_token==1]

		if (X_test_token_0.shape[0] == 0 or X_test_token_1.shape[0] == 0):
			continue

		X_test_token_balanced = np.zeros((examples, X_test_token.shape[1]), dtype=np.uint32)
		Y_test_token_balanced = np.zeros(examples, dtype=np.uint32)
		for k in range(examples):
			if (np.random.rand() <= positive_sample_p):
				X_test_token_balanced[k,:] = X_test_token_1[np.random.randint(X_test_token_1.shape[0]),:]
				Y_test_token_balanced[k] = 1
			else:
				X_test_token_balanced[k,:] = X_test_token_0[np.random.randint(X_test_token_0.shape[0]),:]
				Y_test_token_balanced[k] = 0

		Y_test_token_balanced_predicted = tm.predict(X_test_token_balanced)

		print("\tToken: %s Accuracy: %.2f%% Precision: %.2f Recall: %.2f" % (feature_names[selected_features[j]], 100*accuracy_score(Y_test_token_balanced, Y_test_token_balanced_predicted), 100*precision_score(Y_test_token_balanced, Y_test_token_balanced_predicted), 100*recall_score(Y_test_token_balanced, Y_test_token_balanced_predicted)))

		# Store Tsetlin machine for token 'j' of class 'i'
		tms[i].append(tm)

# Perform composite classification using the individual Tsetlin machines

class_sums = np.zeros((2, Y_test.shape[0]))

# Calculate composite clause sum per class
for i in range(2):
	# Add opp clause sum scores from each token Tsetlin machine
	for j in range(args.features):
		# Create input to token model
		cols = np.arange(X_test.shape[1]) != j
		X_test_token = X_test[:,cols]

		print(X_test_token.shape)

		Y_test_predicted_token, Y_test_scores_token = tms[i][j].predict(X_test_token, return_class_sums=True)

		# Measure ability of token model to match the example. The class with the most fitting token model gets the highest score.
		Y_test_scores_token_combined = np.where(X_test[:,j] == 1, Y_test_scores_token[:,1] - Y_test_scores_token[:,0], Y_test_scores_token[:,0] - Y_test_scores_token[:,1])
		#Y_test_scores_token_combined = np.where(X_test[:,j] == 1, Y_test_scores_token[:,1] - Y_test_scores_token[:,0], 0)

		class_sums[i] += Y_test_scores_token_combined

# The class with the largest composite class sum wins
Y_test_predicted = class_sums.argmax(axis=0)

print("\nClass prediction accuracy:", 100*(Y_test_predicted == Y_test).mean())