import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from time import time

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

profile_size = 50

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=1000, type=int)
    parser.add_argument("--T", default=10000, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--clause_drop_p", default=0.0, type=float)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--features", default=10000, type=int)
    parser.add_argument("--imdb-num-words", default=10000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    args = parser.parse_args()

    _LOGGER.info("Preparing dataset")
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
        binary=True,
        min_df=5
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    feature_names = vectorizer_X.get_feature_names_out()
    Y_train = train_y.astype(np.uint32)

    print(X_train.shape)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    X_train = X_train.astype(np.uint32)
    X_test = X_test.astype(np.uint32)

    tm = TMClassifier(args.num_clauses, args.T, args.s, platform='CPU_sparse', weighted_clauses=args.weighted_clauses, absorbing=100, literal_insertion_state=127, literal_sampling=0.05)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        training_start = time()
        tm.fit(X_train, Y_train)
        training_stop = time()

        absorbed = 0.0
        unallocated = 0
        for i in range(2):
                for j in range(args.num_clauses):
                        absorbed += 1.0 - (tm.number_of_include_actions(i, j) + tm.number_of_exclude_actions(i, j)) / (X_train.shape[1]*2)
                        unallocated += tm.number_of_unallocated_literals(i, j)
        absorbed = 100 * absorbed / (2*args.num_clauses)

        testing_start = time()
        result = 100 * (tm.predict(X_test) == Y_test).mean()
        testing_stop = time()

        print("Accuracy: %.2f%% Absorbed: %.2f%% Unallocated: %d Training time: %.1f Testing time: %.1f" % (result, absorbed, unallocated, training_stop-training_start, testing_stop-testing_start))