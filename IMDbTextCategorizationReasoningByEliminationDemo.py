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

    tm = TMClassifier(args.num_clauses, args.T, args.s, platform=args.device, weighted_clauses=args.weighted_clauses,
                      clause_drop_p=args.clause_drop_p)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train)

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == Y_test).mean()

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

print("\nClass 0 Positive Clauses:\n")

precision = tm.clause_precision(0, 0, X_test, Y_test)
recall = tm.clause_recall(0, 0, X_test, Y_test)

for j in range(number_of_clauses//2):
    print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 0, j), precision[j], recall[j]), end=' ')
    l = []
    for k in range(number_of_features*2):
        if tm.get_ta_action(j, k, the_class = 0, polarity = 0):
            if k < number_of_features:
                l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 0, polarity = 0)))
            else:
                l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 0, polarity = 0)))
    print(" ∧ ".join(l))

    print("\nClass 0 Negative Clauses:\n")

    precision = tm.clause_precision(0, 1, X_test, Y_test)
    recall = tm.clause_recall(0, 1, X_test, Y_test)

    for j in range(number_of_clauses//2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(0, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(number_of_features*2):
            if tm.get_ta_action(j, k, the_class = 0, polarity = 1):
                if k < number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
                else:
                    l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 0, polarity = 1)))
        print(" ∧ ".join(l))

    print("\nClass 1 Positive Clauses:\n")

    precision = tm.clause_precision(1, 0, X_test, Y_test)
    recall = tm.clause_recall(1, 0, X_test, Y_test)

    for j in range(number_of_clauses//2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(number_of_features*2):
            if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
                if k < number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
                else:
                    l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
        print(" ∧ ".join(l))

    print("\nClass 1 Negative Clauses:\n")

    precision = tm.clause_precision(1, 1, X_test, Y_test)
    recall = tm.clause_recall(1, 1, X_test, Y_test)

    for j in range(number_of_clauses//2):
        print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 1, j), precision[j], recall[j]), end=' ')
        l = []
        for k in range(number_of_features*2):
            if tm.get_ta_action(j, k, the_class = 1, polarity = 1):
                if k < number_of_features:
                    l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
                else:
                    l.append("¬x%d(%d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 1)))
        print(" ∧ ".join(l))


    print("\nPositive Polarity:", end=' ')
    literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=False).astype(np.int32)
    sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
    for k in sorted_literals:
        if literal_importance[k] == 0:
            break

        print(feature_names[selected_features[k]], end=' ')

    literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=False).astype(np.int32)
    sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
    for k in sorted_literals:
        if literal_importance[k] == 0:
            break

        print("¬'" + feature_names[selected_features[k - args.features]] + "'", end=' ')

    print()
    print("\nNegative Polarity:", end=' ')
    literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=True).astype(np.int32)
    sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
    for k in sorted_literals:
        if literal_importance[k] == 0:
            break

        print(feature_names[selected_features[k]], end=' ')

    literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
    sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
    for k in sorted_literals:
        if literal_importance[k] == 0:
            break

        print("¬'" + feature_names[selected_features[k - args.features]] + "'", end=' ')
    print()
