import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit
from glob import glob
from scipy.sparse import csc_matrix, csr_matrix, dok_array

relevance_threshold = 0.5
profile_threshold = 2.5

@jit(nopython=True)
def count_tokens(X_indices, X_indptr, word_profile_data, word_profile_indices, word_profile_indptr, feature_map):
    document_vector = np.zeros(word_profile_indptr.shape[0]-1)
    target_word_profile = np.zeros(word_profile_indptr.shape[0]-1)
    target_word_refined_profile = np.zeros(word_profile_indptr.shape[0]-1)
    other_word_profile = np.zeros(word_profile_indptr.shape[0]-1)

    global_token_count = 0

    # Iterate over documents
    for row in range(X_indptr.shape[0]-1):
        document_vector[:] = 0 # Initializes document vector

        # Iterate over each word in the document
        for i in range(X_indptr[row], X_indptr[row+1]):
            # Skip word if not in feature map (i.e., no profile exists for that word). Alternatively, the word can be included directly instead.
            if feature_map[X_indices[i]] == 0:
                continue 

            # Initialize target word profile and refined profile.
            # The purpose of the refined profile is to narrow down the profile by multiplying in the profiles of related words in the document.
            target_word_profile[:] = 0 # For storing a raw copy of the target word profile
            target_word_refined_profile[:] = 0 # For storing the refined target word profile

            # First, the profile words of the target word is added to the refined profile
            for k in range(word_profile_indptr[feature_map[X_indices[i]]], word_profile_indptr[feature_map[X_indices[i]]+1]):
                target_word_profile[word_profile_indices[k]] = word_profile_data[k]
                target_word_refined_profile[word_profile_indices[k]] = word_profile_data[k]

            # Next, the profile of each other word in the document is inspected
            for j in range(X_indptr[row], X_indptr[row+1]):
                if feature_map[X_indices[j]] == 0 or i == j:
                    continue

                # Words that are related to the target word are processed further
                if target_word_profile[feature_map[X_indices[j]]] >= relevance_threshold:
                    # The other word profile is created first
                    other_word_profile[:] = 0
                    for k in range(word_profile_indptr[feature_map[X_indices[j]]], word_profile_indptr[feature_map[X_indices[j]]+1]):
                        other_word_profile[word_profile_indices[k]] = word_profile_data[k]

                    # The other word profile is multiplied into the refined target word profile
                    target_word_refined_profile = target_word_refined_profile * other_word_profile

            for k in range(target_word_refined_profile.shape[0]):
                if target_word_refined_profile[k] < profile_threshold:
                    target_word_refined_profile[k] = 0

            # The refiend profile is added to the document vector
            document_vector += target_word_refined_profile

        document_token_count = 0
        for i in range(document_vector.shape[0]):
            if document_vector[i] > 0:
                document_token_count += 1
                global_token_count += 1
        
        print(row, X_indptr.shape[0]-1, document_token_count)

        #print(row, X_indptr[row], X_indptr[row+1], X_indptr[row+1] - X_indptr[row], document_token_count)

    print(global_token_count)
    return global_token_count

@jit(nopython=True)
def embed_X(X_indices, X_indptr, word_profile_data, word_profile_indices, word_profile_indptr, feature_map, token_count):
    document_vector = np.zeros(word_profile_indptr.shape[0]-1)
    target_word_profile = np.zeros(word_profile_indptr.shape[0]-1)
    target_word_refined_profile = np.zeros(word_profile_indptr.shape[0]-1)
    other_word_profile = np.zeros(word_profile_indptr.shape[0]-1)

    X_embedded_data = np.empty(token_count, dtype=np.uint32)
    X_embedded_indices = np.empty(token_count, dtype=np.uint32)
    X_embedded_indptr = np.empty(X_indptr.shape, dtype=np.uint32)

    global_token_count = 0

    # Iterate over documents
    for row in range(X_indptr.shape[0]-1):
        document_vector[:] = 0 # Initializes document vector

        # Iterate over each word in the document
        for i in range(X_indptr[row], X_indptr[row+1]):
            # Skip word if not in feature map (i.e., no profile exists for that word). Alternatively, the word can be included directly instead.
            if feature_map[X_indices[i]] == 0:
                continue 

            # Initialize target word profile and refined profile.
            # The purpose of the refined profile is to narrow down the profile by multiplying in the profiles of related words in the document.
            target_word_profile[:] = 0 # For storing a raw copy of the target word profile
            target_word_refined_profile[:] = 0 # For storing the refined target word profile

            # First, the profile words of the target word is added to the refined profile
            for k in range(word_profile_indptr[feature_map[X_indices[i]]], word_profile_indptr[feature_map[X_indices[i]]+1]):
                target_word_profile[word_profile_indices[k]] = word_profile_data[k]
                target_word_refined_profile[word_profile_indices[k]] = word_profile_data[k]

            # Next, the profile of each other word in the document is inspected
            for j in range(X_indptr[row], X_indptr[row+1]):
                if feature_map[X_indices[j]] == 0 or i == j:
                    continue

                # Words that are related to the target word are processed further
                if target_word_profile[feature_map[X_indices[j]]] >= relevance_threshold:
                    # The other word profile is created first
                    other_word_profile[:] = 0
                    for k in range(word_profile_indptr[feature_map[X_indices[j]]], word_profile_indptr[feature_map[X_indices[j]]+1]):
                        other_word_profile[word_profile_indices[k]] = word_profile_data[k]

                    # The other word profile is multiplied into the refined target word profile
                    target_word_refined_profile = target_word_refined_profile * other_word_profile

            for k in range(target_word_refined_profile.shape[0]):
                if target_word_refined_profile[k] < profile_threshold:
                    target_word_refined_profile[k] = 0

            # The refiend profile is added to the document vector
            document_vector += target_word_refined_profile

        X_embedded_indptr[row] = global_token_count
        document_token_count = 0
        for i in range(document_vector.shape[0]):
            if document_vector[i] > 0:
                X_embedded_data[global_token_count] = 1
                X_embedded_indices[global_token_count] = i
                document_token_count += 1
                global_token_count += 1
        X_embedded_indptr[row+1] = global_token_count

        print(row, X_indptr.shape[0]-1, document_token_count)

        #print(row, X_indptr[row], X_indptr[row+1], X_indptr[row+1] - X_indptr[row], document_token_count)
    return (X_embedded_data, X_embedded_indices, X_embedded_indptr)

_LOGGER = logging.getLogger(__name__)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_ngram", default=1, type=int)
    parser.add_argument("--num_clauses", default=10000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=2.0, type=float)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--clause_drop_p", default=0.75, type=float)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--reuters-num-words", default=10000, type=int)
    parser.add_argument("--reuters-index-from", default=2, type=int)
    args = parser.parse_args()


    _LOGGER.info("Preparing dataset")
    train, test = keras.datasets.reuters.load_data(num_words=args.reuters_num_words, index_from=args.reuters_index_from)
    train_x, train_y = train
    test_x, test_y = test

    word_to_id = keras.datasets.reuters.get_word_index()
    word_to_id = {k: (v + args.reuters_index_from) for k, v in word_to_id.items()}
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
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)

    words = pickle.load(open("words.p", "rb"))
    word_profile = pickle.load(open("word_profile.p", "rb"))
   
 
    # Creates mapping of word to word id for the profiles
    word_to_id = {}
    for i in range(len(words)):
        word_to_id[words[i]] = i

    # Creates mapping from text id to profile id.
    feature_names_out = vectorizer_X.get_feature_names_out()
    feature_map = np.empty(feature_names_out.shape[0], dtype=np.uint32)
    for i in range(feature_names_out.shape[0]):
        if feature_names_out[i] in word_to_id:
            feature_map[i] = word_to_id[feature_names_out[i]]
        else:
            feature_map[i] = 0

    # Counts number of tokens in the augmented dataset to allocate memory for sparse data structure
    token_count = count_tokens(X_train.indices, X_train.indptr, word_profile.data, word_profile.indices, word_profile.indptr, feature_map)
    (X_train_embedded_data, X_train_embedded_indices, X_train_embedded_indptr) = embed_X(X_train.indices, X_train.indptr, word_profile.data, word_profile.indices, word_profile.indptr, feature_map, token_count)
    X_train_embedded = csr_matrix((X_train_embedded_data, X_train_embedded_indices, X_train_embedded_indptr))

    _LOGGER.info("Producing bit representation... Done!")

    _LOGGER.info("Selecting Features....")

    SKB = SelectKBest(chi2, k=args.features)
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train).toarray()
    X_test = SKB.transform(X_test).toarray()

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
