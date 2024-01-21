import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import csv

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine

batches = 100

s = 1.0
T = 10000
clauses = 10000

print("READ")

f = open("/data/yahoo_answers_csv/train.csv", "r")
reader = csv.reader(f, delimiter=',', quotechar='"')
training_documents = []
training_y = []
for document in reader:
	training_documents.append(" ".join(document[1:]))
	training_y.append(int(document[0]))
f.close()

f = open("/data/yahoo_answers_csv/test.csv", "r")
reader = csv.reader(f, delimiter=',', quotechar='"')
testing_documents = []
testing_y = []
for document in reader:
	testing_documents.append(" ".join(document[1:]))
	testing_y.append(int(document[0]))
f.close()

print(len(training_documents))

vectorizer_X = CountVectorizer(binary=True, max_features=10000)

print("VECTORIZE")
X_train = vectorizer_X.fit_transform(training_documents)
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
Y_train = np.array(training_y)

X_test = vectorizer_X.transform(testing_documents)
Y_test = np.array(testing_y)

print("DONE")

epochs = 100

batch_size_train = Y_train.shape[0] // batches

tm = MultiClassTsetlinMachine(clauses, T, s, max_included_literals=32)
for i in range(epochs):
	for batch in range(batches):
		start_training = time()
		tm.fit(X_train[batch*batch_size_train:(batch+1)*batch_size_train], Y_train[batch*batch_size_train:(batch+1)*batch_size_train], epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result_test = 100*(tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		print("#%d Accuracy Test: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result_test, stop_training-start_training, stop_testing-start_testing))