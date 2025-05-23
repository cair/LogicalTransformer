from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

number_of_training_examples = 500000
number_of_testing_examples = 10000

noise_p = 0.0

number_of_features = 1000

number_of_characterizing_features = 10 # Each class gets this many unique features in total

number_of_characterizing_features_per_example = 1 # Each example consists of this number of unique features
number_of_common_features_per_example = 10

number_of_clauses = 100
T = number_of_clauses*10
s = 1.0

a = 1.1
b = 2.7

characterizing_features = np.arange(number_of_characterizing_features*2).reshape((2, number_of_characterizing_features)).astype(np.uint32)
common_features = np.setdiff1d(np.arange(number_of_features), characterizing_features.reshape(-1))

p_common_feature = np.empty(common_features.shape[0])
for k in range(common_features.shape[0]):
	p_common_feature[k] = (k + b)**(-a)
p_common_feature = p_common_feature / p_common_feature.sum()

X_train = np.zeros((number_of_training_examples, number_of_features), dtype=np.uint32)
Y_train = np.zeros(number_of_training_examples, dtype=np.uint32)
for i in range(number_of_training_examples):
	x_1 = np.random.choice(2)
	x_2 = np.random.choice(2)
	Y_train[i] = np.logical_xor(x_1, x_2)

	indexes = np.random.choice(characterizing_features[x_1], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		X_train[i, j] = 1

	indexes = np.random.choice(characterizing_features[x_2], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		X_train[i, j] = 1

	indexes = np.random.choice(common_features, number_of_common_features_per_example, replace=False, p=p_common_feature)
	for j in indexes:
		X_train[i, j] = 1

Y_train = np.where(np.random.rand(Y_train.shape[0]) <= noise_p, 1 - Y_train, Y_train)  # Adds noise

X_test = np.zeros((number_of_testing_examples, number_of_features), dtype=np.uint32)
Y_test = np.zeros(number_of_testing_examples, dtype=np.uint32)
for i in range(number_of_testing_examples):
	x_1 = np.random.choice(2)
	x_2 = np.random.choice(2)
	Y_test[i] = np.logical_xor(x_1, x_2)

	indexes = np.random.choice(characterizing_features[x_1], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		X_test[i, j] = 1

	indexes = np.random.choice(characterizing_features[x_2], number_of_characterizing_features_per_example, replace=False)
	for j in indexes:
		X_test[i, j] = 1

	indexes = np.random.choice(common_features, number_of_common_features_per_example, replace=False, p=p_common_feature)
	for j in indexes:
		X_test[i, j] = 1

tm = TMClassifier(number_of_clauses, T, s, platform='CPU', weighted_clauses=True, max_included_literals=10)

start = time()
tm.fit(X_train, Y_train)
stop = time()

print(stop-start)

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
				l.append(" x%d(%d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
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

print("\nLiteral Clause Frequency:", end=' ')
literal_frequency = tm.literal_clause_frequency()
sorted_literals = np.argsort(-1*literal_frequency)
for k in sorted_literals:
		if literal_frequency[k] == 0:
			break

		if k < number_of_features:
			print("%d(%.2f)" % (k, 1.0*literal_frequency[k]/number_of_clauses), end=' ')
		else:
			print("¬%d(%.2f)" % (k - number_of_features, 1.0*literal_frequency[k]/number_of_clauses), end=' ')
print()

for i in range(2):
	print("\nLiteral Importance Class #%d:" % (i), end=' ')

	literal_importance = tm.literal_importance(i, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print(k, end=' ')

	literal_importance = tm.literal_importance(i, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		print("¬" + str(k - number_of_features), end=' ')
	print()

print(characterizing_features, common_features.shape)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
