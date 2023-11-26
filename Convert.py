import pickle
import numpy as np

profile_threshold = 1.0

words = pickle.load(open("words_10.p", "rb"))
word_profile = pickle.load(open("word_profile_10.p", "rb"))
word_profile.data = np.where(word_profile.data >= profile_threshold, word_profile.data, 0)

f = open("Bayesian_profile.vec", "w+")
f.write("%d %d\n" % (word_profile.shape[0], word_profile.shape[1]))
for i in range(word_profile.shape[0]):
  if i % 100 == 0:
    print(i, words[i])

  f.write(words[i] + " ")
  np.savetxt(f, word_profile[[i]].toarray(), delimiter=' ', fmt='%.2f')
f.close()
