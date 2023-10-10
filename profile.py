import pickle
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--target_one", default='hate', type=str)
parser.add_argument("--target_two", default='hate', type=str)

args = parser.parse_args()

feature_names = pickle.load(open("words.p", "rb"))
word_profile = pickle.load(open("word_profile.p", "rb"))

target_id_one = list(feature_names).index(args.target_one)
target_id_two = list(feature_names).index(args.target_two)
target_id_three = list(feature_names).index(args.target_three)

print(feature_names[target_id_one])
print(feature_names[word_profile.getrow(target_id_one).indices][0:25])
print(word_profile.getrow(target_id_one)[0:25])

print(feature_names[target_id_two])
print(feature_names[word_profile.getrow(target_id_two).indices][0:25])
print(word_profile.getrow(target_id_two)[0:25])

joint_score = word_profile.getrow(target_id_one).toarray().reshape(-1) * word_profile.getrow(target_id_two).toarray().reshape(-1) 
#joint_score = word_profile.getrow(target_id_one).toarray().reshape(-1) + word_profile.getrow(target_id_two).toarray().reshape(-1)

print(joint_score.shape)
sorted_ids = np.argsort(-joint_score)
print(feature_names[sorted_ids][0:100])

print("Similarity:", cosine_similarity(word_profile.getrow(target_id_one), word_profile.getrow(target_id_two)))
