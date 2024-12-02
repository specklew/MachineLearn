import numpy as np


def calculate(a, b):

    # assuming that both movies have the same number of other features
    a_num = a.get_num_vector()
    b_num = b.get_num_vector()
    num_similarity = cosine(a_num, b_num)

    a_misc = a.get_other_features()
    b_misc = b.get_other_features()
    misc_similarity = 0

    for i in range(len(a_misc)):
        misc_similarity += jaccard(set(a_misc[i]), set(b_misc[i]))

    total_features = len(a_num) + len(a_misc)
    num_multiplier = len(a_num) / total_features
    misc_multiplier = 1 / total_features  # 1 / total - because misc_similarity stores the sum of similarities

    return num_similarity * num_multiplier + misc_similarity * misc_multiplier


def cosine(u, v):
    dot = np.dot(u, v)
    len_u = np.linalg.norm(u)
    len_v = np.linalg.norm(v)

    if len_u == 0 or len_v == 0:
        return 0

    return dot / (len_u * len_v)


def jaccard(u, v):

    # ref: https://www.geeksforgeeks.org/how-to-calculate-jaccard-similarity-in-python/
    intersection = len(u.intersection(v))
    union = len(u.union(v))

    if union == 0:
        return 0

    return intersection / union

