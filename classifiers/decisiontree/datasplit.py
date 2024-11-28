import numpy as np
from collections.abc import Callable
from decisiontree.predicates import find_predicate_for_threshold


def _gini_impurity(y) -> float:
    probs = np.bincount(y) / len(y)
    return 1 - np.dot(probs, probs)


def _entropy(y) -> float:
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


gain_function = _gini_impurity


def setup_gain_function(gain: str):
    global gain_function
    if gain == 'gini':
        gain_function = _gini_impurity
    elif gain == 'entropy':
        gain_function = _entropy
    else:
        raise ValueError(f"Unknown gain function: {gain}")


def _calculate_gain(y) -> float:
    return gain_function(y)


def split_database(x, y, feature, threshold):

    predicate = find_predicate_for_threshold(threshold)

    x_left, y_left, x_right, y_right = [], [], [], []

    for movie, label in zip(x, y):
        if predicate(movie[feature], threshold):
            x_left.append(movie)
            y_left.append(label)
        else:
            x_right.append(movie)
            y_right.append(label)

    return x_left, x_right, y_left, y_right


def find_best_split(x: dict, y):
    best_gain = float('inf')
    best_feature = None
    best_threshold = None

    for feature in x[0].keys():
        thresholds = {tuple(movie[feature]) if isinstance(movie[feature], list) else movie[feature] for movie in x}

        for threshold in thresholds:
            _, _, left_y, right_y = split_database(x, y, feature, threshold)

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_gain = _calculate_gain(left_y)
            right_gain = _calculate_gain(right_y)

            gain = (len(left_y) * left_gain + len(right_y) * right_gain) / len(y)
            if gain < best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

            if best_gain == 0:
                return best_feature, best_threshold, best_gain

    return best_feature, best_threshold, best_gain
