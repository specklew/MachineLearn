import math
import os
import numpy as np

from classifiers.decisiontree import DecisionTree
from classifiers.decisiontree.randomforest import RandomForestClassifier
from movie import train, task, get_all_possible_features

is_test = os.environ.get('IS_TEST') == 'true'
should_run_rf = os.environ.get('RF') == 'true'


def mode(x: list[int]) -> int:
    vals, counts = np.unique(x, return_counts=True)
    return int(vals[np.argmax(counts)])


if __name__ == '__main__':

    num_features = len(get_all_possible_features())

    print("Starting decision trees with: " + str(num_features) + " features...")

    if should_run_rf:
        classifier = RandomForestClassifier(num_trees=100,
                                            num_features=int(math.sqrt(num_features)),
                                            batch_size=int(len(train)),
                                            test_count=10,
                                            test_divide=0.4,
                                            max_depth=12,
                                            aggregation_function=mode)
        if is_test:
            classifier.fit_test_predict(train)
        else:
            predictions = classifier.fit_predict(train, task)
            predictions.sort_values(predictions.columns[0], ascending=True)

            print("Saving predictions to disk...")
            predictions.to_csv('task2/submission_forest_1.csv', sep=';', index=False)
    else:
        classifier = DecisionTree(batch_size=len(train), test_count=3, test_divide=0.2, max_depth=8)
        if is_test:
            classifier.fit_test_predict(train)
        else:
            predictions = classifier.fit_predict(train, task)
            predictions.to_csv('task2/submission.csv', sep=';', index=False)


