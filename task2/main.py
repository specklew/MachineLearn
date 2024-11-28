import math
import os
import cProfile

from classifiers.decisiontree import DecisionTree
from classifiers.decisiontree.randomforest import RandomForestClassifier
from movie import train, task, get_all_possible_features

is_test = os.environ.get('IS_TEST') == 'true'
should_run_rf = os.environ.get('RF') == 'true'

if __name__ == '__main__':

    num_features = len(get_all_possible_features())

    print("Starting decision trees with: " + str(num_features) + " features...")

    if should_run_rf:
        classifier = RandomForestClassifier(num_trees=100, num_features=int(math.sqrt(num_features)), batch_size=int(len(train) * 0.5), test_count=10, test_divide=0.2, max_depth=12)
        if is_test:
            classifier.fit_test_predict(train)
        else:
            predictions = classifier.fit_predict(train, task)
            predictions.to_csv('task2/submission_forest.csv', sep=';', index=False)
    else:
        classifier = DecisionTree(batch_size=len(train), test_count=3, test_divide=0.2, max_depth=8)
        if is_test:
            classifier.fit_test_predict(train)
        else:
            predictions = classifier.fit_predict(train, task)
            predictions.to_csv('task2/submission.csv', sep=';', index=False)


