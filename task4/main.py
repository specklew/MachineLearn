import os

import pandas as pd

from classifiers.collaborativefiltering import CollaborativeFilterClassifier
from movie import train, task

is_test = os.environ.get('IS_TEST') == 'true'

if __name__ == '__main__':

    classifier = CollaborativeFilterClassifier(rate=0.005, epochs=3000, feature_space=6, batch_size=len(train), test_count=10, test_divide=0.2)

    if is_test:
        classifier.fit_test_predict(train)
    else:
        predictions = classifier.fit_predict(train, task)
        print("Saving predictions to disk...")
        predictions.to_csv('task4/submission.csv', sep=';', index=False)

