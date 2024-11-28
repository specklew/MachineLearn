import os
import pandas as pd
from classifiers.neighbours import KNeighboursClassifier
from movie import train, task

is_test = os.environ['IS_TEST'] == 'true'

if __name__ == '__main__':

    classifier = KNeighboursClassifier(k_neighbours=5, batch_size=len(train), test_count=10, test_divide=0.2)

    if is_test:
        classifier.fit_test_predict(train)
    else:
        predictions = classifier.fit_predict(train, task)
        predictions.to_csv('task1/submission.csv', sep=';', index=False)

