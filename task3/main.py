import os

from classifiers.similarusers import SimilarUsersClassifier
from movie import train

is_test = os.environ.get('IS_TEST') == 'true'

if __name__ == '__main__':

    classifier = SimilarUsersClassifier(batch_size=len(train), test_count=3, test_divide=0.2)

    if is_test:
        classifier.fit_test_predict(train)
