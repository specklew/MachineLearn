import os

from classifiers.similarusers import SimilarUsersClassifier
from movie import train, task

is_test = os.environ.get('IS_TEST') == 'true'

if __name__ == '__main__':

    classifier = SimilarUsersClassifier(batch_size=len(train), test_count=3, test_divide=0.2, min_similar_users=5, cosine_weight=0.2)

    if is_test:
        classifier.fit_test_predict(train)
    else:
        predictions = classifier.fit_predict(train, task)
        print("Saving predictions to disk...")
        predictions.to_csv('task3/submission.csv', sep=';', index=False)
