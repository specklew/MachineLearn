import pandas as pd
import numpy as np

from classifiers import Classifier


def clamp(minimal, maximal, value):
    return max(minimal, min(maximal, value))


class CollaborativeFilterClassifier(Classifier):

    def __init__(self, rate: float, epochs: int, feature_space: int, alternate: bool = False, batch_size: int = 1024, test_count: int = 3,
                 test_divide: float = 0.2):

        super().__init__(batch_size, test_count, test_divide)

        print("\nOPTIONS:")
        print("Rate: ", rate)
        print("Epochs: ", epochs)
        print("Feature space: ", feature_space)
        print("Alternate: ", alternate)

        self.rate = rate
        self.epochs = epochs
        self.feature_space = feature_space
        self.alternate = alternate
        self.switch = False

    def _init_train(self, train: pd.DataFrame):
        mov_usr_rating = train.pivot(index='movie', columns='user', values='rating')

        self.user_index_dict = {user: i for i, user in enumerate(mov_usr_rating.columns)}

        self.ratings = mov_usr_rating.fillna(0).to_numpy()

        self.features = np.random.rand(mov_usr_rating.shape[0], self.feature_space)  # columns are features, rows are movies
        self.parameters = np.random.rand(self.feature_space + 1, mov_usr_rating.shape[1])

    def _predict(self, user: int, movie: int) -> int:
        return clamp(0, 5, round(self.parameters[0, user] + np.dot(self.features[movie], self.parameters[1:, user])))

    def _get_user_index(self, user: int) -> int:
        return self.user_index_dict[user]

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:

        self.fit(x_train)

        predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])

        for _, row in x_test.iterrows():
            user = row['user']
            movie = row['movie']
            predicted_rating = self._predict(self._get_user_index(user), int(movie - 1))
            predictions.loc[len(predictions.index)] = [
                int(row['id']),
                int(row['user']),
                int(row['movie']),
                int(predicted_rating)]

        return predictions

    def fit(self, x_train: pd.DataFrame):
        self._init_train(x_train)

        for epoch in range(self.epochs):
            descent_p0, descent_p, descent_x = self._cal_descent(0.1)
            self.parameters[0] -= self.rate * descent_p0
            self.parameters[1:] -= self.rate * descent_p
            self.features -= self.rate * descent_x

    def _cal_descent(self, regularization: float) -> (np.ndarray, np.ndarray, np.ndarray):
        f_xm_pr = self._cal_all_f_xm_pr()

        error = np.where(self.ratings != 0, f_xm_pr - self.ratings, 0)

        descent_p0 = np.sum(error, axis=0).T
        descent_p = (error.T @ self.features).T
        descent_x = error @ self.parameters[1:].T

        descent_p += regularization * self.parameters[1:]
        descent_x += regularization * self.features

        if self.alternate:
            self.switch = not self.switch

            if self.switch:
                return descent_p0, descent_p, self.features
            else:
                return self.parameters[0], self.parameters[1:], descent_x

        return descent_p0, descent_p, descent_x

    def _cal_all_f_xm_pr(self) -> np.ndarray:
        matmul = self.features @ self.parameters[1:]  # omit bias
        return self.parameters[0] + matmul  # add bias
