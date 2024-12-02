import pandas as pd
import numpy as np

from classifiers import Classifier


class SimilarUsersClassifier(Classifier):
    def __init__(self, batch_size=1024, test_count=3, test_divide=0.2):
        super().__init__(batch_size, test_count, test_divide)

    @staticmethod
    def _compare_users(user1_task_movie_ratings: list[tuple[int, int]],
                       user2_test_movie_ratings: list[tuple[int, int]],
                       require : int | None = None):

        if require is not None:
            user2_movies = [movie[0] for movie in user2_test_movie_ratings]
            if require not in user2_movies:
                return float('inf')

        def find_common_movies(user1_movies_ratings, user2_movies_ratings):

            common: list[tuple[int, int, int]] = []

            for movie_1_i in range(len(user1_movies_ratings)):
                for movie_2_i in range(len(user2_movies_ratings)):

                    if user1_movies_ratings[movie_1_i][0] == user2_movies_ratings[movie_2_i][0]:
                        movie = user1_movies_ratings[movie_1_i][0]
                        rating1 = user1_movies_ratings[movie_1_i][1]
                        rating2 = user2_movies_ratings[movie_2_i][1]
                        common.append((movie, rating1, rating2))

            return common

        def calculate_similarity(movie_rating1_rating2):
            ratings1 = np.array([rating[1] for rating in user1_task_movie_ratings])
            ratings2 = np.array([rating[2] for rating in user1_task_movie_ratings])

            return np.dot(ratings1, ratings2)

        common_movies = find_common_movies(user1_task_movie_ratings, user2_test_movie_ratings)
        return calculate_similarity(common_movies)


    def _predict_single(self, x: list[dict], y: list[int], considered_user: int, considered_movie: int):

        unique_users = np.unique(np.array([row['user'] for row in x]))

        def get_movies_ratings(user: int):
            user_movies = [row for row in x if row['user'] == user]

            user_movies_ratings = []
            for user_movie in user_movies:
                movie = user_movie['movie']
                rating = y[int(user_movie['id'])]

                user_movies_ratings.append((movie, rating))

            return user_movies_ratings

        def get_most_similar_users(num_users: int):

            users_similarities = [tuple[int, float]]
            user1_task_movies_ratings = get_movies_ratings(considered_user)

            for user in unique_users:

                user2_train_movies_ratings = get_movies_ratings(user)
                similarity = self._compare_users(user1_task_movies_ratings, user2_train_movies_ratings, considered_movie)
                users_similarities.append((user, similarity))

            users_similarities.sort(key=lambda row: row[1], reverse=True)

            return users_similarities[:num_users]

        # TODO: calculate prediction for single user movie


    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        pass  # TODO: implement
