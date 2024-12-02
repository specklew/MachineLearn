import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter

from classifiers import Classifier
from classifiers.similarity import cosine


class SimilarUsersClassifier(Classifier):
    def __init__(self, batch_size=1024, test_count=3, test_divide=0.2):
        super().__init__(batch_size, test_count, test_divide)

    @staticmethod
    def _compare_users(user1_task_movie_ratings: list[tuple[int, int]],
                       user2_test_movie_ratings: list[tuple[int, int]],
                       require: int | None = None):

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
                        movie_rating1_rating2 = (movie, rating1, rating2)
                        common.append(movie_rating1_rating2)

            return common

        def calculate_similarity(movie_rating1_rating2: list[tuple[int, int, int]]):
            ratings1 = np.array([rating[1] for rating in movie_rating1_rating2])
            ratings2 = np.array([rating[2] for rating in movie_rating1_rating2])

            return cosine(ratings1, ratings2)  # TODO: implement ways of changing similarity

        common_movies = find_common_movies(user1_task_movie_ratings, user2_test_movie_ratings)

        if len(common_movies) <= 3:
            return float('inf')

        return calculate_similarity(common_movies)

    def _predict_single(self, x: dict[int, [int, int]], considered_user: int, considered_movie: int):

        def get_movies_ratings(user: int):
            return x[user]

        def get_most_similar_users(num_users: int):

            users_similarities: [tuple[int, float]] = []
            user1_task_movies_ratings = x[considered_user]

            for user2 in x.keys():

                if user2 == considered_user:
                    continue

                similarity = self._compare_users(user1_task_movies_ratings, x[user2], considered_movie)
                users_similarities.append((user2, similarity))

            users_similarities = sorted(users_similarities, key=itemgetter(1), reverse=False)

            return users_similarities[:num_users]

        def calculate_rating_from_similar_users(similar_usr: list[tuple[int, float]]):

            ratings = []

            for user in similar_usr:
                user_movies_ratings = get_movies_ratings(user[0])
                for movie_rating in user_movies_ratings:
                    if movie_rating[0] == considered_movie:
                        ratings.append(movie_rating[1])

            vals, counts = np.unique(ratings, return_counts=True)
            index = np.argmax(counts)
            return vals[index]

        similar_users = get_most_similar_users(10)  # TODO: implement ways of changing number of users
        return calculate_rating_from_similar_users(similar_users)

    @staticmethod
    def _get_users_movies_ratings(x: pd.DataFrame) -> dict[int, [int, int]]:

        users_movies_ratings: dict[[int, int]] = {}

        for _, row in x.iterrows():
            user = row['user']
            movie = row['movie']
            rating = row['rating']

            if user not in users_movies_ratings:
                users_movies_ratings[user] = []

            users_movies_ratings[user].append((movie, rating))

        return users_movies_ratings

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:

        predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])

        users = np.unique(x_test['user'])

        x = self._get_users_movies_ratings(x_train)

        for user in tqdm(users):
            user_movie = x_test[x_test['user'] == user]

            for _, row in user_movie.iterrows():
                movie = row['movie']
                rating = self._predict_single(x, user, movie)

                predictions.loc[row['id']] = [
                    int(row['id']),
                    int(user),
                    int(movie),
                    int(rating)]

        return predictions
