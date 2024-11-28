import classifiers.neighbours.similarity
import pandas as pd

from classifiers import Classifier
from movie import Movie, movies_features
from tqdm import tqdm



class KNeighboursClassifier(Classifier):
    def __init__(self, k_neighbours, batch_size=1024, test_count=3, test_divide=0.2):

        super().__init__(batch_size, test_count, test_divide)
        self.k_neighbours = k_neighbours

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])

        for _, row in tqdm(x_test.iterrows(), total=len(x_test)):
            user_id = row['user']
            tmdb_id = row['movie']
            predicted_rating = self.predict(user_id, tmdb_id, x_train)
            predictions.loc[len(predictions.index)] = [
                int(row['id']),
                int(row['user']),
                int(row['movie']),
                int(predicted_rating)]

        return predictions

    def predict(self, user_id: int, tmdb_id: int, train_set: pd.DataFrame) -> int:

        user_movie = train_set[train_set['user'] == user_id]

        considered_movie_raw = movies_features[movies_features['id'] == tmdb_id].iloc[0]
        considered_movie = Movie(considered_movie_raw['id'],
                                 considered_movie_raw['tmdb'],
                                 considered_movie_raw['title'],
                                 considered_movie_raw['genres'],
                                 considered_movie_raw['popularity'],
                                 considered_movie_raw['year'],
                                 considered_movie_raw['runtime'],
                                 considered_movie_raw['cast'],
                                 considered_movie_raw['keywords'],
                                 considered_movie_raw['similar'])

        user_movies = train_set[train_set['user'] == user_id]

        movies_watched = [(
            Movie(row['id'],
                  row['tmdb'],
                  row['title'],
                  row['genres'],
                  row['popularity'],
                  row['year'],
                  row['runtime'],
                  row['cast'],
                  row['keywords'],
                  row['similar']))
            for _, row in movies_features.iterrows()
            if row['id'] in user_movies['movie'].values]

        movies_ratings = {}

        for movie in movies_watched:
            rating = user_movies[user_movies['movie'] == movie.id]['rating'].values[0]
            movies_ratings[movie] = rating

        predicted_rating = self.fit_predict_single(considered_movie, movies_watched, movies_ratings)
        return predicted_rating

    def fit_predict_single(self, considered_movie: Movie, watched_movies: [], movie_rating: {}) -> int:

        movie_distance = {}

        for movie in watched_movies:
            movie_distance[movie] = similarity.calculate(considered_movie, movie)

        sorted_movies = [movie for movie, _ in sorted(movie_distance.items(), key=lambda item: item[1])]

        rating_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for neighbour in sorted_movies[-self.k_neighbours:]:
            rating_num[movie_rating[neighbour]] += 1

        for rating, num in rating_num.items():
            if num == max(rating_num.values()):
                return rating

