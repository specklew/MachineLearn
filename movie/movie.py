import numpy as np
import pandas as pd
import ast
import os

train = pd.read_csv('dataset/train.csv', sep=';', names=['id', 'user', 'movie', 'rating'])
task = pd.read_csv('dataset/task.csv', sep=';', names=['id', 'user', 'movie', 'rating'])

try:
    movies_features = pd.read_csv('dataset/processed_movies.csv', sep=';')
except Exception:
    movies_features = None

separate_list_features = os.environ.get('SEPARATE_LIST_FEATURES') == 'true'


def get_all_possible_features():
    features = ['popularity', 'year', 'runtime']  # initialize with scalar features

    if separate_list_features:
        all_movies = [(
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
            for _, row in movies_features.iterrows()]

        for movie in all_movies:
            for feature_list in [movie.genres, movie.cast, movie.keywords, movie.similar]:
                for feature in feature_list:
                    if feature not in features:
                        features.append(str(feature))
    else:
        features += ['genres', 'cast', 'keywords', 'similar']

    return set(features)


class Movie:
    def __init__(self,
                 dataset_id,
                 tmdb_id,
                 title,
                 genres: list[str],
                 popularity,
                 year,
                 runtime,
                 cast: list[str],
                 keywords: list[str],
                 similar: list[str]):
        self.id = dataset_id
        self.tmdb_id = tmdb_id
        self.title = title

        if isinstance(genres, str):
            genres = list(ast.literal_eval(genres))
        self.genres: list[str] = genres
        self.popularity = popularity
        self.year = year
        self.runtime = runtime

        if isinstance(cast, str):
            cast = list(ast.literal_eval(cast))
        self.cast: list[str] = cast

        if isinstance(keywords, str):
            keywords = list(ast.literal_eval(keywords))
        self.keywords: list[str] = keywords

        if isinstance(similar, str):
            similar = list(ast.literal_eval(similar))
        self.similar: list[str] = similar

    def get_num_vector(self):
        return (np.
        array([
            self.popularity,
            self.runtime / 60,
            self.year / 1000]))

    def get_other_features(self):
        return [self.genres,
                self.cast,
                self.keywords,
                self.similar]

    def as_dict(self):
        return {
            'id': self.id,
            'tmdb': self.tmdb_id,
            'title': self.title,
            'genres': self.genres,
            'popularity': self.popularity,
            'year': self.year,
            'runtime': self.runtime,
            'cast': self.cast,
            'keywords': self.keywords,
            'similar': self.similar
        }

    def features_as_dict(self, possible_features):
        if separate_list_features:
            features_dict = dict.fromkeys(possible_features, 0)
            features_dict['popularity'] = self.popularity
            features_dict['year'] = self.year
            features_dict['runtime'] = self.runtime

            for feature in self.genres:
                features_dict[feature] = 1
            for feature in self.cast:
                features_dict[feature] = 1
            for feature in self.keywords:
                features_dict[feature] = 1
            for feature in self.similar:
                features_dict[str(feature)] = 1

            return features_dict
        else:
            return {
                'popularity': self.popularity,
                'year': self.year,
                'runtime': self.runtime,
                'genres': self.genres,
                'cast': self.cast,
                'keywords': self.keywords,
                'similar': self.similar
            }

    def __repr__(self):
        return self.as_dict().__repr__()
