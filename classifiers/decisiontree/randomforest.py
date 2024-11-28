from typing import Callable

import numpy as np
import pandas as pd
import graphviz  # just for visualization

from classifiers.decisiontree import DecisionTree as Tree
from movie import Movie, get_all_possible_features, movies_features
from classifiers import Classifier
from tqdm import tqdm


class _RandomFeatureGenerator:
    _movie_features = get_all_possible_features()

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.selected_features = set(np.random.choice(list(self._movie_features), self.num_features, replace=False))  # no need to subtract 3
        self.selected_features |= {'popularity', 'year', 'runtime'}  # manually add most important features | TODO: find a way to determine them

    def transform(self, movie: dict) -> dict:
        all_features = movie
        selected_features = {key: all_features[key] for key in self.selected_features}
        return selected_features


class RandomForestClassifier(Classifier):
    def __init__(self,
                 num_trees: int,
                 num_features: int,
                 batch_size: int = 1024,
                 test_count: int = 3,
                 test_divide: float = 0.2,
                 aggregation_function: Callable[[list[int]], float] = np.average,
                 max_depth: int = 8,
                 min_samples_split: int = 2,
                 criterion: str = 'gini'):

        super().__init__(batch_size, test_count, test_divide)

        self.trees: list[Tree] = []
        self.num_trees = num_trees
        self.num_features = num_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.aggregation_function = aggregation_function
        self.criterion = criterion
        pass

    def _select_random_features(self, movies: list[dict]) -> list[dict]:
        random_feature_selector = _RandomFeatureGenerator(self.num_features)
        return [random_feature_selector.transform(movie) for movie in movies]

    def _randomize_x_y(self, movies: list[dict], ratings: list[int]):
        num_movies = len(movies)
        movies_ratings = list(zip(movies, ratings))

        randomized_x_ys = []
        random_generator = np.random.default_rng()
        for _ in range(self.num_trees):
            random_movies = random_generator.choice(movies_ratings, num_movies, replace=True, axis=0)
            movies = random_movies[:, 0].tolist()
            ratings = random_movies[:, 1].tolist()
            movies = self._select_random_features(movies)
            randomized_x_y = (movies, ratings)
            randomized_x_ys.append(randomized_x_y)

        return randomized_x_ys

    def _fit_tree(self, random_x_y) -> Tree:
        tree = Tree(self.max_depth, self.min_samples_split, criterion=self.criterion)
        movies, ratings = random_x_y
        tree.fit(movies, ratings)
        return tree

    def fit(self, movies: list[dict], ratings: list[int]):
        print("Fitting random forest...")
        trees = self._randomize_x_y(movies, ratings)
        self.trees = list(map(self._fit_tree, tqdm(trees)))
        pass

    def _aggregate_predictions(self, predictions: list[int]) -> int:
        result = int(round(self.aggregation_function(predictions), 0))
        return result

    def _predict_single(self, movie: Movie) -> int:
        predictions = list(map(lambda tree: tree.predict([movie]), self.trees))
        return self._aggregate_predictions(predictions)

    def predict(self, movies: list[Movie]) -> list[int]:
        print("Predicting random forest...")
        return list(map(self._predict_single, tqdm(movies)))

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        possible_features = get_all_possible_features()

        predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])

        users_movies_train = [(
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
            if row['id'] in x_train['movie'].values]

        users_movies_test = [(
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
            if row['id'] in x_test['movie'].values]

        labels = x_train['rating']

        movies_watched = [movie
                          for movie in users_movies_train
                          if movie.id in x_train['movie'].values]

        features = [movie.features_as_dict(possible_features) for movie in movies_watched]

        self.fit(features, labels.to_list())

        movies_to_rate = [movie
                          for movie in users_movies_test
                          if movie.id in x_test['movie'].values]

        features = [movie.features_as_dict(possible_features) for movie in movies_to_rate]

        user_predictions = self.predict(features)

        for i in range(len(user_predictions)):
            predictions.loc[len(predictions.index)] = {
                'id': int(x_test.iloc[i]['id']),
                'user': int(x_test.iloc[i]['user']),
                'movie': int(x_test.iloc[i]['movie']),
                'rating': int(user_predictions[i])
            }
        return predictions

    def get_plot(self):
        dot = graphviz.Digraph(comment='Random Forest')

        for i, tree in enumerate(self.trees):
            dot.subgraph(tree.get_plot_tree_for(f'Tree {i}'))

        dot.render('random_forest.gv', view=True)

