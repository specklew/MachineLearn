import numpy as np
import pandas as pd
import numbers

from classifiers import Classifier
from decisiontree.datasplit import find_best_split, split_database, setup_gain_function
from collections import Counter
from tqdm import tqdm
from movie import movies_features, Movie, get_all_possible_features
from decisiontree.predicates import find_predicate_for_threshold
import graphviz  # just for visualization


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature = feature_index
        self.threshold = threshold
        self.left: Node = left
        self.right: Node = right
        self.info_gain = info_gain

        # Only for leaf nodes
        self.value = value

    def get_label(self):
        feature = "Feature: " + str(self.feature) + "\n" if self.feature is not None else ''
        threshold = "Threshold: " + str(self.threshold) + "\n" if self.threshold is not None else ''
        gain = "Gain: " + str(self.info_gain) + "\n" if self.info_gain is not None else ''
        value = "Value: " + str(self.value) + "\n" if self.value is not None else ''

        return f"{feature}{threshold}{gain}{value}"


class DecisionTree(Classifier):
    def __init__(self, max_depth=8, min_samples_split=2, batch_size=1024, test_count=3, test_divide=0.2,
                 criterion='gini'):
        super().__init__(batch_size=batch_size, test_count=test_count, test_divide=test_divide)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        setup_gain_function(criterion)
        self.tree: Node | None = None

    def _build_tree(self, features, labels, depth=0):
        if len(np.unique(labels)) <= 1 or len(labels) < self.min_samples_split or depth >= self.max_depth:
            return Node(value=Counter(labels).most_common(1)[0][0])

        feature, threshold, gain = find_best_split(features, labels)
        if feature is None:
            return Node(value=Counter(labels).most_common(1)[0][0])

        left_x, right_x, left_y, right_y = split_database(features, labels, feature, threshold)

        return Node(
            feature,
            threshold,
            self._build_tree(left_x, left_y, depth + 1),
            self._build_tree(right_x, right_y, depth + 1),
            gain)

    def fit(self, features, labels):
        self.tree = self._build_tree(features, labels)

    def _predict(self, features, tree: Node):
        if isinstance(tree.value, numbers.Number):
            return tree

        feature = tree.feature
        threshold = tree.threshold

        predicate = find_predicate_for_threshold(threshold)

        if predicate(features[feature], threshold):
            return self._predict(features, tree.left)
        else:
            return self._predict(features, tree.right)

    def predict(self, features):
        return np.array([self._predict(feature, self.tree).value for feature in features])

    def fit_predict(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> pd.DataFrame:
        possible_features = get_all_possible_features()

        predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])

        users = np.unique(x_train['user'])

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

        for user in tqdm(users):
            user_train = x_train[x_train['user'] == user]

            labels = user_train['rating']

            movies_watched = [movie
                              for movie in users_movies_train
                              if movie.id in user_train['movie'].values]

            features = [movie.features_as_dict(possible_features) for movie in movies_watched]

            self.fit(features, labels.to_list())

            user_test = x_test[x_test['user'] == user]

            movies_to_rate = [movie
                              for movie in users_movies_test
                              if movie.id in user_test['movie'].values]

            features = [movie.features_as_dict(possible_features) for movie in movies_to_rate]

            user_predictions = self.predict(features)
            for i in range(len(user_predictions)):
                predictions.loc[len(predictions.index)] = [
                    int(user_test.iloc[i]['id']),
                    int(user),
                    int(user_test.iloc[i]['movie']),
                    int(user_predictions[i])
                ]

        return predictions

    def get_plot_tree_for(self, tree_id: str):

        dot = graphviz.Digraph(name=f'cluster_{tree_id}', comment='Decision Tree')
        def add_nodes_recursively(root: Node, depth=0):
            if root is None or root.left is None or root.right is None:
                return

            node_left_id = f"{tree_id}_{root.left}"
            dot.node(node_left_id, root.left.get_label())

            node_right_id = f"{tree_id}_{root.right}"
            dot.node(node_right_id, root.right.get_label())

            root_node_id = f"{tree_id}_{root}"
            dot.edges([(f"{root_node_id}", node_left_id), (f"{root_node_id}", node_right_id)])

            add_nodes_recursively(root.left, depth + 1)
            add_nodes_recursively(root.right, depth + 1)

        root_node_id = f"{tree_id}_{self.tree}"
        dot.node(root_node_id, self.tree.get_label())
        add_nodes_recursively(self.tree)

        return dot

    def plot_tree(self, x_train: pd.DataFrame, user: int):
        possible_features = get_all_possible_features()

        user_train = x_train[x_train['user'] == user]

        user_train = pd.merge(user_train, movies_features, left_on='movie', right_on='id')
        user_train = user_train.drop(columns=['id_x', 'id_y', 'user', 'movie', 'tmdb'])

        labels = user_train['rating']

        movies_watched = [(
            Movie(None,
                  None,
                  row['title'],
                  row['genres'],
                  row['popularity'],
                  row['year'],
                  row['runtime'],
                  row['cast'],
                  row['keywords'],
                  row['similar']))
            for _, row in user_train.iterrows()]

        features = [movie.features_as_dict(possible_features) for movie in movies_watched]

        self.fit(features, labels.to_numpy())

        dot = graphviz.Digraph(comment='Decision Tree')

        def add_nodes_recursively(root: Node, depth=0):
            if root is None or root.left is None or root.right is None:
                return

            node_left_id = f"{root.left}"
            dot.node(node_left_id, root.left.get_label())

            node_right_id = f"{root.right}"
            dot.node(node_right_id, root.right.get_label())

            dot.edges([(f"{root}", node_left_id), (f"{root}", node_right_id)])

            add_nodes_recursively(root.left, depth + 1)
            add_nodes_recursively(root.right, depth + 1)

        root_node_id = f"{self.tree}"
        dot.node(root_node_id, self.tree.get_label())
        add_nodes_recursively(self.tree)

        dot.render('decision_tree.gv', view=True)
