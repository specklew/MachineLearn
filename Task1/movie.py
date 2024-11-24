import numpy as np


class Movie:
    def __init__(self,
                 dataset_id,
                 tmdb_id,
                 title,
                 genres,
                 popularity,
                 year,
                 runtime,
                 cast,
                 keywords,
                 similar):
        self.id = dataset_id
        self.tmdb_id = tmdb_id
        self.title = title
        self.genres = genres
        self.popularity = popularity
        self.year = year
        self.runtime = runtime
        self.cast = cast
        self.keywords = keywords
        self.similar = similar

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

    def __repr__(self):
        return self.as_dict().__repr__()
