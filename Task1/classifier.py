import similarity


class Classifier:
    def __init__(self, k_neighbours):

        self.k_neighbours = k_neighbours

    def fit_predict(self, considered_movie, watched_movies, movie_rating):

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


