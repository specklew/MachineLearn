import os
import pandas as pd

from tqdm import tqdm

from classifier import Classifier
from movie import Movie
from trainhelp import train_test_split

train_set = pd.read_csv('../Dataset/train.csv', sep=';', names=['id', 'user', 'movie', 'rating'])
task_set = pd.read_csv('../Dataset/task.csv', sep=';', names=['id', 'user', 'movie', 'rating'])
processed_movie_set = pd.read_csv('../Dataset/processed_movies.csv', sep=';')

is_test = os.environ['IS_TEST'] == 'true'


def predict(user, tmdb):
    knnclassifier = Classifier(k_neighbours=5)

    considered_movie_raw = processed_movie_set[processed_movie_set['id'] == tmdb].iloc[0]
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

    user_movies = train_set[train_set['user'] == user]

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
        for _, row in processed_movie_set.iterrows()
        if row['id'] in user_movies['movie'].values]

    movies_ratings = {}

    for movie in movies_watched:
        rating = user_movies[user_movies['movie'] == movie.id]['rating'].values[0]
        movies_ratings[movie] = rating

    predicted_rating = knnclassifier.fit_predict(considered_movie, movies_watched, movies_ratings)
    return predicted_rating


def calculate_accuracy(predictions_df, test_df):
    correct = 0
    for _, prediction in predictions_df.iterrows():
        if prediction['rating'] == test_df[test_df['id'] == prediction['id']]['rating'].values[0]:
            correct += 1
    return correct / len(predictions_df)


if __name__ == '__main__':

    predictions = pd.DataFrame(columns=['id', 'user', 'movie', 'rating'])
    train, test = pd.DataFrame(), pd.DataFrame()

    if is_test:
        print('Splitting Dataset into train and test sets...\n')
        train, test = train_test_split(train_set)
        print('Train set size:', len(train))
        print('Test set size:', len(test))
    else:
        train = train_set
        test = task_set

    print('\nPredicting ratings for task set...\n')

    for _, row in tqdm(test.iterrows(), total=len(test)):
        predictions.loc[len(predictions.index)] = [
            int(row['id']),
            int(row['user']),
            int(row['movie']),
            int(predict(int(row['user']), int(row['movie'])))]

    accuracy = 0
    if is_test:
        print('Calculating accuracy...\n')
        accuracy = calculate_accuracy(predictions, test)

    print('')
    print('Predictions done!')
    print('===================')
    print('Ratings predicted:', len(predictions))

    if is_test:
        print('Accuracy:', accuracy)

    print('===================')
    print('')
    print('Saving predictions...')
    predictions.to_csv('submission.csv', sep=';', index=False, header=False)
