import pandas as pd
from tqdm import tqdm

processed_movie_set = pd.read_csv('dataset/processed_movies.csv', sep=';')
task_set = pd.read_csv('dataset/task.csv', sep=';', names=['id', 'user', 'movie', 'rating'])
train_set = pd.read_csv('dataset/train.csv', sep=';', names=['id', 'user', 'movie', 'rating'])


if __name__ == '__main__':

    merged_task_set = pd.merge(task_set, processed_movie_set, left_on='movie', right_on='id')
    merged_train_set = pd.merge(train_set, processed_movie_set, left_on='movie', right_on='id')

    merged_task_set.to_csv('dataset/merged_task.csv', sep=';', index=False)
    merged_train_set.to_csv('dataset/merged_train.csv', sep=';', index=False)

