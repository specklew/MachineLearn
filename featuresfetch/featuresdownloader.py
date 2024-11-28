import tmdb
import tqdm
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed


movie_set = pd.read_csv('dataset/movie.csv', sep=';', names=['id', 'tmdb', 'name'])


if __name__ == '__main__':

    print(tmdb.get(1, 11))

    processed_movies = []

    print('Fetching additional features from tmdb...')

    with tqdm.tqdm(total=len(movie_set)) as pbar:
        with ThreadPoolExecutor(max_workers=len(movie_set)) as ex:
            futures = [ex.submit(tmdb.get, row['id'], row['tmdb']) for _, row in movie_set.iterrows()]
            for future in as_completed(futures):
                pbar.update(1)
                processed_movies.append(future.result())

    print('Saving processed movies to disk...')

    df = pd.DataFrame([movie.as_dict() for movie in processed_movies])

    df.to_csv('dataset/processed_movies.csv', sep=';', index=False)

    print('Successfully saved {} processed movies to dataset/processed_movies.csv.'.format(len(df)))

