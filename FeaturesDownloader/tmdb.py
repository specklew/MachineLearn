import requests as req
import os
from movie import Movie

api_key = os.environ['API_KEY']
api_url = 'https://api.themoviedb.org/3/movie/'


def get(dataset_id, tqdm):
    url = api_url + str(tqdm) + '?api_key=' + api_key + '&append_to_response=credits,keywords,similar'
    res = req.get(url=url).json()

    movie = Movie(dataset_id=dataset_id,
                  tmdb_id=tqdm,
                  title=res['original_title'],
                  genres=[genre['id'] for genre in res['genres']],
                  popularity=int(res['popularity']),
                  year=int(res['release_date'][:4]),
                  runtime=res['runtime'],
                  cast=[actor['name'] for actor in res['credits']['cast']],
                  keywords=[keyword['name'] for keyword in res['keywords']['keywords']],
                  similar=[movie['id'] for movie in res['similar']['results']])

    return movie
