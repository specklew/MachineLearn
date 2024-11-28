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
                  genres=[genre['name'] for genre in res['genres'][:3]],
                  popularity=int(res['popularity']),
                  year=int(res['release_date'][:4]),
                  runtime=res['runtime'],
                  cast=[actor['name'] for actor in res['credits']['cast'][:3]],
                  keywords=[keyword['name'] for keyword in res['keywords']['keywords'][:0]],
                  similar=[movie['id'] for movie in res['similar']['results'][:3]])

    return movie
