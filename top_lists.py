import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

cid = 'ea16f8df9e7240e280b0b4ee58bfbfa4'
secret = 'b4d495228cb84289a0b9e8ea7d7a889c'
username = '123442807'

# connect to spotify to import song data
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'user-top-read'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')
if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

# get top tracks
# max track limit is 50; defaults to 20

terms = ['short_term', 'medium_term', 'long_term']

track_list = []
n = len(terms)-1
while n >= 0:

    n -=1
    data = sp.current_user_top_tracks(limit=50, time_range=terms[n])

    for i in range(0, len(data['items'])):

        root = data['items'][i]
        artist = root['artists'][0]['name']
        song_name = root['name']
        tracks = [artist, song_name, terms[n]]
        track_list.append(tracks)


artist_df = pd.DataFrame(track_list, columns=['artist', 'song_name', 'term'])


print('In the short-term, the artist I listened to the most is: '
      '{}; the medium term is {}; and the long term is {}'
      .format(artist_df.loc[artist_df['term'] == 'short_term']['artist'].value_counts().idxmax(),
              artist_df.loc[artist_df['term'] == 'medium_term']['artist'].value_counts().idxmax(),
              artist_df.loc[artist_df['term'] == 'long_term']['artist'].value_counts().idxmax()))

# using the above results, get top 10 artists by length of time (short, medium, long)
# 10 results is arbitrary - i can use any number
artist_count = pd.DataFrame(columns=['artist', 'term', 'count'])

n = len(terms)-1
while n >= 0:
    n -= 1

    def artist_summary(df_result, data, num_results):

        df1 = data[['artist', 'term']].loc[data['term'] == terms[n]]
        df2 = df1.groupby(['artist', 'term']
                         ).size().to_frame('count').sort_values(['count'], ascending=False).reset_index()[:num_results]
        df_result = df_result.append(df2)
        return df_result

    artist_count = artist_summary(artist_count, artist_df, 10)

sns.barplot(x='artist', y='count', hue='term', data=artist_count)
plt.legend(loc='upper right')
plt.xticks(rotation=45)

# get top artists
# max artist limit is 50; defaults to 20
# get genres, term, and artist into a single dataframe

a_genre = []

n = len(terms)-1
while n >= 0:
    n -=1
    z = sp.current_user_top_artists(limit=50, time_range=terms[n])
    for j in range(0, len(z['items'])):
        root = z['items']
        genre_root = root[j]['genres']
        artist = root[j]['name']

        for i in range(0, len(genre_root)):
            genre = genre_root[i]
            ag = [artist, genre, terms[n]]
            a_genre.append(ag)


genre_df = pd.DataFrame(a_genre, columns=['artist', 'genre', 'term'])

print('In the short-term, the genre associated with the most artists is: '
      '{}; the medium term is {}; and the long term is {}'
      .format(genre_df.loc[genre_df['term'] == 'short_term']['genre'].value_counts().idxmax(),
              genre_df.loc[genre_df['term'] == 'medium_term']['genre'].value_counts().idxmax(),
              genre_df.loc[genre_df['term'] == 'long_term']['genre'].value_counts().idxmax()))

# using the above results, get top 10 genres by length of time (short, medium, long)
# 10 results is arbitrary - i can use any number
genre_count = pd.DataFrame(columns=['genre', 'term', 'count'])

n = len(terms)-1
while n >= 0:
    n -= 1

    def genre_summary(df_result, data, num_results):

        df1 = data[['genre', 'term']].loc[data['term'] == terms[n]]
        df2 = df1.groupby(['genre', 'term']
                         ).size().to_frame('count').sort_values(['count'], ascending=False).reset_index()[:num_results]
        df_result = df_result.append(df2)
        return df_result

    genre_count = genre_summary(genre_count, genre_df, 10)

sns.barplot(x='genre',y='count',hue='term',data=genre_count)
plt.legend(loc='upper right')
plt.xticks(rotation=45)

