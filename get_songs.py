import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

cid = 'ea16f8df9e7240e280b0b4ee58bfbfa4'
secret = 'b4d495228cb84289a0b9e8ea7d7a889c'
username = '123442807'

# connect to spotify to import song data
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'playlist-read-private'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')
if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

# this is a simplified script i can use to get audio features from a playlist


def get_playlist_tracks(playlist_id):
    results = sp.user_playlist_tracks(username, playlist_id)
    tracks = results['items']
    while results['next']:
        # iterates over every song in the playlist and stores it in the results string
        results = sp.next(results)
        # extends the initial tracks list with the items i've iterated over in the new results string
        # now have every song in the playlist
        # think of this as extending all of the song information in 'song_dict' below
        tracks.extend(results['items'])
    return tracks


def get_song_data(song_dict):

    song_list = []

    for i in range(0, len(song_dict)):
        basics = song_dict[i]

        # Song basics
        song_name = basics['track']['name']
        artist_name = basics['track']['artists'][0]['name']
        total_songs = basics['track']['album']['total_tracks']
        date_added = basics['added_at']
        release_date = basics['track']['album']['release_date']
        song_length = basics['track']['duration_ms']
        song_number = basics['track']['track_number']
        song_id = basics['track']['id']

        # Song sound features
        features = sp.audio_features(song_id)
        tempo = features[0]['tempo']
        acousticness = features[0]['acousticness']
        danceability = features[0]['danceability']
        energy = features[0]['energy']
        instrumentalness = features[0]['instrumentalness']
        liveness = features[0]['liveness']
        loudness = features[0]['loudness']
        speechiness = features[0]['speechiness']

        track = [song_name, artist_name, total_songs, date_added, release_date,
                    song_length, song_number, song_id, tempo, acousticness,
                    danceability, energy, instrumentalness, liveness, loudness, speechiness]
        song_list.append(track)
    return song_list


def create_song_df(data, target_number):
    song_data = pd.DataFrame(data, columns=[
        'song_name', 'artist_name', 'total_album_songs', 'date_added', 'release_date',
        'song_length', 'song_number', 'song_id', 'tempo', 'acousticness',
        'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness'])
    song_data['target'] = target_number
    song_data['song_length'] = (song_data['song_length'] / 1000) / 60
    return song_data


bs_dict = get_playlist_tracks('2aAjiIFAV7sfiStrQ2rFH9')
bs_data = get_song_data(bs_dict)
bs_df = create_song_df(bs_data, 0)

gs_dict = get_playlist_tracks('7k3FLP5PuUGQIcmdZqUcYe')
gs_data = get_song_data(gs_dict)
gs_df = create_song_df(gs_data, int(1))

ds_dict = get_playlist_tracks('6UorRuGqRJqnHJZOjxYxXX')
ds_data = get_song_data(ds_dict)
ds_df = create_song_df(ds_data, 0)
del ds_df['target']

# create all songs dataframe
as_df = pd.concat([gs_df, bs_df], ignore_index=True, sort=False)


# remove duplicates from discover songs
def remove_duplicates(ds_dupes):
    cond = ds_dupes['song_id'].isin(as_df['song_id'])
    ds_dupes.drop(ds_dupes[cond].index, inplace=True)
    print(
        'After I remove songs that appear in both my discover songs and good songs, '
        'I have a total of {} songs to potentially discover'.format(
            len(ds_dupes)))
    return ds_dupes


ds_df = remove_duplicates(ds_df)

# export/import good, bad, and discover song dataframes
gs_df.to_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\good_songs.csv", sep=',', index=False)
bs_df.to_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\bad_songs.csv", sep=',', index=False)
ds_df.to_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\discover_songs.csv", sep=',', index=False)
as_df.to_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\all_songs.csv", sep=',', index=False)
