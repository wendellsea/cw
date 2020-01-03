import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

cid = 'ea16f8df9e7240e280b0b4ee58bfbfa4'
secret = 'b4d495228cb84289a0b9e8ea7d7a889c'
username = '123442807'
ds_playlist ='37i9dQZF1DX6ujZpAN0v9r'
bs_playlist = '1wHyQ9y7Yk0zBj8ojBRcFJ'
gs_playlist = '3rU16AG4zvcofGclLGTjT9'

# sp.audio_analysis --> gives very detailed song feature data. i could use this for a deeper analysis
# get recommendations by genre
# genres = pd.DataFrame(sp.recommendation_genre_seeds())

# connect to spotify to import song data
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'playlist-read-private'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')


#gs_df = pd.read_csv("L:\\MgdCareAnalysis\\PERSONAL FOLDERS\\Cody Wendell\\PFiles\\s\\good_songs.csv")
#ds_df = pd.read_csv('C:\\Users\\cwendell\\Desktop\\spotify\\discov_songs.csv')
#bs_df = pd.read_csv('L:\\MgdCareAnalysis\\PERSONAL FOLDERS\\Cody Wendell\\PFiles\\s\\bad_songs.csv')

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
        song_id = song_dict[i]['track']['id']

        track = [song_id]
        song_list.append(track)
    return song_list


gs_dict = get_playlist_tracks(gs_playlist)
gs_df = get_song_data(gs_dict)

bs_dict = get_playlist_tracks(bs_playlist)
bs_df = get_song_data(bs_dict)

ds_dict = get_playlist_tracks(ds_playlist)
ds_df = get_song_data(ds_dict)

gs_ids = np.array(gs_df).ravel()
ds_ids = np.array(ds_df).ravel()
bs_ids = np.array(bs_df).ravel()


# get audio analysis features from 'good', 'bad', and 'discover' songs
# audio analysis doesn't have song id. so, ill have to use the song id list to get the data
# then i will have to merge the data into my main dataframe

gs_song_ids = []
ds_song_ids = []
bs_song_ids = []


def get_sp_audio(song_ids, song_id_list):

    for i in range(0, len(song_ids)):
        aa = sp.audio_analysis(song_ids[i])
        song_id_list.append(aa)


get_sp_audio(ds_ids, ds_song_ids)


def get_audio_analysis(song_id_list):

    app = []

    for j in range(0, len(song_id_list)):
        track_root = song_id_list[j]
        samples = track_root['track']['num_samples']
        duration = track_root['track']['duration']
        loudness = track_root['track']['loudness']
        tempo = track_root['track']['tempo']

        for h in range(0, len(track_root)):
            bars = len(track_root['bars'])
            beats = len(track_root['beats'])
            tatums = len(track_root['tatums'])
            sections = len(track_root['sections'])
            segments = len(track_root['segments'])

            a = [samples, duration, loudness, tempo, bars, beats, tatums, sections, segments]

        app.append(a)

    df = pd.DataFrame(app, columns=['samples', 'duration', 'loudness', 'tempo',
                                    'bars', 'beats', 'tatums', 'sections', 'segments'])

    df['bars/second'] = df['bars'] / df['duration']
    df['beats/second'] = df['beats'] / df['duration']
    df['tatums/second'] = df['tatums'] / df['duration']

    return df


ds_df = get_audio_analysis(ds_song_ids)


def concat_song_ids(data, song_ids):

    ids = pd.DataFrame(np.vstack(song_ids), columns=['song_id'])
    df = pd.concat([data, ids], axis=1)
    return df


ds_df = concat_song_ids(ds_df, ds_ids)

gs_df['target'] = 1
bs_df['target'] = 0


as_df = pd.concat([gs_df, bs_df], ignore_index=True, sort=False)


# remove duplicates from discover songs
def remove_duplicates():
    cond = ds_df['song_id'].isin(as_df['song_id'])
    ds_df.drop(ds_df[cond].index, inplace=True)
    print(
        'After I remove songs that appear in both my discover songs and all songs, '
        'I have a total of {} songs to potentially discover'.format(
            len(ds_df)))
    return ds_df


ds_df = remove_duplicates()


# set up my x and y variables for the models
x = as_df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = as_df.iloc[:, 13]
x_ds = ds_df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]


def training_data(x, y):
    return train_test_split(x, y, test_size=0.30, random_state=1)


x_train, x_test, y_train, y_test = training_data(x, y)
print('The length of training data is {} songs and the length of the test data is {} songs'
      .format(len(x_train), len(x_test)))


def optimal_model_level():

    rf_levels = range(1, 100)
    rf_scores = []
    k_levels = range(1, 25)
    k_scores = []

    for n in rf_levels:
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        rf_scores.append(metrics.accuracy_score(y_test, y_pred))
    print('The first 5 n scores are: {}'.format(rf_scores[0:5]))

    for k in k_levels:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train.ravel())
        y_pred = knn.predict(x_test)

        k_scores.append(metrics.accuracy_score(y_test, y_pred))
    print('The first 5 k scores are: {}'.format(k_scores[0:5]))

    plt.subplot(1, 2, 1)
    plt.plot(rf_levels, rf_scores)
    plt.title('Random Forest Model: Level of accuracy by level of n')
    plt.xlabel('N level')
    plt.ylabel('Accuracy Score')

    plt.subplot(1, 2, 2)
    plt.plot(k_levels, k_scores)
    plt.title('K Nearest Neighbor: Level of accuracy by level of k')
    plt.xlabel('K level')
    plt.ylabel('Accuracy Score')


optimal_model_level()


def knn_model(model, x):

    model.fit(x_train, y_train)
    knn_pred = model.predict(x)
    print('Of the total {} songs in my discovery playlist, using the KNN model, '
          'it is predicted I will like approximately {} song(s)'.format(len(knn_pred), knn_pred.sum()))
    print('The mean accuray level of my model is: {}'.format(model.score(x_test, y_test)))
    return knn_pred


knn_ds = knn_model(KNeighborsClassifier(n_neighbors=5), x_ds)


def rand_forest_model(model, x):
    model.fit(x_train, y_train)
    rf_pred = model.predict(x)
    print('Of the total {} songs in my discovery playlist, using the Random Forest model, '
          'it is predicted I will like approximately {} song(s)'.format(len(rf_pred), rf_pred.sum()))
    print('The mean accuray level of my model is: {}'.format(model.score(x_test, y_test)))

    return rf_pred


rf_ds = rand_forest_model(RandomForestClassifier(n_estimators=33), x_ds)


def get_knn_ds_results(results, df):
    knn_binary = pd.DataFrame(np.vstack(results)).reset_index()
    df['knn_binary'] = knn_binary[0].values
    df = df[df['knn_binary'] == 1]
    return df


knn_results = get_knn_ds_results(knn_ds, ds_df)


def get_rf_ds_results(results, df):

    rf_binary = pd.DataFrame(np.vstack(results)).reset_index()
    df['rf_binary'] = rf_binary[0].values
    df = df[df['rf_binary'] == 1]
    return df


rf_results = get_rf_ds_results(rf_ds, ds_df)


def get_final_ds_songs(model_binary_name):
    # get final recommended songs (based on the model with the higher accuracy score)
    ds = ds_df[ds_df[model_binary_name] == 1]
    return ds


ds_final = get_final_ds_songs('rf_binary')

# upload all unique songs to 'discover songs' playlist in Spotify
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')
if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

ds_list = list(ds_final['song_id'])
ds_playlist = '3i2jtjChM8YqpxBTUxPg02'

ds_list_upload = []

n = len(ds_list)
while n > 0:
    n -= 1
    k = n+1
    ds_list_upload = ds_list[n:k]
    sp.user_playlist_add_tracks(user=username, playlist_id=ds_playlist, tracks=ds_list_upload)
    ds_list_upload = []


