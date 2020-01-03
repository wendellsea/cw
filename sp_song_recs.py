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


# connect to spotify to import song data
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'playlist-read-private'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')

gs_df = pd.read_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\good_songs.csv")
bs_df = pd.read_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\bad_songs.csv")
as_df = pd.read_csv("C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\all_songs.csv")

# get audio features using the built in recommendation feature from spotify
# i will pas a list of 'good' songs into the spotify function and it will return
# songs spotify believes i will like

# get recommended song ids based on 'good' song ids
gs_list = gs_df['song_id'].values.ravel()


def get_gs_recs(song_ids):

    gs_recs = []
    gs_recs_list = []
    # loop through my good song ids and use the spotify recommendation function
    for n in range(0, len(song_ids)):
        sr = sp.recommendations(seed_tracks=[song_ids[n]])
        gs_recs_list.append(sr)
    # once i get results from above, i need to isolate the specific fields i need
    # to start modeling
    # for now, i only need high level data because i am going to pass the song ids into the audio feature call below
    for i in range(0, len(gs_recs_list)):
        root = gs_recs_list[i]['tracks']
        for n in range(0, len(root)):
            song_id = root[n]['id']
            artist = root[n]['artists'][0]['name']
            song_name = root[n]['name']
            data = [song_id, artist, song_name]
            gs_recs.append(data)
    # i only want unique songs, so i filter for a song count equal to 1
    gs_recs = [i for i in gs_recs if gs_recs.count(i) == 1]
    df = pd.DataFrame(gs_recs, columns=['song_id', 'artist', 'song_name']).sort_values(
        by='song_id', ascending=True).reset_index(drop=True)
    print('I have a total of {} song ids that have been recommended to me based on my "good songs"'.format(len(df)))
    return df


ds_data = get_gs_recs(gs_list)

ds_data_np = ds_data['song_id'].values.ravel()
# this is simply an array of song ids that have been recommended to me from above
# i need this because i cannot pass a one-column vector into the sp call


def get_gs_features(song_ids):
    # get the audio features for the song spotify believes i will like
    df_list = []
    data = []

    for n in range(0, len(song_ids)):
        af = sp.audio_features(tracks=[song_ids[n]])
        df_list.append(af)

    for i in range(0, len(df_list)):

        root = df_list[i][0]

        song_id = root['id']
        song_length = root['duration_ms']
        tempo = root['tempo']
        acousticness = root['acousticness']
        danceability = root['danceability']
        energy = root['energy']
        instrumentalness = root['instrumentalness']
        liveness = root['liveness']
        loudness = root['loudness']
        speechiness = root['speechiness']

        afs = [song_id, song_length, tempo, acousticness, danceability, energy,
               instrumentalness, liveness, loudness, speechiness]

        data.append(afs)

    df = pd.DataFrame(data, columns=['song_id', 'song_length', 'tempo',
                                     'acousticness', 'danceability',
                                     'energy', 'instrumentalness', 'liveness',
                                     'loudness', 'speechiness'])
    df['song_length'] = (df['song_length'] / 1000) / 60
    df.insert(10, 'artist', df['song_id'].map(ds_data.set_index('song_id')['artist']))
    df.insert(11, 'song_name', df['song_id'].map(ds_data.set_index('song_id')['song_name']))

    print('I have a total of {} song features that have been recommended to me based on my "good songs"'
          .format(len(df)))
    print('The length of my dataframe matches the length of my song id array: {}'.format(len(ds_data) == len(df)))
    return df


ds_df = get_gs_features(ds_data_np)

# set up my x and y variables for the models
x = as_df.iloc[:, [5, 8, 9, 10, 11, 12, 13, 14, 15]]
y = as_df.iloc[:, 16]
x_ds = ds_df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9]]


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
    df = pd.DataFrame(ds.groupby('artist')[model_binary_name].sum()).sort_values(by=model_binary_name, ascending=False)
    print('The top five artist my model suggest I listen to are: {}'.format(df.iloc[0:5]))
    return ds


ds_final = get_final_ds_songs('rf_binary')

ds_final.to_csv('C:\\Users\\cwendell\\Desktop\\\\spotify\\updated\\ds_songs_final.csv')

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
ds_playlist = '0QDX3TbFlqjZ5wWweDl2lT'

ds_list_upload = []

n = len(ds_list)
while n > 0:
    n -= 1
    k = n+1
    ds_list_upload = ds_list[n:k]
    sp.user_playlist_add_tracks(user=username, playlist_id=ds_playlist, tracks=ds_list_upload)
    ds_list_upload = []
