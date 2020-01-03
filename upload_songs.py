import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

cid = 'ea16f8df9e7240e280b0b4ee58bfbfa4'
secret = 'b4d495228cb84289a0b9e8ea7d7a889c'
username = '123442807'

# connect to spotify for adding good songs to new playlist
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
scope = 'playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret,
                                   redirect_uri='http://localhost/')
if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

# i want to create a playlist with all of my songs i like
# i have songs saved in multiple playlists, so i will need to combine all songs into one
# import song id's list from 'good song' playlists: top 2019, panda 19, panda 18, and electronic
gs_list = []

gs_playlists = ['37i9dQZF1EtbrolxU1Y9wS', '0527PAYSfDBZsYpJWZ3TYL',
                       '3S2QZM9Pv6lINrMnJexbuK','7k3FLP5PuUGQIcmdZqUcYe']

n = len(gs_playlists)
while n > 0:
    n -= 1
    # the spotify call retrieves data based on the full playlist
    # so i only need to loop through my playlist list by playlist id

    def create_gs_ids():
        results = sp.user_playlist_tracks(username, gs_playlists[n])
        tracks = results['items']
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
        # once i get the data from the playlist, i only need the song id to upload to spotify
        # i loop through all the tracks and isolate song ids only and append to a list
        for i in range(0, len(tracks)):
            song_id = tracks[i]['track']['id']
            gs_list.append(song_id)
        return gs_list
    gs_list = create_gs_ids()

print(len(gs_list))


# find duplicates within good song ids before uploading to spotify
gs_list = [f for f in gs_list if gs_list.count(f) == 1]
print(len(gs_list))

# upload all unique songs to 'good songs' playlist in Spotify
gs_playlist = '3rU16AG4zvcofGclLGTjT9'

gs_list_upload = []

n = len(gs_list)
while n > 0:
    n -= 1
    k = n+1
    gs_list_upload = gs_list[n:k]
    sp.user_playlist_add_tracks(user=username, playlist_id=gs_playlist, tracks=gs_list_upload)
    gs_list_upload = []

# import song id's list from bad songs playlist
bs_list = []

bs_playlists = ['37i9dQZEVXbLRQDuF5jeBp', '37i9dQZF1DXbDSHGzTpRHX',
                '37i9dQZF1DX82GYcclJ3Ug', '37i9dQZF1DX2Nc3B70tvx0',
                '6UZqaILSm6RfhkaegChJvU','3rU16AG4zvcofGclLGTjT9','2PjXETfoou9bVLsF9l9kzR']

n = len(bs_playlists)
while n > 0:
    n -= 1

    def create_bs_ids():
        results = sp.user_playlist_tracks(username, bs_playlists[n])
        tracks = results['items']
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
        for i in range(0, len(tracks)):
            song_id = tracks[i]['track']['id']
            bs_list.append(song_id)
        return bs_list
    bs_list = create_bs_ids()

print(len(bs_list))


# find duplicates within bad song ids before uploading to spotify
bs_list = [f for f in bs_list if bs_list.count(f) == 1]
print(len(bs_list))

# upload all unique songs to 'bad songs' playlist in Spotify
bs_playlist = '1wHyQ9y7Yk0zBj8ojBRcFJ'

bs_list_upload = []

n = len(bs_list)
while n > 0:
    n -= 1
    k = n+1
    bs_list_upload = bs_list[n:k]
    sp.user_playlist_add_tracks(user=username, playlist_id=bs_playlist, tracks=bs_list_upload)
    bs_list_upload = []
