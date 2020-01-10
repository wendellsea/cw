# Spotify - Python: Predictive Modeling, Machine Learning, & Song Recommendations.
This is my first attempt at using open-source Spotify data to predict songs I will enjoy listening to.

The purpose/objective of this project is:

Gather all song listening data from Spotify; use this data to summarize my musical taste; create a tool - similar to the song recommendation
feature in Spotify - to independently get song recommendations based on my listening history.

# The code in the attached files mainly uses the following Python libraries:

1.) Pandas - I used this library for all data analysis/manipulation, organization (filtering, sorting, slicing, etc.), and categorization.

2.) Numpy - This was primarily used for data manipulation, transformation to arrays and to furture help develop my predictive models.

3.) Matplotlib - I used this library for: basic graphing of the results from my models, graphing of the current music I like, and
to help assign the most efficient variables to my predictive models.

4.) Sklearn - This library was used for my two predictive modeling techniques (K-Nearest Neighbors and Random Forests Classifier).

# Explanation of files:

1.) get_songs: This program is used to fetch my song data from Spotify. I am able to specify the particular playlist I want to get
data from and the corresponding song features (tempo, song name, artist, etc.). I can use this to split up my data into song I like, dislike, and songs I want to predict if I will like. Finally, I can use this program to export my data to csv for manipulation in Excel.

2.) sp_song_recs: This program is used in conjunction with the get_songs program above. Primarily, this program will use the two predictive modeling libraries to determine if I will like or dislike songs I have yet to listen to. I use basic song feature data (tempo, length of the song, energy, etc) provided by Spotify in my predictive model. I use the KNN and RandomForest models to first understand the accuracy of predicting whether I will like a song. Additionally, I use a few functions to get the most accurate input levels in my predictive models. Most importantly, after I run the data through my models, I get a final dataset of songs the model predicts (with x amount of accuracy) I will like listening to.

3.) upload_songs: This program has a primary function of uploading songs to a Spotify playlist. Once I get the songs my model predicts I will like, I use this program to upload the songs to Spotify so I can listen to them. 

4.) sp_audio_analysis: This program has a similar purpose to the 'sp_song_recs' program in that I am using song features in my predictive models. Whereas the 'sp_song_recs' program uses basic song features, this program uses more in depth song data from Spotify (bars, beats, beats/second, sections, etc). I pass these features through my predictive models to get a final list of songs the program believes I will like.

5.) top_lists: This program does not predict song, rather, it gives me a history of my most listened to genres and artists by time period. I am able to understand how my musical taste has changed over the short, medium, and long term. I plot some basic graphs that shows which artists I like the most, by period of time, and genres I like the most.
