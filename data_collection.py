# handles the environment variables
import os
# deals with dataframes and csv format
import pandas as pd
# deals with the API calls
from googleapiclient.discovery import build
# to clean the data
import re
import html
from nltk.corpus import stopwords
# for dimension reduction
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
# for efficient operations
import numpy as np


YT_API_KEY = os.environ['YT_API_KEY']

# ----------------------------------------- PART 1: Extract the data ------------------------------------------ #


def get_channel_id(service):
    """Extracts details about channel, so we can visualise the channel id."""
    # list method to list specific channels. We look for the id, because we need it for the playlist
    # content Details will be accessed, we can find the channel id there
    id_request = service.channels().list(
        part='contentDetails, statistics',
        forUsername='deutschewelleenglish'
    )
    id_response = id_request.execute()
    print(id_response)


def get_playlist(service):
    """Extracts details about channel playlists. Saves relevant playlists into a dictionary. Returns the dictionary."""
    # after having retrieved the id, we can look for the playlist
    # the title of the playlist is within the snippet dictionary
    pl_request = service.playlists().list(
        part="snippet",
        channelId="UCknLrEdhRCp1aegoMqRaCZg",
        maxResults=20
    )
    pl_response = pl_request.execute()
# creating a dictionary with the playlist titles and their ids if the playlists are about covid
    covid_playlists = {}
    for item in pl_response['items']:
        if 'coronavirus' in item['snippet']['title'].lower():
            covid_playlists[item['snippet']['title']] = item['id']
    print(covid_playlists)
    return covid_playlists


def get_playlist_video(service, pl_dict):
    """Extracts details about playlist videos and saves the video ids into a list. Returns a list of video ids."""
    # access all videos in the playlists
    # we could also extract the publication date
    all_videos = []
    for playlist_id in pl_dict.values():
        video_request = service.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=45
        )
        video_response = video_request.execute()
        # adding all the video ids to the list
        for item in video_response['items']:
            all_videos.append(item['contentDetails']['videoId'])
    return all_videos


def get_comments(service, video_ids):
    """Extracts comments each video in the list of videos. Returns a list of all comments."""
    # get comments (for this purpose I have avoided replies)
    all_comments = []
    for video in video_ids:
        comment_request = service.commentThreads().list(
            part='snippet',
            order='relevance',
            videoId=video,
            maxResults=100
        )
        comment_response = comment_request.execute()

        for item in comment_response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            all_comments.append(comment)
    return all_comments


def save_comments(comments_list):
    """Stores comments into a pandas dataframe and saves the comments into a csv file. """
    # TODO: pandas dataframe
    comment_df = pd.DataFrame(comments_list, columns=['Comments'])
    comment_df.to_csv('Data.csv')
    # extracted on 27th 02 2022
    print("Comments saved successfully.")


def get_raw_data():
    """Creates a service and calls other functions."""
    yt_service = build('youtube', 'v3', developerKey=YT_API_KEY)
    # get_channel_id(yt_service) only needed once
    playlists_dict = get_playlist(yt_service)
    video_ids_list = get_playlist_video(yt_service, playlists_dict)
    comments = get_comments(yt_service, video_ids_list)
    #save_comments(comments)
    print("Comments extracted successfully")


# ------------------------------------------ PART 2: clean the data -------------------------------------------------#

def clean_data(dataframe):
    """Removes stopwords from the dataframe, creates a new column in the dataframe without them."""
    # requires: nltk.download('omw-1.4') - Open Multilingual Wordnet, to be performed only once.
    #lemmatizer requires pos tag
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    sw.remove("not")
    no_sw_comments = []
    for comment in dataframe["Comments"]:
        esc_comment = html.unescape(comment)
        dataframe['Comments'] = dataframe['Comments'].replace(comment, esc_comment)
        no_punct = re.sub(r"[^\w\s]", " ", comment)
        no_num = re.sub(r"\d+", '', no_punct)
        tokens = [w for w in no_num.lower().split() if w not in sw]
        lemmas = []
        # because of the linguistic characteristics of the data, pos tagging and lemmatization are not always accurate.
        for word, tag in pos_tag(tokens):
            w_tag = tag[0].lower()
            w_tag = w_tag if w_tag in ['a', 'r', 'n', 'v'] else ''
            if w_tag == '':
                lemma = word
            else:
                lemma = lemmatizer.lemmatize(word)
            lemmas.append(lemma)
        lemma_sent = ' '.join(lemmas)
        no_sw_comments.append(lemma_sent)

    dataframe["Clean"] = no_sw_comments

    return dataframe


def label_data(dataframe):
    """Labels data based on keyword lists. Drops unlabeled data from the dataset"""
    no_vax_pattern = r"(stop|scam|tyrant|lie|control|bs|fear|no vaccine|freedom|discrimination|brainwashed|profit|" \
                     r"circus|unvaxxed|oppressor|inject|dignity|toxic|covid|unvaccinated|fishy|" \
                     r"against vaccine|manipulated)"
    pro_vax_pattern = r"(stay healthy|stay safe|fully vaccinated|long covid|nurses|get vaccinated)"
    dataframe["Label"] = ''

    for i, comment in dataframe["Comments"].iteritems():
        match = re.search(pro_vax_pattern, comment, re.I)
        if match:
            dataframe.loc[i, "Label"] = 1
        # labelling the novax comments.
    for i, comment in dataframe["Comments"].iteritems():
        match = re.search(no_vax_pattern, comment, re.I)
        if match:
            # if the comment is found, then it gets a 0
            dataframe.loc[i, "Label"] = 0
    # pandas does not recognize empty strings as not null values
    dataframe.replace(r'^\s*$', np.NaN, inplace=True, regex=True)
    # removing non latin alphabet
    dataframe.replace(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', np.NaN, inplace=True, regex=True)
    dataframe.dropna(inplace=True)
    # the label column consists of floats, whereas we want integers
    dataframe["Label"] = dataframe["Label"].astype(np.int64)
    # print(dataframe["Label"].value_counts())  how many labeled comments in each class.
    return dataframe


def main():
    """Calls all other functions to create a clean and labeled dataframe, which will be saved as a separate csv file
    for further use."""
    # calls the function only to create the dataframe
    #get_raw_data()
    data = pd.read_csv("Data.csv", index_col=0, encoding="utf-8")
    clean = clean_data(data)
    labeled = label_data(clean)
    # print(labeled.shape)
    labeled.to_csv("new_data.csv")


main()