# To run this code, first edit config.py with your configuration, then:
#
# mkdir data
# python twitter_stream_download.py -q apple -d data
#
# It will produce the list of tweets for the query "apple"
# in the file data/stream_apple.json

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from config_ignore import *
import time
import argparse
import string
import json
from tqdm import tqdm
import random

THRESHOLD_SAMPLE = 0.2

RETRIEVE_MAX_TWEETS = 200



if __name__ == '__main__':
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    tweets = {}
    user_data = api.get_user(screen_name='Albert_Rivera')
    MAX_TWEETS = user_data.statuses_count
    print('MAX TWEETS: {}'.format(MAX_TWEETS))
    user_id = user_data.id_str
    statuses = api.user_timeline(id=user_id, count=1, include_rts=False, tweet_mode="extended")
    tweets[0] = statuses[0]._json['full_text']
    last_id =  statuses[0]._json['id']
    for n_tweets in tqdm(range(0, MAX_TWEETS, RETRIEVE_MAX_TWEETS)):
        statuses = api.user_timeline(id=user_id, count=RETRIEVE_MAX_TWEETS, max_id=last_id, include_rts=False, tweet_mode="extended")
        for tweet in statuses:
            tweets[len(tweets)] = tweet._json['full_text']
        if random.uniform(0, 1) < THRESHOLD_SAMPLE:
            print(tweet._json['full_text'])
        last_id =  tweet._json['id']
        if (n_tweets/RETRIEVE_MAX_TWEETS)%1000==0 and n_tweets>0:
            time.sleep(15*60)



    json.dump(tweets, open('lloron.json', 'a'))
    print(len(tweets))
