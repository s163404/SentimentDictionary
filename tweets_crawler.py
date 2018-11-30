# - coding: utf-8 -

import sys
import json
from requests_oauthlib import OAuth1Session
import tweepy

# Constants
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""


def get_tweets():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    tweets = api.search(q='iPhone', count=10)
    for tweet in tweets:
        print(tweet.text, "/n")


if __name__ == '__main__':
    get_tweets()

sys.exit()
