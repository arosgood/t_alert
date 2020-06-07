from credentials import *
import tweepy 
import pickle
import datetime
import pandas as pd
from pandas import DataFrame

class Alert:

    def __init__(self, hours):
        self.username = 'MBTA'
        self.hours = hours
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)   
        self.now = datetime.datetime.now()
        self.end_time = datetime.datetime(int(self.now.year), int(self.now.month), int(self.now.day), int(self.now.hour)-self.hours, int(self.now.minute))  
        
    def collect(self):
        tweets = []
        temp = self.api.user_timeline(self.username)
        for tweet in temp:
            if tweet.created_at < self.now and tweet.created_at > self.end_time:
                tweets.append(tweet)
        return tweets

        while (temp[-1].created_at > self.end_time):
            temp = api.user_timeline(self.username, max_id = temp[-1].id)
            for tweet in temp:
                if tweet.created_at > self.end_time and tweet.created_at < self.now:
                    tweets.append(tweet)


    def clean(self, tweets):
        classify = []   
        filename = 'model.sav'
        model = pickle.load(open(filename, 'rb'))
        for twit in tweets:
            classify.append(twit.text)
        classify = DataFrame(classify)
        print(classify)


if __name__ == "__main__":
    a = Alert(2)
    tweets = a.collect()
    print(a.clean(tweets))

