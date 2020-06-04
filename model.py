import pandas as pd
import sklearn as sk
import numpy as np
import pickle 
import GetOldTweets3 as got
from pandas import DataFrame

username = 'MBTA'
count = 30

# Creation of query 
tweetCriteria = got.manager.TweetCriteria().setUsername(username).setMaxTweets(count)

#Tweets into a list
tweets = []
for i in range(count):
    twit = got.manager.TweetManager.getTweets(tweetCriteria)[i]
    tweets.append(twit.text)
df = DataFrame(tweets, columns = ['Tweet'])
print(df)



