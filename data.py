import GetOldTweets3 as got
from pandas import DataFrame

username = 'MBTA'

# Creation of query 
tweetCriteria = got.manager.TweetCriteria().setUsername(username).setSince("2019-06-04").setUntil("2020-06-04")

data= got.manager.TweetManager.getTweets(tweetCriteria)
tweets = []
for twit in data:
    tweets.append(twit.text)
df = DataFrame(tweets)
df.to_csv("data.csv")

""" tweets = []
for i in range(count):
        tweets.append(got.manager.TweetManager.getTweets(tweetCriteria)[i].text)
df = DataFrame(tweets, columns = ['Tweet'])
print(df)
df.to_csv("data.csv") """