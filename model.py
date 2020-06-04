import pandas as pd
import sklearn as sk
import numpy as np
import pickle 
import GetOldTweets3 as got
from pandas import DataFrame

df = pd.read_csv("data.csv", engine = "python")
del df['Unnamed: 0']
df['tweet'] = pd.Series.to_string(df['tweet'])
print(type(df['tweet']))
df['alert'] = np.where(df['tweet'].contains("statement"), 1, 0)
test = df
test.to_csv("check.csv")


