import pandas as pd
import numpy as np
from pandas import DataFrame

####PREPROCESSING
#selected keywords which indicate delays
keywords = 'delayed|delay|resume|resuming|apologies|resumed|Update|bypassing|suspended|detour|delays'
#use ?=(match this expression), to add line preferences --> TODO 

#importing data
df = pd.read_csv("data.csv", engine = "python")
del df['Unnamed: 0']

#adding alert column to flag relevant columnns
df['alert'] = np.where(df['tweet'].str.contains(keywords), 1, 0)
final = df
final.to_csv("final.csv")




