import pandas as pd
import numpy as np
import sklearn as sk

df = pd.read_csv('final.csv', engine = "python")
del df['Unnamed: 0']



