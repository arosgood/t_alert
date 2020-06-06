import pandas as pd
from pandas import DataFrame
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn import metrics

#importing data
df = pd.read_csv('final.csv', engine = "python")
del df['Unnamed: 0']

df.tweet = df.tweet.astype(str)
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df.tweet)
tfidf_transformer = TfidfTransformer()
X_tf = tfidf_transformer.fit_transform(X_counts)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X_tf,df.alert,test_size=0.4)
model = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
model.fit(Train_X,Train_Y)
pred = model.predict(Test_X)

# save the model to disk
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))