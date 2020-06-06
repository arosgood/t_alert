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











""" df.tweet = df.tweet.str.replace('1', '')
df.tweet = df.tweet.str.replace('2', '')
df.tweet = df.tweet.str.replace('3', '')
df.tweet = df.tweet.str.replace('4', '')
df.tweet = df.tweet.str.replace('5', '')
df.tweet = df.tweet.str.replace('6', '')
df.tweet = df.tweet.str.replace('7', '')
df.tweet = df.tweet.str.replace('8', '')
df.tweet = df.tweet.str.replace('9', '')
df.tweet = df.tweet.str.replace('0', '')
df.tweet = df.tweet.astype(str)
#attempt
np.random.seed(500)
Corpus = df
Corpus['tweet'] = [entry.lower() for entry in Corpus['tweet']]
Corpus['tweet']= [word_tokenize(entry) for entry in Corpus['tweet']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index,entry in enumerate(Corpus['tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index,'text_final'] = str(Final_words)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['alert'],test_size=0.3)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100) """








