import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
import string
import nltk
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle
from matplotlib import cm as c

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

print(fake.shape)
print(true.shape)

# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
print(data.shape)

# Shuffle the data
data = shuffle(data)
data = data.reset_index(drop=True)

# Check the data
print(data.head())

# Removing the date (we won't use it for the analysis)
data.drop(["date"],axis=1,inplace=True)

# Removing the title (we will only use the text)
data.drop(["title"],axis=1,inplace=True)

# Convert to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())

# Remove punctuation    !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

# Removing stopwords   
nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

accuracy=0

diff_size=[0.1,0.2,0.3,0.4,0.5]

for i in diff_size:

    # Split the data
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=i, random_state=42)
    print("size of train data set")
    print(len(X_train))
    # Vectorizing and applying TF-IDF

    pipe_MNB = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                        ('model', MultinomialNB())])

    # Fitting the model
    model_MNB = pipe_MNB.fit(X_train, y_train)

    prediction = model_MNB.predict(X_test)

    # Accuracy
    accuracy_temp = accuracy_score(y_test, prediction)*100
    print("accuracy: {}%".format(round(accuracy_temp,2)))
    
    if(accuracy_temp>accuracy):
        accuracy=accuracy_temp
        # dumping whole model into pkl file
        pickle.dump(model_MNB,open('model_MNB.pkl','wb'))
    
    # cunfusion matrix
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))


accuracy=0

for i in diff_size:

    # Split the data
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=i, random_state=42)
    print("size of train data set")
    print(len(X_train))

    pipe_DT = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
    
    # Fitting the model
    model_DT = pipe_DT.fit(X_train, y_train)

    prediction = model_DT.predict(X_test)

    # Accuracy
    accuracy_temp = accuracy_score(y_test, prediction)*100
    print("accuracy: {}%".format(round(accuracy_temp,2)))
    
    if(accuracy_temp>accuracy):
        accuracy=accuracy_temp
        # dumping whole model into pkl file
        pickle.dump(model_DT,open('model_DT.pkl','wb'))
    
    # cunfusion matrix
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))


accuracy=0

for i in diff_size:

    # Split the data
    X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=i, random_state=42)
    print("size of train data set")
    print(len(X_train))

    pipe_RF = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
    
    # Fitting the model
    model_RF = pipe_RF.fit(X_train, y_train)

    prediction = model_RF.predict(X_test)

    # Accuracy
    accuracy_temp = accuracy_score(y_test, prediction)*100
    print("accuracy: {}%".format(round(accuracy_temp,2)))
    
    if(accuracy_temp>accuracy):
        accuracy=accuracy_temp
        # dumping whole model into pkl file
        pickle.dump(model_RF,open('model_RF.pkl','wb'))
    
    # cunfusion matrix
    print(metrics.classification_report(y_test, prediction))
    print(metrics.confusion_matrix(y_test, prediction))