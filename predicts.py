import numpy as np
import pandas as pd
import seaborn as sns
import re


import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

#lưu stopword vào 1 list
def stopword():
    s = open('stopwords1.txt','r',encoding='utf-8')
    list = []
    while True:
        text = s.readline()

        if not text:
            break
        text = re.sub('\n','',text)
        list.append(text)
    return list
# Input
file = pd.read_csv("r.csv")
test = pd.read_csv("test1.csv")


stopword = stopword()
print(stopword)
X_train = file['text']
y_train = file['label']
X_test = test['text']
print(X_train.shape, X_test.shape, y_train.shape)
tfidf = TfidfVectorizer(stop_words=stopword)
clf = Pipeline([('vect', tfidf),
                ('clf', LinearSVC())])
clf.fit(X_train, y_train)
print(y_train.head(5))
predictions = clf.predict(X_test)
testing = open('predicts.txt','w',encoding='utf-8')
print(predictions)

for i in predictions:
    testing.write(i+'\n')

