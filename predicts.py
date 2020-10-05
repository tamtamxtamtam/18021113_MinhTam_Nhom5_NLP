import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

from vncorenlp import VnCoreNLP
from sklearn.feature_extraction.text import CountVectorizer #vector
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
print(file.sample(5))
#x = file['text'].head(1)
#print(line)


stopword = stopword()
print(stopword)
X_train = file['text']
y_train = file['label']
X_test = test['text']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape)
tfidf = TfidfVectorizer(stop_words=stopword)
#tfidf = TfidfVectorizer()
clf = Pipeline([('vect', tfidf),
                ('clf', LinearSVC())])
clf.fit(X_train, y_train)
print(y_train.head(5))
predictions = clf.predict(X_test)
testing = open('predicts.txt','w',encoding='utf-8')
print(predictions)

for i in predictions:
    testing.write(i+'\n')

