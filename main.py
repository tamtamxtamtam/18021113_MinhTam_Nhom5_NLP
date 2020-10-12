import numpy as np
import pandas as pd
import seaborn as sns
import re


import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

from vncorenlp import VnCoreNLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

annotator = VnCoreNLP("C:/Users/acer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

stopword = stopword()
print(stopword)
X = file['text']
y = file['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tfidf = TfidfVectorizer(stop_words=stopword,sublinear_tf=True)
clf = Pipeline([('vect', tfidf),
                ('clf', LinearSVC())])
clf.fit(X_train, y_train)
print(y_train.head(5))
predictions = clf.predict(X_test)
print(predictions)
print('accuracy:',accuracy_score(y_test,predictions))
predictions = clf.predict(X_test)
print('accuracy:',accuracy_score(y_test,predictions))
print('confusion matrix:\n',confusion_matrix(y_test,predictions))
print('classification report:\n',classification_report(y_test,predictions))



