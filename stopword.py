import numpy as np
import seaborn as sns
import re


import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

from vncorenlp import VnCoreNLP
from collections import Counter

test = open('trainning.txt','r',encoding='utf-8')
annotator = VnCoreNLP("C:/Users/acer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

def stopword():
    s = open('vietnamese-stopwords-dash.txt','r',encoding='utf-8')
    list = []
    while True:
        text = s.readline()

        if not text:
            break
        text = re.sub('\n','',text)
        list.append(text)
    return list

vocab = Counter()
for i in range(5000):
    print(i)
    z = test.readline()
    print(z)
    k = annotator.tokenize(z)
    print(k)
    np.set_printoptions(precision=2)
    np.set_printoptions(linewidth=np.inf)

    for sens in k:
        for letter in sens:
            vocab[letter] += 1

vocab_reduced = Counter()
'''
#lọc ra các stopword
for w,c in vocab.items():
    if w in stopword():
        vocab_reduced[w]=c
'''
print(vocab.most_common(100))
