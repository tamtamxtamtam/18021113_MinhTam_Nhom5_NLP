
import seaborn as sns
import csv
import re

from vncorenlp import VnCoreNLP
import warnings
warnings.filterwarnings('ignore')


sns.set_style("whitegrid")

#Xóa các ký tự đặc biệt và số, chuyển viết hoa thành viết thường
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[^\w\d\s]+', '', text)
    text = re.sub('[0-9]', '0', text)
    text = text.lower()
    return text


annotator = VnCoreNLP("C:/Users/acer/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')


file = open("testing.txt",'r',encoding='utf-8')



rfile = open("test1.csv", "a", encoding="utf-8")
csv_writer = csv.writer(rfile)
csv_writer.writerow(['text'])
count = 0
while True:
    count += 1
    line = file.readline()
    #print(line)
    if not line:
        break
    '''
    x = line.split("__")
    #print(x)
    x[1] = '__' + x[1] +'__'
    x[2] = preprocessor(x[2])
    word_list = annotator.tokenize(x[2])
    '''
    line = preprocessor(line)
    word_list = annotator.tokenize(line)
    y = ''
    #print(word_list)
    for s in word_list:
        for i in s:
            y += i + ' '
    print(y)
    '''
    x[2] = y
    print(x[1],x[2])
    csv_writer.writerow([x[1],x[2]])
    '''
    csv_writer.writerow([y])


