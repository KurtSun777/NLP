#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import jieba.posseg
import jieba.analyse
from collections import Counter
from pprint import pprint
from wordcloud import WordCloud
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import ckpe


# In[ ]:


# 讀取檔案變成字串

with open('./data1/0.txt', 'r', encoding = 'utf-8') as f:
    words = f.read()
print(type(words))
# 讀取分詞庫

jieba.load_userdict('mydict.txt')

# 分詞

seg1 = jieba.lcut(words, cut_all = True)

print('分詞：', '|'.join(seg1))

print('word', type(words))
print('seg1', type(seg1))

# 開啟停用詞庫

stop1 = open('stopword.txt', 'r', encoding = 'utf-8')
lines = stop1.readlines()
stop_word = []
for line in lines:
    line = line.replace('\n', '')
    stop_word.append(line)
    
rest_word = list(filter(lambda a : a not in stop_word, seg1))

print('rest_word', type(rest_word))
print('分詞＋停用詞: ', '|'.join(rest_word))
# print('\n, ========================================================================, \n')

# 算出詞數量

count_dict = Counter(rest_word)

# print(count_dict)
# print('\n, ========================================================================, \n')

# TF - term frequency 詞頻

def TF(wordDict, word):
    tfDict = {}
    wordLen = len(wordDict)
    for word, count in wordDict.items():
        tfDict[word] = count / wordLen
    return tfDict

TFDict = TF(count_dict, rest_word)

# pprint(TFDict)
# print('\n, ========================================================================, \n')

# analyse 取關鍵字 extract and textrank

keywords = jieba.analyse.extract_tags(' '.join(rest_word), topK = 5, withWeight = True, allowPOS = ())

pprint(keywords)
print('\n, ========================================================================, \n')

keywords1 = jieba.analyse.textrank(' '.join(rest_word), topK = 5, withWeight = True)

# pprint(keywords1)
# print('\n, ========================================================================, \n')

# 驗證關鍵字，統計出現次數

for word, count in Counter(rest_word).most_common(10):
    print(word, count)

# 關鍵詞 (安裝ckpe)
# git clone https://github.com/dongrixinyu/chinese_keyphrase_extractor
# cd ./chinese_keyphrase_extractor
# pip install .
# 確認環境 pip list

keyphraseExtractor = ckpe.ckpe()
keyPhrases = keyphraseExtractor.extract_keyphrase(words)
pprint(keyPhrases)

# 文本分類(監督式學習)






# In[ ]:




