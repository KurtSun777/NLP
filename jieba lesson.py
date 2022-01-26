#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba


# ## 斷詞

# In[2]:


sentence = '彰化某大學應用日語系男大生前年3月在網路論壇公開留言批評新來的魏姓系主任超色，常以不懷好意眼神打量女同學，令人不舒服，更以不雅字眼謾罵批評，系主任不堪名譽受損提告附帶民事損害賠償100萬元。刑事部分去年9月遭彰化地院依加重誹謗罪判處拘役30日，得易科罰金；民事部分則判賠15萬元。'
seg1 = jieba.cut(sentence, cut_all = True)
print('全模式：', '|'.join(seg1))


# In[3]:


sentence = '彰化某大學應用日語系男大生前年3月在網路論壇公開留言批評新來的魏姓系主任超色，常以不懷好意眼神打量女同學，令人不舒服，更以不雅字眼謾罵批評，系主任不堪名譽受損提告附帶民事損害賠償100萬元。刑事部分去年9月遭彰化地院依加重誹謗罪判處拘役30日，得易科罰金；民事部分則判賠15萬元。'
seg2 = jieba.cut(sentence, cut_all = False)
print('精準模式：', '|'.join(seg2))


# In[4]:


seg3 = jieba.cut_for_search(sentence)
print('搜尋引擎模式：', '|'.join(seg3))


# In[18]:


f = open('mydict.txt', 'w', encoding = 'utf-8')
f.write('日語系\n男大生\n不舒服\n易科\n罰金\n判賠\n謾罵\n判處')  # 監督式學習，bert是非監督式
f.close()

jieba.load_userdict('mydict.txt')
seg4 = jieba.cut_for_search(sentence)
print('定義詞庫：', '|'.join(seg4))


# In[5]:


sentence = '彰化某大學應用日語系男大生前年3月在網路論壇公開留言批評新來的魏姓系主任超色，常以不懷好意眼神打量女同學，令人不舒服，更以不雅字眼謾罵批評，系主任不堪名譽受損提告附帶民事損害賠償100萬元。刑事部分去年9月遭彰化地院依加重誹謗罪判處拘役30日，得易科罰金；民事部分則判賠15萬元。'
seg1 = jieba.cut(sentence, cut_all = True)
print('全模式：', '|'.join(seg1))


# ## 詞性標記

# In[6]:


import jieba.posseg


# In[14]:


seg4 = jieba.posseg.cut(sentence)
jieba.load_userdict('mydict.txt')
for w , pos in seg4:
    print(w, pos, end = ', ')  # end = ',' 把直變橫


# ## 停用詞

# In[9]:


from collections import Counter
from pprint import pprint


# In[15]:


seg2 = jieba.lcut(sentence, cut_all = False)
# print(seg2)

print('===================================================================================')

stop1 = open('zh_stop.txt', 'r', encoding = 'utf-8')
lines = stop1.readlines()
stop_word = []
for line in lines:
    line = line.replace('\n', '')
    stop_word.append(line)
    
rest_word = list(filter(lambda a : a not in stop_word, seg2))
print(rest_word)


# ## Count and TF

# In[16]:


count_dict = Counter(rest_word)
print(count_dict)

def TF(wordDict, word):
    tfDict = {}
    wordLen = len(wordDict)
    for word, count in wordDict.items():
        tfDict[word] = count / wordLen
    return tfDict

TFDict = TF(count_dict, rest_word)
pprint(TFDict)


# In[ ]:




