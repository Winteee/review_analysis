
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
# 参考代码：https://github.com/Winteee/KerasTest/blob/master/KerasLSTM.py
# 这是代码的文字解释：http://spaces.ac.cn/index.php/archives/3414/comment-page-1

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import jieba.analyse
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


# In[2]:


print("————————语料预处理：数据导入————————")
reviewfile=pd.read_csv('review30000.csv') #读取训练语料完毕

reviewdata = pd.DataFrame()
reviewdata["reviewbody"]= reviewfile["b.reviewbody"]
reviewdata["score1"]= reviewfile["口味评分"]
reviewdata["score2"]= reviewfile["环境评分"]
reviewdata["score3"]= reviewfile["服务评分"]
reviewdata = reviewdata.dropna(axis=0,how='any') #去掉所有有缺失值的行
print("reviewdata的数据格式为：",reviewdata.columns,reviewdata.index)
print("——————————over——————————")


# In[3]:


def cutwords(review_data):
    cw = lambda x: list(jieba.cut(x)) #定义分词函数
    review_data['review_cut_words'] = review_data["reviewbody"].apply(cw)
    print("——————————分词结束——————————")
    return review_data

def cleanwords(review_data):
    delwordslist = "，/,/。/的/！/～/、/（/）/ /./。/吧".split("/")
    if "review_cut_words" in review_data.columns:
        for i in review_data.index:
            one = review_data.review_cut_words[i]
            for k in one:
                if k in delwordslist:
                    one.remove(k)
            i +=1
        return review_data
    
    elif "review_cut_words" not in review_data.columns:
        print("cleanwords运行出错，请先运行cutwords函数以获取review_cut_words")

def countwords(review_data):
    w = []
    for i in review_data["review_cut_words"]:
        w.extend(i)
    reviewdict = pd.DataFrame(pd.Series(w).value_counts(),columns = ["counts"])
    print("——————————词频统计结束——————————")
    return reviewdict

def arraywords(review_data,reviewdict):
    reviewdict['id']=list(range(1,len(reviewdict)+1))
    get_sent = lambda x: list(reviewdict["counts"][x])
    maxlen = 50
    review_data['sent'] = review_data['review_cut_words'].apply(get_sent) 
    review_data['sent'] = list(sequence.pad_sequences(review_data['sent'], maxlen=maxlen))
    print("——————————词向量化结束——————————")
    return review_data

#def markwords(review_data):
 #   review_data["totalscore"]= review_data.score1+review_data.score2+review_data.score3 
  #  review_data["pos_or_neg"] = pd.Series()
    #print(review_data[:2])
    #print(len(review_data.index))
   # for i in range(len(review_data.index)):
    #    if review_data["totalscore"][i]>11:
     #       review_data["pos_or_neg"][i] = 1
      #  elif review_data["totalscore"][i]<4:
       #     review_data["pos_or_neg"][i] = 0
        #i +=1
    #print("——————————标注结束——————————")
    #return review_data


def markwords(review_data):
    review_data["totalscore"]= review_data.score1+review_data.score2+review_data.score3 
    review_data["pos_or_neg"] = pd.Series()
    pos = review_data[review_data.totalscore>11]
    pos["pos_or_neg"]=1
    neg = review_data[review_data.totalscore<4]
    neg["pos_or_neg"] = 0
    review_data = pos.append(neg)
    print("——————————标注结束——————————")
    return review_data


# In[4]:


def train_data_input(review_data):
    a1 = cutwords(review_data)
    a2 = cleanwords(a1)
    a3 = countwords(a2)
    a4 = arraywords(a2,a3) #a4.index:[reviewbody,score1,score2,score3,review_cut_words,sent]
    a5 = markwords(a4)
    return a5

def predict_data_input(review_data):
    a1 = cutwords(review_data)
    a2 = cleanwords(a1)
    a3 = countwords(a2)
    a4 = arraywords(a2,a3) #a4.index:[reviewbody,score1,score2,score3,review_cut_words,sent]
    return a4


# In[5]:


a0 = reviewdata[0:20000]
a1 = cutwords(a0)
a2 = cleanwords(a1)
reviewdict = countwords(a2)
a4 = arraywords(a2,reviewdict) #a4.index:[reviewbody,score1,score2,score3,review_cut_words,sent]
a5 = markwords(a4)
traindata = a5
traindata.to_csv("traindata.csv",index=False,header=True)
#predictdata = predict_data_input(reviewdata[20000:20100])
print("traindata information————————————————————————————————————————————————————")
print(traindata[:3],traindata.columns,traindata.index)

#print("predictdata information————————————————————————————————————————————————————")
#print(predictdata[:3],predictdata.columns,predictdata.index)


# In[6]:


traindata = traindata.dropna(axis=0,how='any') #去掉所有有缺失值的行
print(traindata.index)
x = np.array(list(traindata['sent']),dtype = object)[::2]  #x：训练数据的Numpy数组（如果模型有单个输入）或Numpy数组列表（如果模型有多个输入）。
y = np.array(list(traindata['pos_or_neg']),dtype = object)[::2] #y：目标（标签）数据的Numpy数组（如果模型具有单个输出）或Numpy数组列表
xt = np.array(list(traindata['sent']),dtype = object)[1::2] #测试集
yt = np.array(list(traindata['pos_or_neg']),dtype = object)[1::2]
xa = np.array(list(traindata['sent']),dtype = object) #全集
ya = np.array(list(traindata['pos_or_neg']),dtype = object)

print("————————模型构建中————————")
model = Sequential()
model.add(Embedding(len(reviewdict)+1, 256))
model.add(LSTM(output_dim=32, activation='sigmoid', inner_activation='hard_sigmoid')) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(input_dim = 32, output_dim = 1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) #class_mode="binary"
print("————————模型构建完成————————")
model.fit(x, y, batch_size=32, nb_epoch=5, validation_data=(xt, yt),validation_steps=None) #训练时间为若干个小时

classes = model.predict_classes(xa)
print(classes)
score = model.evaluate(xt, yt, verbose=1) #evaluate用来测试函数的性能
print ("Test accuracy = :",score)
#Keras文档使用三组不同的数据：培训数据，验证数据和测试数据。
#(training data)训练数据用于优化模型参数。(validation data)验证数据用于对元参数进行选择，例如时期的数量。
#在用最佳元参数优化模型之后，使用测试数据(test data)来获得对模型性能的合理估计。



# In[10]:


b0 = reviewdata[0:200]
b1 = cutwords(a0)
b2 = cleanwords(a1)
reviewdict_b = countwords(a2)
b4 = arraywords(a2,reviewdict_b) #a4.index:[reviewbody,score1,score2,score3,review_cut_words,sent]
predictdata = b4

xb = np.array(list(predictdata['sent']),dtype = object)
predict_result = model.predict(xb, batch_size=32, verbose=0)
print("predict = :",predict_result)
print("predict type = :",type(predict_result))
p = predict_result.tolist()
predict_data = pd.DataFrame()
predict_data["prediction"]= pd.Series()
predict_data["prediction"] = p
predict_data["review_cut_words"]=predictdata["review_cut_words"]
predict_data.to_csv("predict_data.csv",index=False,header=True)

