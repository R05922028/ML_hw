import numpy as np
import collections
import sys
import csv
import keras
from keras.utils import *
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing import sequence

test = []
ID_test = []
X_test = []
X_train = []
Y_train = []
cnt_train = 0
cnt_test = 0

train_data = "/home/xpoint/ML2017FALL/hw4/training_label.txt"
test_data = "/home/xpoint/ML2017FALL/hw4/testing_data.txt"
prediction_data = "prediction.csv"


## Read Train Data
fin = open(train_data, 'r')
for line in fin:
  split = line.split(' ')
  Y_train.append(split[0])
  split[len(split)-1] = split[len(split)-1][:len(split[len(split)-1])-1] #去除換行
  X_train.append([])
  for i in range(len(split)-2):
    X_train[cnt_train].append(split[i+2]) 
  cnt_train = cnt_train + 1
#print(X_trailabel)
#print(Y_train[0])
fin.close()

## Read Test Data
fin = open(test_data, 'r')
for line in fin:
  split = line.split(',',1)
  ID_test.append(split[0])
  split = split[1].split(' ')
  split[len(split)-1] = split[len(split)-1][:len(split[len(split)-1])-1] #去除換行
  X_test.append([])
  for i in range(len(split)):
    X_test[cnt_test].append(split[i]) 
  cnt_test = cnt_test + 1
ID_test = ID_test[1:]
X_test = X_test[1:]
#print(ID_test)
#print(X_test)
fin.close()

## Build Dictionary
dic = {}    # Dictionary 
word_index = 1
word_cnt = {}
for i in range (len(X_train)):
  for j in range (len(X_train[i])):
    if(X_train[i][j] not in word_cnt):
      word_cnt[X_train[i][j]] = 1
    else:
      word_cnt[X_train[i][j]] = word_cnt[X_train[i][j]] + 1
for i in range (len(X_train)):
  for j in range (len(X_train[i])):	
    if(X_train[i][j] not in dic and word_cnt[X_train[i][j]]>=2):
      dic[X_train[i][j]] = word_index
      word_index = word_index + 1
dic['others_tmp'] = word_index
print(dic)


## Bag of Words(BOW)
X_train_bow = np.zeros((len(X_train), len(dic)))
#X_test_bow = np.zeros((len(X_test), len(dic_final)))
#print([ind for ind, v in enumerate(X_train[0]) if v=='are'])
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    if X_train[i][j] in dic:
      index = int(dic[X_train[i][j]])-1
      X_train_bow[i][index] += 1
    else:
      X_train_bow[i][len(dic)-1] += 1
     
#for i in range(len(X_test)):
#  for j in range(len(X_test[i])):
#    if X_test[i][j] in dic_final:
#      X_test_bow[i][int(dic_final[X_test[i][j]])-1] += 1

#print(dic_final)
#print(X_train)
#print(X_train_bow)
 
#X_train = sequence.pad_sequences(X_train, maxlen = 100)      
#X_test = sequence.pad_sequences(X_test, maxlen = 100)      

Y_train = keras.utils.to_categorical(Y_train, 2)



model = Sequential()
model.add(Dense(128, activation='relu', input_dim = len(dic)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_bow, Y_train, batch_size=128, epochs=2, validation_split=0.1)

model.save('model_bow.h5')

model = load_model('model_bow.h5')

'''
prediction = model.predict(X_test_bow)

ans = []
fout = open(prediction_data,'w')
#Testing
for x in range(len(X_test)):                                               
  if prediction[x]>=0.5:
    ans.append(1)
  else:
    ans.append(0)
print(len(ans))
fout.write('id,label\n')
for x in range(len(ans)):
    fout.write(str(x)+','+str(ans[x])+'\n')
fout.close()
'''
