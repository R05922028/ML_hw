import numpy as np
import collections
import sys
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing import sequence

test = []
ID_test = []
X_test = []
X_train = []
Y_train = []
X_train_semi = []
Y_train_semi = []
cnt_train_semi = 0
cnt_train = 0
cnt_test = 0

train_data_semi = "/home/yao/workspace/ML_data/hw4/training_nolabel.txt"
train_data = "/home/yao/workspace/ML_data/hw4/training_label.txt"
test_data = "/home/yao/workspace/ML_data/hw4/testing_data.txt"
prediction_data = "prediction_semi.csv"


## Read Semi Train Data
fin = open(train_data_semi, 'r')
for line in fin:
  split = line.split(' ')
  split[len(split)-1] = split[len(split)-1][:len(split[len(split)-1])-1] #去除換行
  X_train_semi.append([])
  for i in range(len(split)):
    X_train_semi[cnt_train_semi].append(split[i]) 
  cnt_train_semi = cnt_train_semi + 1
#print(X_trailabel)
#print(X_train[0])
fin.close()
X_train_semi = X_train_semi[:250000]


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
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    if X_train[i][j] not in dic:
      dic[X_train[i][j]] = word_index
      word_index = word_index + 1
for i in range(len(X_train_semi)):
  for j in range(len(X_train_semi[i])):
    if X_train_semi[i][j] not in dic:
      dic[X_train_semi[i][j]] = word_index
      word_index = word_index + 1


## generate vec
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    X_train[i][j] = dic[X_train[i][j]]
  #print(X_train[i])            
for i in range(len(X_train_semi)):
  for j in range(len(X_train_semi[i])):
    X_train_semi[i][j] = dic[X_train_semi[i][j]]
for i in range(len(X_test)):
  for j in range(len(X_test[i])):
    if X_test[i][j] in dic:
      X_test[i][j] = dic[X_test[i][j]]
    else:
      X_test[i][j] = 0


X_train_semi = sequence.pad_sequences(X_train_semi, maxlen = 100)      
X_test = sequence.pad_sequences(X_test, maxlen = 100)      

model = load_model('model.h5')
prediction = model.predict(X_train_semi)
#Testing
for x in range(len(X_train_semi)):                                               
  if prediction[x]>=0.8:
    Y_train_semi.append(1)
  else:
    Y_train_semi.append(0)



 
#print(X_train)

model = Sequential()
model.add(Embedding(len(dic), 128))
model.add(LSTM(128, dropout = 0.3, recurrent_dropout=0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_semi, Y_train_semi, batch_size=128, epochs=2, validation_split=0.1)

model.save('model_semi.h5')

model = load_model('model_semi.h5')

prediction = model.predict(X_test)
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
