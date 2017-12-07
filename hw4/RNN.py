import numpy as np
import collections
import sys
import csv
import keras
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

test = []
ID_test = []
X_test = []
X_train = []
Y_train = []
cnt_train = 0
cnt_test = 0

train_data = "/home/yao/workspace/ML_data/hw4/training_label.txt"
test_data = "/home/yao/workspace/ML_data/hw4/testing_data.txt"
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
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    if X_train[i][j] not in dic:
      dic[X_train[i][j]] = word_index
      word_index = word_index + 1
#print(dic)


## Bag of Words(BOW)
#print([ind for ind, v in enumerate(X_train[0]) if v=='dfsda'])
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    X_train[i][j] = dic[X_train[i][j]]
  #print(X_train[i])            
for i in range(len(X_test)):
  for j in range(len(X_test[i])):
    if X_test[i][j] in dic:
      X_test[i][j] = dic[X_test[i][j]]
    else:
      X_test[i][j] = 0

 
X_train = sequence.pad_sequences(X_train, maxlen = 100)      
X_test = sequence.pad_sequences(X_test, maxlen = 100)      
#print(X_train)

model = Sequential()
model.add(Embedding(len(dic), 256, embeddings_initializer=initializers.random_normal(stddev=1)))
#model.add(LSTM(128),dropout=0.3, recurrent_dropout=0.2)
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.2)))
model.add(Dense(units=1, activation='sigmoid'))
earlystop = EarlyStopping(monitor='val_acc', patience=1, verbose=1, mode='max')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=4, validation_split = 0.1, callbacks=[earlystop])

model.save('model.h5')

model = load_model('model.h5')


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
