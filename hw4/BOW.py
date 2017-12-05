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
word_cnt = []
for i in range(len(X_train)):
  for j in range(len(X_train[i])):
    if X_train[i][j] not in dic:
      word_cnt.append(1)
      dic[X_train[i][j]] = word_index
      word_index = word_index + 1
    else:
      word_cnt[int(dic[X_train[i][j]])-1] += 1

dic_final = {}
final_cnt = 1
for j in dic:
  if word_cnt[int(dic[j])-1] > 1:
    dic_final[j] = final_cnt
    final_cnt = final_cnt + 1



## Bag of Words(BOW)
X_train_bow = []
X_test_bow = []
#print([ind for ind, v in enumerate(X_train[0]) if v=='are'])
for i in range(len(X_train)):
  X_train_bow.append([])
  for cnt in range(len(dic_final)):
    X_train_bow[i].append(0)
  for j in range(len(X_train[i])):
    if X_train[i][j] in dic_final:
      index = int(dic_final[X_train[i][j]])-1
      X_train_bow[i][index] += 1

     
for i in range(len(X_test)):
  X_test_bow.append([])
  for cnt in range(len(dic_final)):
    X_test_bow[i].append(0)
  for j in range(len(X_test[i])):
    if X_test[i][j] in dic_final:
      X_test_bow[i][int(dic_final[X_test[i][j]])-1] += 1


 
#X_train = sequence.pad_sequences(X_train, maxlen = 100)      
#X_test = sequence.pad_sequences(X_test, maxlen = 100)      

X_train_bow = np.array(X_train_bow).astype(np.int) 
X_test_bow = np.array(X_test_bow).astype(np.int) 


model = Sequential()
model.add(Dense(1, activation='sigmoid', input_dim = len(dic_final)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_bow, Y_train, batch_size=128, epochs=2)

model.save('model_bow.h5')

model = load_model('model_bow.h5')


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

