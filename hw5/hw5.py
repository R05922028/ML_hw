import numpy as np
import pandas as pd
import collections
import sys
import csv
import keras
from keras import initializers
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Dot, Add, Dropout, Concatenate, Reshape
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

user_ID = []
user_ID_test = []
movie_ID = []
movie_ID_test = []
ranting = []
num_users = 6040
num_movies = 3952

train_data= sys.argv[1]
test_data= sys.argv[2]
prediction_data= sys.argv[3]

## Read Train Data
fin = pd.read_csv(train_data)
fin = fin.values
np.random.shuffle(fin)
for i in range(len(fin)):
  user_ID.append(int(fin[i][1]))
  movie_ID.append(int(fin[i][2]))
  ranting.append(int(fin[i][3]))
user_ID = np.array(user_ID)
movie_ID = np.array(movie_ID)
ranting = np.array(ranting)

## Read Test Data
fin = open(test_data, 'r')
for line in fin:
  split = line.split(',')
  user_ID_test.append([split[1]])
  movie_ID_test.append([split[2]])
fin.close()
user_ID_test = np.array(user_ID_test[1:])
movie_ID_test = np.array(movie_ID_test[1:])

## normalize
mean = np.mean(ranting)
dev = np.std(ranting)
ranting = (ranting-mean) / dev

'''
UID = []
Gender = []
Age = []
Occupation = []
Zip = []
fin = open('users.csv', 'r')
for line in fin:
  split = line.split('::')
  UID.append([split[0]])
  Gender.append([split[1]])
  Age.append([split[2]])
  Occupation.append([split[3]])
  Zip.append([split[4]])
fin.close()
UID = np.array(UID[1:])
Gender = np.array(Gender[1:])
Age = np.array(Age[1:])
Occupation = np.array(Occupation[1:])
Zip = np.array(Zip[1:])

num_users = len(UID)
print(Age[2])

MID = []
Title = []
Genres = []
fin = open('movies.csv', 'r')
for line in fin:
  split = line.split('::')
  MID.append([split[0]])
  Title.append([split[1]])
  if len(split) == 3:
    Genres.append([split[2]])
  else:
    Genres.apapend([''])
fin.close()
MID = np.array(MID[1:])
Title = np.array(Title[1:])
Genres = np.array(Genres[1:])
print(Title[0])
'''
latent_dim = 16

users = Input(shape=[1])
movies = Input(shape=[1])


user_embedding = Embedding(num_users, latent_dim,embeddings_initializer='random_normal')(users)
user_embedding = Flatten()(user_embedding)
movie_embedding = Embedding(num_movies, latent_dim,embeddings_initializer='random_normal')(movies)
movie_embedding = Flatten()(movie_embedding)
users_bias = Embedding(num_users, 1, embeddings_initializer='zeros')(users)
users_bias = Flatten()(users_bias)
movies_bias = Embedding(num_movies, 1, embeddings_initializer='zeros')(movies)
movies_bias = Flatten()(movies_bias)
r_hat = Dot(axes=1)([user_embedding, movie_embedding])
r_hat = Add()([r_hat, users_bias, movies_bias])

#concate = Concatenate()([user_embedding, movie_embedding])
#hidden = Dense(256, activation='relu')(concate)
#hidden = Dropout(0.7)(hidden)
#output = Dense(1, activation='relu')(hidden)


model = Model([users, movies], r_hat)
model.compile(loss='mse', optimizer='adam')
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
 
model.fit([user_ID, movie_ID],ranting,epochs=100,batch_size=256, validation_split =0.01,verbose=1, callbacks=[early_stopping])
model.save('model_hw5_mf.h5')

model = load_model('model_hw5_mf.h5')
prediction = model.predict([user_ID_test, movie_ID_test], batch_size=256)
##normalize
prediction = prediction * dev + mean
fout = open(prediction_data,'w')
fout.write('TestDataID,Rating\n')
for x in range(len(prediction)):
  fout.write(str(x+1)+','+str(prediction[x][0])+'\n')
fout.close()
  
