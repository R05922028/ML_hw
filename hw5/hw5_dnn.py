import numpy as np
import collections
import sys
import csv
import keras
import json
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

train_data= sys.argv[1]
test_data= sys.argv[2]
prediction_data= sys.argv[3]


## Read Train Data
fin = open(train_data, 'r')
for line in fin:
  split = line.split(',')
  user_ID.append([split[1]])
  movie_ID.append([split[2]])
  ranting.append([split[3]])
fin.close()
user_ID = np.array(user_ID[1:])
movie_ID = np.array(movie_ID[1:])
ranting = np.array(ranting[1:])

## Read Test Data
fin = open(test_data, 'r')
for line in fin:
  split = line.split(',')
  user_ID_test.append([split[1]])
  movie_ID_test.append([split[2]])
fin.close()
user_ID_test = np.array(user_ID_test[1:])
movie_ID_test = np.array(movie_ID_test[1:])
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
num_users = 6040
num_movies = 3952

'''
users = Input(shape=[1])
movies = Input(shape=[1])


user_embedding = Embedding(input_dim=num_users, output_dim=1024,embeddings_initializer=keras.initializers.random_normal(stddev=0.1))(users)
user_embedding = Flatten()(user_embedding)
movie_embedding = Embedding(input_dim=num_movies, output_dim=1024,embeddings_initializer=keras.initializers.random_normal(stddev=0.1))(movies)
movie_embedding = Flatten()(movie_embedding)
#users_bias = Embedding(input_dim=num_users, 1, embeddings_initializer=keras.initializers.random_normal(stddev=0.1))(users)
#movies_bias = Embedding(input_dim=num_movies, 1, embeddings_initializer=keras.initializers.random_normal(stddev=0.1))(movies)
concate = Concatenate()([user_embedding, movie_embedding])
#concate
hidden = Dense(256, activation='relu')(concate)
hidden = Dropout(0.7)(hidden)
output = Dense(1, activation='relu')(hidden)


model = Model(inputs=[users, movies], outputs=[output])
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

 
model.fit([user_ID, movie_ID],ranting,epochs=18,batch_size=256)
model.save('model_hw5.h5')
'''
model = load_model('model_hw5.h5')
prediction = model.predict([user_ID_test, movie_ID_test], batch_size=256)
fout = open(prediction_data,'w')
fout.write('TestDataID,Rating\n')
for x in range(len(prediction)):
  fout.write(str(x+1)+','+str(prediction[x][0])+'\n')
fout.close()
  
