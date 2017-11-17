from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import pickle as pkl

train = []
test = []
X_train = []
Y_tmp = []
X_test = []
ID_test = []
train_data = sys.argv[1]


#Reading train data

fin = open(train_data, encoding='big5')
for line in fin:
    split = line.split(',')
    train.append(split)
fin.close()
for i in range(1,len(train)):
    Y_tmp.append(train[i][0])
    split = train[i][1].split(' ')
    X_train.append(split)

X_train = np.array(X_train).astype(np.float)   
X_train /= 255   
Y_tmp = np.array(Y_tmp).astype(np.float)   
Y_train = np.zeros((len(Y_tmp), 7))

for i in range(len(Y_tmp)):
    Y_train[i][int(Y_tmp[i])]=1

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
#X_validation = X_train[25000:]
#X_train = X_train[:25000]
#Y_validation = Y_train[25000:]
#Y_train = Y_train[:25000]
#print(X_validation.shape)
#print(Y_validation.shape)

X_train = X_train.reshape((len(X_train)), 48, 48, 1)
X_validation = X_validation.reshape(len(X_validation), 48, 48, 1)

"""
fout = open('X_validation.pkl', 'wb')
pkl.dump(X_validation, fout)
fout.close()
fout = open('Y_validation.pkl', 'wb')
pkl.dump(Y_validation, fout)
fout.close()
"""

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.15))
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.15))
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(48, 48, 1), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train, Y_train, batch_size=100, epochs=50, validation_split=0.1, shuffle=True)
model.fit_generator(datagen.flow(X_train,Y_train, batch_size=128),steps_per_epoch=len(X_train)/64, validation_data = datagen.flow(X_validation, Y_validation, batch_size=128), validation_steps=len(X_validation)/64, epochs=100)

model.save('Train_model_final')


