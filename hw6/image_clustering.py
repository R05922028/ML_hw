import numpy as np
import pandas as pd
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from sklearn.cluster import KMeans

image_data = sys.argv[1]
image = np.load(image_data)
image = image / 255
#pca = PCA(n_components = 28)
#newImage = pca.fit_transform(image)

test_data = sys.argv[2]
#Read testing case
fin = pd.read_csv(test_data)
IDs, image_1, image_2 = np.array(fin['ID']), np.array(fin['image1_index']), np.array(fin['image2_index'])

#print(image.shape)
#print(newImage.shape)
encoding_dim = 64


#input_img = Input(shape=(784,))
#encoded = Dense(1024, activation='relu')(input_img)
#encoded = Dense(512, activation='relu')(encoded)
#encoded = Dense(256, activation='relu')(encoded)
#encoded = Dense(encoding_dim)(encoded)

#decoded = Dense(256, activation='relu')(encoded)
#decoded = Dense(512, activation='relu')(decoded)
#decoded = Dense(1024, activation='relu')(decoded)
#decoded = Dense(784, activation='tanh')(decoded)

#model = Model(input=input_img, output = decoded)
#encoder = Model(input=input_img, output = encoded)
#model.compile(optimizer='adam', loss='mse')

#early = EarlyStopping(monitor='loss', patience=3, verbose=1, mode='auto')
#checkpoint = ModelCheckpoint('autoencoder.h5', monitor='loss', save_best_only=True, mode='auto')
#model.fit(image, image, epochs=200, batch_size=512, shuffle=True)

#encoder.save('img_cluster.h5')
encoder = load_model('img_cluster.h5')

image_en = encoder.predict(image)
image_en = image_en.reshape(image_en.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(image_en)
#print(kmeans.labels_)

predict_data = sys.argv[3]
fout = open(predict_data,'w')
fout.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, image_1, image_2):
	p1 = kmeans.labels_[i1]
	p2 = kmeans.labels_[i2]
	if p1 == p2:
		pred = 1
	else:
		pred = 0
	fout.write("{},{}\n".format(idx, pred))
fout.close()


