from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import sys

test = []
X_test = []
ID_test = []
test_data = sys.argv[1]
prediction_data = sys.argv[2]



#Reading test data

fin = open(test_data, encoding='big5')
for line in fin:
    split = line.split(',')
    test.append(split)
fin.close()
for i in range(1,len(test)):
    ID_test.append(test[i][0])
    split = test[i][1].split(' ')
    X_test.append(split)

X_test = np.array(X_test).astype(np.float)   
ID_test = np.array(ID_test).astype(np.float)   
X_test /= 255

X_test = X_test.reshape(len(X_test), 48, 48, 1)


model = load_model('final_model.h5')
model.summary()


result = model.predict(X_test)

ans = []
#Testing
for x in range(len(X_test)):                                               
    re = np.where(result[x]==np.max(result[x]))
    ans.append(re[0][0])
print(len(ans))
fout = open(prediction_data,'w')
fout.write('id,label\n')
for x in range(len(ans)):
    fout.write(str(x)+','+str(ans[x])+'\n')
fout.close()
