import numpy as np
import math
import sys

X_train = []
Y_train = []
X_test = []
num_attr = 106
weights = np.zeros(num_attr)
iteration = 400
train_raw = sys.argv[1]
test_raw = sys.argv[2]
X_train_data = sys.argv[3]
Y_train_data = sys.argv[4]
X_test_data = sys.argv[5]
prediction_data = sys.argv[6]


bias = np.zeros(1)
lr = 0.1 
lr_b = 0.0
lr_w = 0.0
#Reading train data

fin = open(X_train_data, encoding='big5')
for line in fin:
    split = line.split(',')
    X_train.append(split)
fin.close()

X_train = np.array(X_train[1:]).astype(np.float)

fin = open(Y_train_data,encoding='big5')
for line in fin:
    Y_train.append(line)
fin.close()

Y_train = np.array(Y_train[1:]).astype(np.float)


#Reading test data
fin = open(X_test_data, encoding='big5')
for line in fin:
    split = line.split(',')
    X_test.append(split)
fin.close()

X_test = np.array(X_test[1:]).astype(np.float)


#sigmoid 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Training iteration
for cnt in range(iteration):
    print("iter: ",cnt)
    loss = 0.0
    b_grad = 0.0
    correct = 0
    w_grad = np.zeros(num_attr)
    for i in range(len(X_train)):
        output = sigmoid(bias[0] + (np.dot(X_train[i], weights)))
        b_grad = b_grad - (Y_train[i] - output)
        w_grad = w_grad - (Y_train[i] - output) * X_train[i]
        loss = loss - (Y_train[i]*np.log(output) + (1-Y_train[i]) * np.log(1-(output)))
        if(output >= 0.5 and Y_train[i] == 1):
            correct = correct + 1
        elif(output<0.5 and Y_train[i]==0):
            correct = correct+1
    print("Accuracy :",correct/len(X_train))
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    bias[0] = bias[0] - lr/np.sqrt(lr_b)*b_grad
    weights = weights -lr/np.sqrt(lr_w) * w_grad
    print("Loss :",math.sqrt(loss/len(X_train)))


ans = []
#Testing
for x in range(len(X_test)):                                               
	output = sigmoid(bias[0] + (np.dot(X_test[x], weights)))
	if (output >= 0.5):
		ans.append(1)
	else:
		ans.append(0)
	fout = open(prediction_data,'w')
	fout.write('id,label\n')
	for cnt in range(len(ans)):
		fout.write(str(cnt+1)+','+str(ans[cnt])+'\n')

