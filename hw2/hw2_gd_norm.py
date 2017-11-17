import numpy as np
import math

X_train = []
Y_train = []
X_test = []
num_attr = 106
weights = np.zeros(num_attr)
iteration = 5000


bias = 0
lr = 0.0001 
lr_b = 0.0
lr_w = 0.0
#Reading train data

fin = open('X_train', encoding='big5')
for line in fin:
    split = line.split(',')
    X_train.append(split)
fin.close()

X_train = np.array(X_train[1:]).astype(np.float)

fin = open('Y_train',encoding='big5')
for line in fin:
    Y_train.append(line)
fin.close()

Y_train = np.array(Y_train[1:]).astype(np.float)


#Reading test data
fin = open('X_test', encoding='big5')
for line in fin:
    split = line.split(',')
    X_test.append(split)
fin.close()

X_test = np.array(X_test[1:]).astype(np.float)


#sigmoid 
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
#normalization

mean = np.zeros(num_attr)
variance = np.zeros(num_attr)
for j in range(len(Y_train)):
	mean = mean + X_train[j] 	
mean = mean/len(Y_train)

for j in range(len(Y_train)):
	variance = variance + (X_train[j]-mean) ** 2 
variance = np.sqrt(variance / len(Y_train))


for it in range(len(Y_train)):
	for j in range(len(X_train[0])):
		if variance[j] == 0 : 
			X_train[it][j] = (X_train[it][j] - mean[j])
		else : 
			X_train[it][j] = (X_train[it][j] - mean[j]) / variance[j]


for it in range(len(X_test)):
	for j in range(len(X_test[0])):
		if variance[j] == 0 : 
			X_test[it][j] = (X_test[it][j] - mean[j])
		else : 
			X_test[it][j] = (X_test[it][j] - mean[j]) / variance[j]


#Training iteration
for cnt in range(iteration):
    print("iter: ",cnt)
    loss = 0.0
    b_grad = 0.0
    w_grad = np.zeros(num_attr)
    correct = 0
    for i in range(len(X_train)):
        output = sigmoid(bias + (np.dot(X_train[i], weights)))
        b_grad = b_grad - (Y_train[i] - output)
        w_grad = w_grad - (Y_train[i] - output) * X_train[i]
        if(output >= 0.5 and Y_train[i] == 1):
            correct = correct + 1
        elif(output<0.5 and Y_train[i]==0):
            correct = correct+1
    print("Accuracy :",correct/len(X_train))
    bias = bias - lr*b_grad
    weights = weights -lr * w_grad
ans = []
#Testing
for x in range(len(X_test)):                                               
    output = sigmoid(bias + (np.dot(X_test[x], weights)))
    if (output >= 0.5):
        ans.append(1)
    else:
        ans.append(0)
    fout = open('hw2_gd_norm.csv','w')
    fout.write('id,label\n')
    for cnt in range(len(ans)):
        fout.write(str(cnt+1)+','+str(ans[cnt])+'\n')

