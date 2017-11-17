import numpy as np
import math
import sys

X_train = []
Y_train = []
X_test = []
num_attr = 106
weights = np.zeros(num_attr)
train_raw = sys.argv[1]
test_raw = sys.argv[2]
X_train_data = sys.argv[3]
Y_train_data = sys.argv[4]
X_test_data = sys.argv[5]
prediction_data = sys.argv[6]

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
    return 1 / (1 + math.exp(-x))

mean_high = np.zeros(num_attr)
mean_low = np.zeros(num_attr)
cnt_high = 0
cnt_low = 0
for i in range(len(Y_train)):
	if(Y_train[i] == 1):
		mean_high = mean_high + X_train[i]
		cnt_high = cnt_high + 1
	else:
		mean_low = mean_low + X_train[i]
		cnt_low = cnt_low + 1
mean_high = mean_high/cnt_high
mean_low = mean_low/cnt_low

sigma_high = np.zeros((106,106))
sigma_low = np.zeros((106,106))
sig_high = 0
sig_low = 0
for i in range(len(Y_train)):
	if(Y_train[i] == 1):
		sig_high = sig_high+1
		sigma_high =  sigma_high + (np.dot((X_train[i].reshape(106,1) - mean_high.reshape(106,1)), (np.transpose(X_train[i].reshape(106,1) - mean_high.reshape(106,1))))) 
	else:
		sig_low = sig_low + 1
		sigma_low =  sigma_low + (np.dot((X_train[i].reshape(106,1) - mean_low.reshape(106,1)), (np.transpose(X_train[i].reshape(106,1) - mean_low.reshape(106,1))))) 
sigma_high = sigma_high/sig_high
sigma_low = sigma_low/sig_low

sigma_final = np.zeros((106,106))
sigma_final = (sig_high/(len(Y_train)) * sigma_high )+ (sig_low/(len(Y_train)) * sigma_low)



#print(sigma_final)

#testing 
ans = []
mean_sub_pinv = np.dot(np.transpose((mean_high.reshape(106,1)-mean_low.reshape(106,1))),np.linalg.pinv(sigma_final))
mean_high_pinv = np.dot(np.dot(np.transpose(mean_high.reshape(106,1)),np.linalg.pinv(sigma_final)), mean_high.reshape(106,1))
mean_low_pinv = np.dot(np.dot(np.transpose(mean_low.reshape(106,1)),np.linalg.pinv(sigma_final)), mean_low.reshape(106,1))
ln = np.log(sig_high/sig_low)
for x in range(len(X_test)):                                               
	output = sigmoid(np.dot(mean_sub_pinv,X_test[x].reshape(106,1))- 0.5 * mean_high_pinv + 0.5 * mean_low_pinv + ln)
	if (float(output) >= 0.5):
		ans.append(1)
	else:
		ans.append(0)
	fout = open(prediction_data,'w')
	fout.write('id,label\n')
	for cnt in range(len(ans)):
		fout.write(str(cnt+1)+','+str(ans[cnt])+'\n')
