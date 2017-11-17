import numpy as np
import math
train = []
test = []
weights = np.zeros(5)
bias = 1.71453
lr = 10000
lr_b = 0.0
lr_w = 0.0
lr_w_2 = 0.0
iteration = 15000


#Reading test data

fin = open('train.csv',encoding='big5')
for line in fin:
    split = line.split(',')
    if(split[2] == "PM2.5"):
            train.extend(map(lambda x: float(x), split[3:]))
fin.close()

#Reading test data
fin = open('test.csv', encoding='big5')
for line in fin:
    split = line.split(',')
    if(split[1] == "PM2.5"):
            test.extend(map(lambda x: float(x), split[2:]))
fin.close()


# Training 
for i in range(iteration):
    print(i)
    b_grad = 0.0
    loss = 0
    w_grad = np.zeros(5)
    w_grad_2 = np.zeros(5)
    for j in range(len(train)-6):
        train_data = train[j:j+6]
        ideal = float(train_data[5])
        input_data = np.array(train_data)
        input_data = np.delete(input_data,5)
        b_grad = b_grad - 2.0*(ideal - bias - (weights*input_data).sum()-(weights_2*input_data*input_data).sum())*1.0 
        w_grad = w_grad - 2.0*(ideal - bias - (weights*input_data).sum()-(weights_2*input_data*input_data).sum())*input_data 
        w_grad_2 = w_grad_2 - 2.0*(ideal - bias - (weights*input_data).sum() - (weights_2*input_data*input_data).sum())*input_data*input_data 
        loss = loss +math.pow((ideal - bias - (weights*input_data).sum()-(weights_2*input_data*input_data).sum()),2)
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    lr_w_2 = lr_w_2 + w_grad_2 ** 2
        
    bias = bias - lr/np.sqrt(lr_b)*b_grad
    weights = weights -lr/np.sqrt(lr_w) * w_grad
    weights_2 = weights_2 -lr/np.sqrt(lr_w_2) * w_grad_2
    print(math.sqrt(loss/len(train)))

ans = []
#Testing
for x in range(0,len(test),5):
    test_data = test[x:x+5]
    testing_data = np.array(test_data)
    ans.append((weights_2*testing_data*testing_data).sum()+(weights*testing_data).sum()+bias)

fout = open('submit_one_five.csv','w')
fout.write('id,value\n')
for cnt in range(len(ans)):
    fout.write('id_'+str(cnt)+','+str(ans[cnt])+'\n')


