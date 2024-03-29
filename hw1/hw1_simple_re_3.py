import numpy as np
import math
train = []
test = []
weights = np.zeros(9)
bias = 1.71453
lr = 10000
lr_b = 0.0
lr_w = 0.0
lam = 0.001
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
    w_grad = np.zeros(9)
    for j in range(len(train)-10):
        train_data = train[j:j+10]
        ideal = float(train_data[9])
        input_data = np.array(train_data)
        input_data = np.delete(input_data,9)
        b_grad = b_grad - 2.0*(ideal - bias - (weights*input_data).sum())*1.0 
        w_grad = w_grad - 2.0*(ideal - bias - (weights*input_data).sum())*input_data 
        loss = loss +math.pow((ideal - bias - (weights*input_data).sum()),2)
    w_grad = w_grad + 2 * lam * weights    
    loss = loss + lam * ((weights ** 2).sum())
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
        
    bias = bias - lr/np.sqrt(lr_b)*b_grad
    weights = weights -lr/np.sqrt(lr_w) * w_grad
    print(math.sqrt(loss/len(train)))
ans = []
#Testing
for x in range(0,len(test),9):
    test_data = test[x:x+9]
    testing_data = np.array(test_data)
    ans.append((weights*testing_data).sum()+bias)

fout = open('submit_simple_one_re_3.csv','w')
fout.write('id,value\n')
for cnt in range(len(ans)):
    fout.write('id_'+str(cnt)+','+str(ans[cnt])+'\n')


