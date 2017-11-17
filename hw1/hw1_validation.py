import numpy as np
import math
train_d = []
train = []
valid = []
test = []

wg_simple = np.zeros(9)

wg_quadratic = np.zeros(9)
wg_quadratic_2 = np.zeros(9)

bias_q = 1.71453
bias_s = 1.71453

lr_s = 10000
lr_s_b = 0.0
lr_s_w = 0.0

lr_q = 10000
lr_q_b = 0.0
lr_q_w = 0.0
lr_q_w_2 = 0.0
iteration = 12000


#Reading train data

fin = open('train.csv')
for line in fin:
    split = line.split(',')
    if(split[2] == "PM2.5"):
            train_d.extend(map(lambda x: float(x), split[3:]))
fin.close()

train = train_d[:len(train_d)-100]
valid = train_d[len(train_d)-100:]

#Reading test data

fin = open('test.csv')
for line in fin:
    split = line.split(',')
    if(split[1] == "PM2.5"):
            test.extend(map(lambda x: float(x), split[2:]))
fin.close()


# Training 
for i in range(iteration):
    print(i)
    
    b_simple_grad = 0.0
    w_simple_grad = np.zeros(9)
    loss_simple = 0
    
    b_quadratic_grad = 0.0
    w_quadratic_grad_grad = np.zeros(9)
    w_quadratic_grad_grad_2 = np.zeros(9)
    loss_quadratic = 0
    for j in range(len(train)-10):
        train_data = train[j:j+10]
        ideal = float(train_data[9])
        input_data = np.array(train_data)
        input_data = np.delete(input_data,9)
        
        #simple 
        b_simple_grad = b_simple_grad - 2 * (ideal - bias_s - (wg_simple*input_data).sum()) * 1
        w_simple_grad = w_simple_grad - 2.0*(ideal - bias_q - (wg_simple*input_data).sum()*input_data 
        loss_simple = loss_simple + math.pow((ideal - bias_q - (wg_quadratic*input_data).sum(2    )),2)
        # Quadratic 
        b_quadratic_grad = b_quadratic_grad - 2.0*(ideal - bias_q - (wg_quadratic*input_data).sum()-(wg_quadratic_2*input_data*input_data).sum())*1.0 
        w_quadratic_grad = w_quadratic_grad - 2.0*(ideal - bias_q - (wg_quadratic*input_data).sum()-(wg_quadratic_2*input_data*input_data).sum())*input_data 
        w_quadratic_grad_2 = w_quadratic_grad_2 - 2.0*(ideal - bias_q - (wg_quadratic*input_data).sum() - (wg_quadratic_2*input_data*input_data).sum())*input_data*input_data 
        loss_quadratic = loss_quadratic +math.pow((ideal - bias_q - (wg_quadratic*input_data).sum()-(wg_quadratic_2*input_data*input_data).sum()),2)

       
    
    #simpe adagrad
    lr_s_b = lr_s_b + b_simple_grad ** 2
    lr_s_w = lr_s_w + w_simple_grad ** 2
    
    bias_s = bias_s - lr_s/np.sqrt(lr_s_b)*b_simple_grad
    wg_simple = wg_simple - lr_s/np.sqrt(lr_s_w)* w_simple_grad

    #Quadratic adagrad
    lr_q_b = lr_q_b + b_quadratic_grad ** 2
    lr_q_w = lr_q_w + w_quadratic_grad ** 2
    lr_q_w_2 = lr_q_w_2 + w_quadratic_grad_2 ** 2
        
    bias_q = bias_q - lr_q/np.sqrt(lr_q_b)*b_quadratic_grad
    wg_quadratic = wg_quadratic -lr_q/np.sqrt(lr_q_w) * w_quadratic_grad
    wg_quadratic_2 = wg_quadratic_2 -lr_q/np.sqrt(lr_q_w_2) * w_quadratic_grad_2
    print(math.sqrt(loss_quadratic/len(train)))


#validation



ans = []

#Testing
for x in range(0,len(test),9):
    test_data = test[x:x+9]
    testing_data = np.array(test_data)
    ans.append((wg_quadratic_2*testing_data*testing_data).sum()+(wg_quadratic*testing_data).sum()+bias_q)

fout = open('submit.csv','w')
fout.write('id,value\n')
for cnt in range(len(ans)):
    fout.write('id_'+str(cnt)+','+str(ans[cnt])+'\n')


