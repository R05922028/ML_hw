import numpy as np
import math
train = [None] * 18
test = [None] * 18
weights = np.zeros((18,5))
bias = 1.71453
lr = 1
lr_b = 0.0
lr_w = 0.0
iteration = 15000
def FeatureIndex(string):
    if(string == "AMB_TEMP"):   return 0
    elif(string == "CH4"):      return 1
    elif(string == "CO"):       return 2
    elif(string == "NMHC"):     return 3
    elif(string == "NO"):       return 4
    elif(string == "NO2"):      return 5
    elif(string == "NOx"):      return 6
    elif(string == "O3"):       return 7
    elif(string == "PM10"):     return 8
    elif(string == "PM2.5"):    return 9
    elif(string == "RAINFALL"): return 10
    elif(string == "RH"):       return 11
    elif(string == "SO2"):      return 12
    elif(string == "THC"):      return 13
    elif(string == "WD_HR"):    return 14
    elif(string == "WIND_DIREC"):return 15
    elif(string == "WIND_SPEED"):return 16
    elif(string == "WS_HR"):    return 17
    else : return 18 

    


fin = open('train.csv',encoding='big5')
for line in fin:
    split = line.split(',')
    if(FeatureIndex(split[2])<18):
        if(train[FeatureIndex(split[2])]==None):
            train[FeatureIndex(split[2])]=list(map(lambda x: 0 if 'NR' in x else float(x), split[3:]))
        else:
            train[FeatureIndex(split[2])].extend(map(lambda x: 0 if 'NR' in x else float(x), split[3:]))
fin.close()

#Reading test data
fin = open('test.csv',encoding='big5')
for line in fin:
    split = line.split(',')
    if(FeatureIndex(split[1])<18):
        if(test[FeatureIndex(split[1])]==None):
            test[FeatureIndex(split[1])] = list(map(lambda x: 0 if 'NR' in x else float(x), split[2:]))
        else:
            test[FeatureIndex(split[1])].extend(map(lambda x: 0 if 'NR' in x else float(x), split[2:]))
fin.close()

# Training 
for i in range(iteration):
    print(i)
    b_grad = 0.0
    loss = 0.0
    w_grad = np.zeros((18,5))
    for j in range(len(train[9])-6):
        train_data = list(map(lambda i: i[j: j+6] if i!=None else None, train))
        ideal = float(train_data[9][5])
        input_data = np.array(train_data)
        input_data = np.delete(input_data,5,1)
        b_grad = b_grad - 2.0*(ideal - bias - (weights*input_data).sum())*1.0 
        w_grad = w_grad - 2.0*(ideal - bias - (weights*input_data).sum())*input_data 
        loss = loss +math.pow((ideal - bias - (weights*input_data).sum()),2)
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2
    
    bias = bias - lr/np.sqrt(lr_b)*b_grad
    weights = weights -lr/np.sqrt(lr_w) * w_grad
    
    print("loss : "+str(math.sqrt(loss/len(train[10]))))

ans = []
#Testing
for x in range(0,len(test[9]),9):
    test_data = list(map(lambda i : i[x+4:x+9] if i!=None else None, test))
    testing_data = np.array(test_data)
    ans.append((weights*testing_data).sum()+bias)

fout = open('submit_simple_all_five.csv','w')
fout.write('id,value\n')
for cnt in range(len(ans)):
    fout.write('id_'+str(cnt)+','+str(ans[cnt])+'\n')


