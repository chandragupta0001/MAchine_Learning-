import pandas as pd
import numpy as np
import math
Data= pd.read_csv('/home/chandragupta/Desktop/heart_data/heart.csv')#non argum
#training and testing set size
train_size=int(0.80*Data.shape[0])
test_size=int(0.20*Data.shape[0])
print("Training set size : ",train_size)
print("Testing set size : ",test_size)

#Getting features from dataset
X=Data.iloc[:,0:13].values
y=Data.iloc[:,13].values
#feature scaling
#X=(X-X.mean())/X.std()
#training set split
X_train=X[0:train_size,:]
y_train=y[0:train_size]
#testing set split
X_test=X[train_size:,:]
y_test=y[train_size:]
class_data_dic={}

def generate_data(class_data_dic, X_train, y_train):
    first_one = True
    first_zero = True
    for i in range(y_train.shape[0]):
        X_temp = X_train[i, :].reshape(X_train[i, :].shape[0], 1)
        if y_train[i] == 1:
            if first_one == True:
                class_data_dic[1] = X_temp
                first_one = False
            else:
                class_data_dic[1] = np.append(class_data_dic[1], X_temp, axis=1)
        elif y_train[i] == 0:
            if first_zero == True:
                class_data_dic[0] = X_temp
                first_zero = False
            else:
                class_data_dic[0] = np.append(class_data_dic[0], X_temp, axis=1)
    return class_data_dic
generate_data(class_data_dic, X_train, y_train)

mean_0=np.mean(class_data_dic[0],axis=1)
mean_1=np.mean(class_data_dic[1],axis=1)
cov_0=np.cov(class_data_dic[0])
cov_1=np.cov(class_data_dic[1])
prior_1=class_data_dic[1].shape[1]/train_size
prior_0=class_data_dic[0].shape[1]/train_size
#discriment function for the naormal density
#upon analys. we find its case 2.6.3 E=arbitray
W1=-.5*np.linalg.inv(cov_1)
W0=-.5*np.linalg.inv(cov_0)

w1=np.linalg.inv(cov_1)@mean_1
w0=np.linalg.inv(cov_0)@mean_0
b1=-.5*mean_1.T@np.linalg.inv(cov_1)@mean_1-.5*math.log(np.linalg.det(cov_1),math.e)+math.log(prior_1,math.e)
b0=-.5*mean_0.T@np.linalg.inv(cov_0)@mean_0-.5*math.log(np.linalg.det(cov_0),math.e)+math.log(prior_0,math.e)
print(w1)
y_pred=[]
for i in X_test:
     g1=i@W1@i.T+w1.T@i.T+b1
     g0=i@W0@i.T+w0.T@i.T+b0
     y_pred.append(1*(g1>g0))
print("mae")
print(np.sum(abs(y_pred-y_test.T))/len(y_test))
print("MAPE")
print("Test  : ",np.mean(np.abs((y_test.T- y_pred) / (y_test+1)) * 100))


""""
def likelyhood(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / (np.sqrt(2 * np.pi) * sigma))


def posterior(X, X_train_class, mean_, std_):
    product = np.prod(likelyhood(X, mean_, std_), axis=1)
    product = product * 1#(X_train_class.shape[0] / X_train.shape[0])
    return product
p_1=posterior(X_test,class_data_dic[1],mean_1,std_1)
p_0=posterior(X_test,class_data_dic[0],mean_0,std_0)
y_pred=1*(p_1>p_0)
print(np.sum(y_pred-y_test)/len(y_pred))
mae=np.sum(abs(y_test-y_pred)/len(y_test))
print('MAE =  ',mae)"""