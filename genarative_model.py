#probablistic generative model for continous input under gaussian class conditional density
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
Data= pd.read_csv('/home/chandragupta/Desktop/heart_data/heart.csv')#non arg
print(Data.head())
print(Data.shape)
print(Data.shape[0])
print(Data.shape[1])
#training and testing set size
train_size=int(0.80*Data.shape[0])
test_size=int(0.20*Data.shape[0])
print("Training set size : ",train_size)
print("Testing set size : ",test_size)

#Getting features from dataset
X=Data.iloc[:,:-1].values
y=Data.iloc[:,-1].values
#feature scaling
#X=(X-X.mean())/X.std()
#training set split
X_train=X[0:train_size,:]
y_train=y[0:train_size]
#testing set split
X_test=X[train_size:,:]
y_test=np.asmatrix(y[train_size:])
print(X_test)
print((y_test.T))
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
class_0=class_data_dic[0].T
class_1=class_data_dic[1].T

mean_0=np.asmatrix(np.mean(class_data_dic[0],axis=1))
mean_1=np.asmatrix(np.mean(class_data_dic[1],axis=1))
#cov_0=np.cov(class_data_dic[0])
#cov_1=np.cov(class_data_dic[1])
cov=np.cov(X.T)
print(mean_0.shape,mean_1.shape,cov.shape)
W=np.linalg.inv(cov)@(mean_0-mean_1).T
print(len(class_1))
w0=-.5*mean_0@np.linalg.inv(cov)@mean_0.T+.5*mean_1@np.linalg.inv(cov)@mean_1.T+math.log(len(class_0)/len(class_1),math.e)
def sig(a,b,c):
    return 1/(1+np.exp(-(b@a+c)))
p_0=sig(W,X_test,w0)
y_pred=np.asmatrix(1*(p_0<=.5))
print("mae=",np.sum(abs(y_pred-y_test.T))/test_size)
#print(y_pred-y_test.T)
plt.plot(p_0,1-p_0,Data['age'],'r')
plt.show()
print("MAPE")
print("Test  : ",np.mean(np.abs((y_test.T- y_pred) / (y_test+1)) * 100))