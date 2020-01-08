import pandas as pd
import numpy as np
import math
Data= pd.read_csv('/home/chandragupta/Desktop/heart.csv')#non argumen
print(Data.describe())
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
class_0=class_data_dic[0].T
class_1=class_data_dic[1].T

mean_0=np.asmatrix(np.mean(class_data_dic[0],axis=1))
mean_1=np.asmatrix(np.mean(class_data_dic[1],axis=1))
cov_0=np.cov(class_data_dic[0])

cov_1=np.cov(class_data_dic[1])
a=class_0-mean_0
b=class_1-mean_1
Sw=a.T @ a+b.T@b
print(Sw.shape)
W=np.linalg.inv(Sw)@((mean_1-mean_0).T)
print(W)
y_pred=np.asmatrix(1*(X_train@W>0))
mae=np.sum(abs(y_train-y_pred.T)/len(y_train))

print('MAE =  ',mae)
print("MAPE")
print("Test  : ",np.mean(np.abs((y_train- y_pred.T) / (y_train+1)) * 100))