import pandas as pd
import numpy as np
Data=pd.read_csv('/home/chandragupta/Desktop/heart_arg.csv')#heart_arg
train_size=int(0.80*Data.shape[0])
test_size=int(0.20*Data.shape[0])
X=Data.iloc[:,:-1].values
y=Data.iloc[:,-1].values
X_train=X[0:train_size,:]
y_train=y[0:train_size]
#testing set split
X_test=X[train_size:,:]
y_test=y[train_size:]
y_test=np.asmatrix(y_test)
y_test=y_test.T
print(X_test.shape[1])
w=np.ones([X_train.shape[1],1])
n=0
lrate=.1
itr=5000
y_hat=np.ones([train_size,1])
while n<itr:
    for i in range(0,train_size):
        f=np.dot(X_train[i],w)
        #print(f)
        y_hat[i]=1*(f>0)
        #print(y_pred[i])
        for j in range(0,X_train.shape[1]):
            w[j]=w[j]+lrate*(y_train[i]-y_hat[i])*X_train[i][j]

    n=n+1

print(w)



y_pred=np.zeros([test_size+1,1])
print(y_pred.shape)
for i in range(0,test_size):
    f=np.dot(X_test[i],w)
    ydesh=1*(f>0)
    y_pred[i]=ydesh

print("mae=",np.sum(abs(y_pred-y_test))/test_size)
print("MAPE")
print("Test  : ",np.mean(np.abs((y_test- y_pred) / (y_test+1)) * 100))

"""""
for heart_arg dataset
n=5000 mae=.46 mape=22.9 lrate .001
n=5000 mae=..45 mape=22.13 lrat .01
n=5000 mae=.434 mape=21.31 lrate= .1
n=500 mae=.5 mape=24.59 lrate=.001
for iris_binary data
mae=0 
"""