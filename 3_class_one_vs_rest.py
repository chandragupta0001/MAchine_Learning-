#one vs rest
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
raw_data=pd.read_csv('/home/chandragupta/Desktop/Iris_3.csv')
df=raw_data[0:100]

df_test=raw_data[101:150]
df_test_X = df_test.iloc[:, :-1].values
df_test_y = df_test.iloc[:, -1].values
print(df_test)
df_0=df.copy()
df_1=df.copy()
df_2=df.copy()
print(df_0)
df_0.Species.replace([0,1,2], [1, 0,0], inplace=True)
df_1.Species.replace([0,1,2], [0, 1,0], inplace=True)
df_2.Species.replace([0,1,2], [0, 0,1], inplace=True)
print(df_0)

def cal(Data):
    train_size = int( Data.shape[0])
    test_size = int(0.20 * Data.shape[0])
    X = Data.iloc[:, :-1].values
    y = Data.iloc[:, -1].values
    #X_train = X[0:train_size, :]
    #y_train = y[0:train_size]
    # testing set split
   # X_test = X[train_size:, :]
   # y_test = y[train_size:]
   # y_test = np.asmatrix(y_test)
    #y_test = y_test.T
    w = np.ones([X.shape[1], 1])
    n = 0
    lrate = .001
    y_hat = np.ones([train_size, 1])
    while n < 50:
        for i in range(0, train_size):
            f = np.dot(X[i], w)
            # print(f)
            y_hat[i] = 1 * (f > 0)
            # print(y_pred[i])
            for j in range(0, X.shape[1]):
                w[j] = w[j] + lrate * (y[i] - y_hat[i]) * X[i][j]
        n = n + 1
    return w
W_0=cal(df_0)
W_1=cal((df_1))
W_2=cal(df_2)
print(W_0,W_1,W_2)
y_0=df_test_X@W_0
y_1=df_test_X@W_1
y_2=df_test_X@W_2
print('0=',y_0)
print('1=',y_1)
print('2=',y_2)
print(df_test_y)
y_pred=[]
print(y_0[0])
for i in range(0,len(y_0)):
    m=max(y_0[i],y_1[i],y_2[i])
    if m==y_0[i]:
        y_pred.append(0)
    if m==y_1[i]:
        y_pred.append(1)
    if m==y_2[i]:
        y_pred.append(2)
print(np.asmatrix(y_pred))
print(df_test_y)
error=1*((df_test_y-y_pred)!=0)

print('accuracy=',1-np.sum(error)/len(error))



"""""
#print(np.maximum(y_01,np.maximum(y_12,y_20)))
plt.plot()
print(df_test_y.shape)
z=[y_01,y_20,y_12]
z=np.concatenate((y_01,y_20,y_12),axis=1)
q=[]
for i in z:
    q.append(statistics.mode(i))
print(q)
error=1*((q-df_test_y)!=0)
print('accuracy',1-np.sum(error)/len(q))
#print(np.mod(z,axis=1))
"""