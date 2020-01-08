#one vs one
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
df_01=df[df['Species'].isin([0,1])]
df_12=df[df['Species'].isin([1,2])]
df_12.Species.replace([1,2], [0, 1], inplace=True)
df_20=df[df['Species'].isin([2,0])]
df_20.Species.replace([2,0], [0, 1], inplace=True)

def cal(Data):
    train_size = int(0.80 * Data.shape[0])
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
W_01=cal(df_01)
W_12=cal((df_12))
W_20=cal(df_20)
print(W_01,W_12,W_20)
y_01=1*(df_test_X@W_01>0)
y_12=2*(df_test_X@W_12>0)+1*(df_test_X@W_12<0)
y_20=2*(df_test_X@W_20<0)
print('01=',y_01)
print('12=',y_12)
print('20',y_20)
print(df_test_y)
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
