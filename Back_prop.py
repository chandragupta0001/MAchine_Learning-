import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

layer1 = 100
Data_set = pd.read_csv("digits.csv")
features = Data_set.shape[1] - 1
d = Data_set.iloc[:, :-1].values
Data = (d - d.mean()) / (d.std() + 1)
Data = np.insert(Data, 0, 1, axis=1)
l_rate = .01

# print(Data)
train_X = Data[0:int(Data.shape[0] * .75), :]
y = Data_set.iloc[:, -1:]
y_ = y.T
# creat on hot notation
c = np.zeros((Data.shape[0], 10))
for i in range(Data.shape[0]):
    z = y_[i]
    c[i][z] = 1

train_Y = c[0:int(Data.shape[0] * .75), :]
test_X = Data[int(Data.shape[0] * .75):, :]
test_Y = c[int(Data.shape[0] * .75):, :]



def softmax(x):
    expA = np.exp(x)
    return expA / expA.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward(data_sample, lable):
    global W1, W2, y, z, a
    a = W1 @ data_sample.T
    z = sigmoid(a)
    z = np.insert(z, 0, 1, axis=0)
    y_ = W2 @ z
    y = softmax(y_)


def backward(data_sample, lable):
    global W1, W2
    delta2 = y - lable.T
    d_error2 = delta2 @ z.T
    W2 = W2 - l_rate * d_error2
    d_error1 = d_sigmoid(a) * np.sum(delta2.T @ W2[:, :-1]) @ data_sample
    W1 = W1 - l_rate * d_error1

l=np.empty(0)
e=np.empty(0)
for layer1 in range(50,1000,50):
  W1 = np.random.randn(layer1, features + 1)
  W2 = np.random.randn(10, layer1 + 1)
  for j in range(10):
    for i in range(train_X.shape[0]):
        forward(train_X[i:i + 1, :], train_Y[i:i + 1, :])
        backward(train_X[i:i + 1, :], train_Y[i:i + 1, :])
  err = 0
  for i in range(len(test_X)):
    forward(test_X[i:i + 1, :], test_Y[i:i + 1, :])
    if (np.argmax(test_Y[i:i + 1, :]) != np.argmax(y.T)):
        err += 1
  error=err / len(test_X)
  print(error)
  l=np.append(l,layer1)
  e=np.append(e,error)
plt.plot(l,e,color='red')
plt.title('error vs node in hidden layer')
plt.xlabel('layer')
plt.ylabel('error')
plt.show()



