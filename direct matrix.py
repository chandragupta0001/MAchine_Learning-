import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data is divided into three groups,Raw , test ,training
train_data=pd.read_csv("/home/chandragupta/Desktop/Ainsurance_train.csv")
test_data=pd.read_csv("/home/chandragupta/Desktop/Ainsurance_test.csv")
all_data=pd.read_csv("/home/chandragupta/Desktop/Ainsurance.csv")
feature=['arg','age','sex','bmi','children','smoker',]
X=all_data[feature].values
y=all_data['expenses'].values
test_X=test_data[feature].values
test_y=test_data['expenses'].values
train_X=train_data[feature].values
train_y=train_data['expenses'].values
print(X.mean())
#data normalize
ntest_X=(test_X-X.mean())/X.std()
ntest_y=(test_y-y.mean())/y.std()
ntrain_X=(train_X-X.mean())/X.std()
ntrain_y=(train_y-y.mean())/y.std()
#calculating W matrix
W=np.linalg.inv((ntrain_X.T@ntrain_X)) @ ntrain_X.T @ntrain_y
npredict_y=ntest_X @ W.T
predict_y=npredict_y * y.std() + y.mean()# denormalize
train_predict=(ntrain_X @ W.T)* y.std() +y.mean()
#plot
fig, ax = plt.subplots()
ax.plot(test_y, predict_y, 'p')
ax.set_title('MATRIX : test_y, prd_y')
plt.show()
#MAE
mae=np.sum(abs(test_y-predict_y)/len(test_y))
print('MAE =  ',mae)
print("MAPE")
print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)
print("Test  : ",np.mean(np.abs((test_y - predict_y) / test_y)) * 100)
fig, ax = plt.subplots()
ax.plot(test_y, predict_y, 'p')
ax.set_title('GRADIENT : test_y, prd_y')
plt.show()
