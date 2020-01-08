import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train_data=pd.read_csv("/home/chandragupta/Desktop/Ainsurance_train.csv")
print(train_data.head)
print(train_data)
#y=train_data['expenses']
y = train_data.iloc[:,7:8].values
y=(y-y.mean())/y.std()
#print(y)
#print(train_data.columns)
feature=['arg','age','sex','bmi','children','smoker',]
#X=train_data[feature]
X=train_data.iloc[:,0:6].values
X=(X-X.mean())/X.std()
print(y.shape)
#train_data.info()
#ins_corr=train_data.corr()
#print(ins_corr)
train_data = train_data.drop_duplicates()
plt.figure(figsize=(14, 7))
print(sns.scatterplot(x=train_data['bmi'], y=train_data['expenses'],hue=train_data['smoker'],size=train_data['age']))
plt.show()
lam=0.00001
W=np.zeros([1,6])
def costfn(X,W,y):
    temp=X.dot(W.T)
    #print("temp=" ,temp)
    #print(temp-y)
    cost=np.power((temp-y),2)
    costtotal=np.sum(cost,axis=0)+ lam*.5*np.linalg.norm((W))**2
    return costtotal[0]
print("totalcost=",costfn(X,W,y))
#graident
l_rate=.0005
itr=10000
def grd(X,y,W,l_rate,itr):
    cost=np.zeros(itr)
    for i in range(itr):
        #W=W-l_rate*0.5*np.sum((X @ W.T - y).T@X, axis=0)
        #W = W - l_rate *0.5* (((X @ W.T - y).T) @ X)
        W = W - l_rate *0.5* np.sum((X @ W.T - y).T@X,axis=0) + lam*np.sum(abs(W),axis=0)
        #W = W - l_rate * np.sum(X*(X @ W.T - y),axis=0) +.0000001*np.sum(abs(W), axis=0)

        cost[i]=costfn(X,W,y)
        #print(cost[i])
    return W,cost

g,cost = grd(X,y,W,l_rate,itr)
a,c2=grd(X,y,W,0.0001,itr)
a,c3=grd(X,y,W,0.0003,itr)
a,c4=grd(X,y,W,0.0004,itr)
a,c5=grd(X,y,W,0.0005,itr)
print('g=',g)
print("w=",W)
finalcost=costfn(X,g,y)
print("final cost =",finalcost)
test_data=pd.read_csv("/home/chandragupta/Desktop/Ainsurance_test.csv")
test_X=test_data.iloc[:,0:6].values
test_X=(test_X-test_X.mean())/test_X.std()
test_y = test_data.iloc[:,7:8].values
test_y1=(test_y-test_y.mean())/test_y.std()
prd_y=test_X @ g.T
train_predict=(X @g.T)*y.std()+y.mean()
prd_y=prd_y*test_y.std()+test_y.mean()
#plot the cost
fig, ax = plt.subplots()
ax.plot(np.arange(itr), cost, 'r')
ax.plot(np.arange(itr), c2, 'b')
ax.plot(np.arange(itr), c3, 'g')
ax.plot(np.arange(itr), c4, 'y')
ax.plot(np.arange(itr), c5, 'black')
plt.legend(['Lrate=0.0001', 'Lrate=0.0002', 'Lrate=0.0003','Lrate=0.0004', 'Lrate=0.0005'], loc='upper right')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('GRADIENT: : cost vs. Training  ')
plt.show()
fig, ax = plt.subplots()
ax.plot(test_y, prd_y, 'p')
ax.set_title('GRADIENT : test_y, prd_y')
plt.show()

mae=np.sum(abs(test_y-prd_y)/len(test_y))
print('MAE =  ',mae)
print("MAPE")
print("Test  : ",np.mean(np.abs((test_y - prd_y) / test_y)) * 100)
print(g)
