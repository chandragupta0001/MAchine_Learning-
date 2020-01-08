import pandas as pd
import numpy as np
data=pd.read_csv('/home/chandragupta/Desktop/heart_data/heart.csv')
test=pd.read_csv('/home/chandragupta/Desktop/heart_data/heart_test.csv')
train=pd.read_csv('/home/chandragupta/Desktop/heart_data/heart_training.csv')
train_no=pd.read_csv('/home/chandragupta/Desktop/heart_data/heart_training_no.csv')
train_yes=pd.read_csv('/home/chandragupta/Desktop/heart_data/heart_training_yes.csv')
print(data.describe())

test_X=test.iloc[:,0:13].values
print(np.mean(test_X,axis=0))
test_y=test.iloc[:,13].values
test_X=((test_X-test_X.mean())/test_X.std())
train_yes=((train_yes-train_yes.mean())/train_yes.std())
train_no=((train_no-train_no.mean())/train_no.std())
#print(test_X)
mean_train_yes=np.mean(train_yes.values,axis=0)
mean_train_no=np.mean(train_no.values,axis=0)
std_train_yes=np.std(train_yes.values,axis=0)
std_train_no=np.std(train_no.values,axis=0)
print(std_train_no)
def Lg(x,mean,sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / (np.sqrt(2 * np.pi) * sigma))
def posterior(x,prior,mean,sigma):
    product=np.product(Lg(x,mean,sigma),axis=1)
    print('product,',product)
    pos=product*(len(prior)/len(train))
    return pos
p1=posterior(test_X,train_yes,mean_train_yes,std_train_yes)
p0=posterior(test_X,train_no,mean_train_no,std_train_no)
y_pred=1*(p1>p0)
print(mean_train_yes)
print(y_pred-test_y)
mae=np.sum(abs(test_y-y_pred)/len(test_y))
print('MAE =  ',1-mae)