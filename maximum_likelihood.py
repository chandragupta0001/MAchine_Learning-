import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Data= pd.read_csv('/home/chandragupta/Desktop/heart.csv')
print(Data.describe)
#training and testing set size
train_size=int(0.80*Data.shape[0])
test_size=int(0.20*Data.shape[0])
print("Training set size : ",train_size)
print("Testing set size : ",test_size)
print(Data.describe())

#Getting features from dataset
X=Data.iloc[:,:-1].values
y=Data.iloc[:,-1].values
x=X
#feature scaling
X=(X-X.mean())/X.std()
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
std_0=np.std(class_data_dic[0],axis=1)
std_1=np.std(class_data_dic[1],axis=1)


def likelyhood(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / (np.sqrt(2 * np.pi) * sigma))


def posterior(X, X_train_class, mean_, std_):
    product = np.prod(likelyhood(X, mean_, std_), axis=1)
    product = product * (X_train_class.shape[1] / X_train.shape[0])
    return product
p_1=posterior(X_test,class_data_dic[1],mean_1,std_1)
p_0=posterior(X_test,class_data_dic[0],mean_0,std_0)
#bayse risk
l00=0;
l11=0
l01=1
l10=2
#without risk
#y_pred=1*(p_1>p_0)
#with risk
y_pred=1*(((l10-l00)*p_1)>((l01-l11)*p_0))
print(np.sum(y_pred-y_test)/len(y_pred))
mae=np.sum(abs(y_test-y_pred)/len(y_test))
print('MAE =  ',mae)
print('error',np.sum(np.minimum(p_1/(p_0+p_1),p_0/(p_0+p_1)))/len(p_1))
#plt.plot(Data['SepalLengthCm'],Data['SepalWidthCm'],'ro')
#plt.show()
#sns.pairplot(data=Data['age','sex','target'],hue='target')
plt.show()
print("MAPE")
print("Test  : ",np.mean(np.abs((y_test- y_pred) / (y_test+1)) * 100))





