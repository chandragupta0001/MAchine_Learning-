import pandas as pd
import numpy as np
from numpy import cov
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
data_set=pd.read_csv("/home/chandragupta/Desktop/heart.csv")
data_X=data_set.iloc[:,0:-1].values
target=data_set.iloc[:,-1].values
data_normalised=(data_X-data_X.mean())/data_X.std()

def PCA(X):
   A = X.iloc[:, 0:-1].values
   M = np.mean(A.T, axis = 1)
   C = A-M
   V = cov(C.T)
# eigendecomposition of covariance matrix
   values, vectors = eig(V)
   P = vectors.T.dot(C.T)
   Z = P.T
   return Z

pca=PCA(data_set)
data=np.copy(pca[:,0:2])
principalDf = pd.DataFrame(data = data
             , columns = ['principal_component_1', 'principal_component_2'])
plt.scatter(principalDf.principal_component_1,principalDf.principal_component_2)
plt.title('Scatter plot Lable for hear data(original')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()
print(pca.std(axis=0))
print(principalDf.std()/np.sum(pca.std(axis=0)))
'''
plt.scatter(data_set.SepalLengthCm,data_set.SepalWidthCm)
plt.title('Scatter plot Lable')
plt.xlabel('SepalLeng')
plt.ylabel('SepalWdth')
plt.show()
'''
# initial u,sigmma,prob

#initials
k=3
n,m=data.shape
global phi
phi=np.full(shape=k,fill_value=1/k)
weight=np.full(shape=[n,k],fill_value=1/k)
random_row=np.random.randint(low=0,high=n,size=k)
mu=[data[row_index,:]for row_index in random_row]
sigma=[np.cov(data.T) for _ in range(k)]
print(n,m,phi,weight,mu,sigma)
def e_step(data):
    global phi,weight
    weight=predict_prob(data)
    phi=weight.mean(axis=0)

def m_step(data):
    global weight,mu,sigma
    for i in range(k):
        w=weight[:,[i]]
        total_w=w.sum()
        mu[i]=(data*w).sum(axis=0)/total_w
        sigma[i]=np.cov(data.T,aweights=(w/total_w).flatten(),bias=True)

def predict_prob(data):
    likelihood=np.zeros((n,k))
    for i in range(k):
        distribution=multivariate_normal(mean=mu[i],cov=sigma[i])
        likelihood[:,i]=distribution.pdf(data)
    numerator = likelihood * phi
    denominator = numerator.sum(axis=1)[:, np.newaxis]
    weight = numerator / denominator
    return weight
#calculating
max_iteration=100

for iteration in range(max_iteration):
   e_step(data)
   m_step(data)
weight_new=predict_prob(data)
result =np.argmax(weight_new,axis=1)
print(result)
plt.scatter(principalDf.principal_component_1,principalDf.principal_component_2,c=result)
plt.title('Scatter plot Lable for hear data(GMM+PCA)')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()
'''
ax = plt.axes(projection='3d')
ax.scatter(data_set.age,data_set.thalach,data_set.chol, c=result, cmap='viridis', linewidth=0.5)
plt.show()
ax = plt.axes(projection='3d')
ax.scatter(data_set.age,data_set.thalach,data_set.chol, c=target, cmap='viridis', linewidth=0.5)
plt.show()
'''
