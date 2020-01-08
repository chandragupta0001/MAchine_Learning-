import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
data = pd.read_csv("/home/chandragupta/Desktop/fashion.csv")

X = data.iloc[0:10000,0:-1].values
y = data.iloc[0:10000,-1].values

#Test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying PCA here
from sklearn.decomposition import PCA
pca = PCA(n_components= 2) #We will set it none so that we can see the variance explained and then choose no of comp.
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_


#Create classifier object
from sklearn.svm import SVC
classifier_svm_kernel = SVC(C=5.0,kernel='linear', gamma=0.12,tol=0.00001)
classifier_svm_kernel.fit(X_train,y_train)

#Predict the result for test values
y_pred = classifier_svm_kernel.predict(X_test)
print(y_pred)

#Compute confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test,y_pred)
print(classification_report(y_test, y_pred))

plot_data = np.copy(X_test)
principalDf = pd.DataFrame(data = plot_data, columns = ['principal_component_1', 'principal_component_2'])

plt.scatter(principalDf.principal_component_1, principalDf.principal_component_2, c = y_pred)

plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()
print()