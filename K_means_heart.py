import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_set=pd.read_csv("/home/chandragupta/Desktop/heart.csv")
data=data_set[["age","thalach"]]
#print(data)
X=data.values

colors = 10*["g","r","c","b","k"]
k=2
tol=0.1
max_iter=30

def fit(data):

        centroids={}
        for i in range(k):
            centroids[i] = data[i]

        for i in range(max_iter):
            classifications={}

            for i in range(k):
                classifications[i] = []

            for featureset in data:
                #print('featureset',featureset)
                distances = [np.linalg.norm(featureset-centroids[b]) for b in range(k)]
                #print('distance=',distances)
                classification = distances.index(min(distances))
                classifications[classification].append(featureset)
            prev_centroids = dict(centroids)

            for classes in range(k):
                centroids[classes] = np.average(classifications[classes],axis=0)

            optimized = True

            for c in range(k):
                original_centroid = prev_centroids[c]
                current_centroid = centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))

                    optimized = False

            if optimized:
                break
        return centroids,classifications


centroids,classifications=fit(X)




for classification in classifications:
    color = colors[classification]
    for featureset in classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, s=5, linewidths=5)
for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1],
                marker="X", color="y", s=500, linewidths=1)
plt.title('Scatter plot Lable for hear data(K_mean)')
plt.xlabel('AGE')
plt.ylabel('Thalach')
plt.show()