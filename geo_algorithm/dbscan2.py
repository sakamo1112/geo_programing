import numpy as np
from scipy.spatial import distance
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

def dbscan(X, eps, min_samples):
    labels = [0]*len(X)
    C = 0

    for P in range(0, len(X)):
        if not (labels[P] == 0):
           continue
        NeighborPts = region_query(X, P, eps)
        if len(NeighborPts) < min_samples:
            labels[P] = -1
        else: 
           C += 1
           grow_cluster(X, labels, P, NeighborPts, C, eps, min_samples)

    return np.array(labels)

def grow_cluster(X, labels, P, NeighborPts, C, eps, min_samples):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):    
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = region_query(X, Pn, eps)
            if len(PnNeighborPts) >= min_samples:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1        

def region_query(X, P, eps):
    neighbors = []
    for Pn in range(0, len(X)):
        if np.linalg.norm(X[P] - X[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors

# サンプルデータを生成
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
labels = dbscan(X, eps=0.2, min_samples=5)
# クラスタをプロット
plt.scatter(X[labels == 1, 0], X[labels == 1, 1],
            c='lightblue', marker='o', s=40,
            edgecolor='black', 
            label='cluster 1')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1],
            c='red', marker='s', s=40,
            edgecolor='black', 
            label='cluster 2')
plt.legend()
plt.show()
