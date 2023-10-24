from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# サンプルデータを生成
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# DBSCANを定義
db = DBSCAN(eps=0.3, min_samples=5)

# フィットと予測
y_db = db.fit_predict(X)

# クラスタをプロット
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c='lightblue', marker='o', s=40,
            edgecolor='black', 
            label='cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c='red', marker='s', s=40,
            edgecolor='black', 
            label='cluster 2')
plt.legend()
plt.show()