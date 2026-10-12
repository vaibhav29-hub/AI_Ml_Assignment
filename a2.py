# Assignment 2: Clustering using unsupervised learning
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# load iris data, we do NOT use labels (unsupervised)
X = load_iris().data
X = StandardScaler().fit_transform(X)

# elbow method to find k
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
print("WCSS for k=1..10:")
for k, w in enumerate(wcss, 1):
    print(f"k={k}: {w:.2f}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of clusters k")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig("a2_elbow.png", dpi=110)

# KMeans with k = 3
km = KMeans(n_clusters=3, n_init=10, random_state=42)
km_labels = km.fit_predict(X)

# Hierarchical clustering
hc_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)

# DBSCAN
db_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)

print("\nSilhouette score KMeans      :", round(silhouette_score(X, km_labels), 3))
print("Silhouette score Hierarchical:", round(silhouette_score(X, hc_labels), 3))
print("Silhouette score DBSCAN      :", round(silhouette_score(X, db_labels), 3))
print("DBSCAN clusters found:", len(set(db_labels)) - (1 if -1 in db_labels else 0),
      "| noise points:", list(db_labels).count(-1))

# plot clusters in 2D using PCA
X2 = PCA(n_components=2).fit_transform(X)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, lab, name in zip(ax, [km_labels, hc_labels, db_labels],
                        ["KMeans", "Hierarchical", "DBSCAN"]):
    a.scatter(X2[:, 0], X2[:, 1], c=lab, cmap='viridis', s=20)
    a.set_title(name)
plt.tight_layout()
plt.savefig("a2_clusters.png", dpi=110)
