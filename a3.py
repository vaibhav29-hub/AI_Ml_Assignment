# Assignment 3: Dimensionality reduction using PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
print("Original shape:", X.shape)

# standardize before PCA
X_scaled = StandardScaler().fit_transform(X)

# full PCA to see variance
pca_full = PCA().fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
print("\nExplained variance ratio of first 5 components:")
print(pca_full.explained_variance_ratio_[:5].round(4))
print("Components needed for 95% variance:", np.argmax(cum_var >= 0.95) + 1)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 31), cum_var, marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - Cumulative Variance")
plt.tight_layout()
plt.savefig("a3_variance.png", dpi=110)

# reduce to 2 components for visualization
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)
print("\nShape after PCA (2 components):", X_pca.shape)
print("Variance kept by 2 components:", round(pca2.explained_variance_ratio_.sum(), 4))

plt.figure(figsize=(6, 4))
for label, color, name in [(0, 'red', 'malignant'), (1, 'green', 'benign')]:
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], c=color, s=12, label=name)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.legend(); plt.title("Data in 2 Principal Components")
plt.tight_layout()
plt.savefig("a3_pca2d.png", dpi=110)

# compare classifier accuracy: all features vs 10 PCA components
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("\nAccuracy with 30 features :", round(accuracy_score(yte, clf.predict(Xte)), 4))

pca10 = PCA(n_components=10).fit(Xtr)
clf2 = LogisticRegression(max_iter=1000).fit(pca10.transform(Xtr), ytr)
print("Accuracy with 10 PCA comps:", round(accuracy_score(yte, clf2.predict(pca10.transform(Xte))), 4))
