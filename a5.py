# Assignment 5: KNN algorithm for classification
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3,
                                      random_state=42, stratify=iris.target)

# scaling is important for KNN because it uses distance
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

# try different values of k
acc = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    acc.append(accuracy_score(yte, knn.predict(Xte)))
print("Accuracy for k = 1 to 20:")
print([round(a, 3) for a in acc])

best_k = acc.index(max(acc)) + 1
print("\nBest k:", best_k)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 21), acc, marker='o')
plt.xlabel("k (number of neighbours)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.xticks(range(1, 21))
plt.tight_layout()
plt.savefig("a5_k.png", dpi=110)

# final model with best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(Xtr, ytr)
pred = knn.predict(Xte)
print("Final Accuracy:", round(accuracy_score(yte, pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(yte, pred))
print("\nClassification Report:")
print(classification_report(yte, pred, target_names=iris.target_names))

# predict a new flower
new = sc.transform([[5.1, 3.5, 1.4, 0.2]])
print("New sample [5.1, 3.5, 1.4, 0.2] predicted as:", iris.target_names[knn.predict(new)[0]])
