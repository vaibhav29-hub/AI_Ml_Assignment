# Assignment 7: SVM for Regression (SVR)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---- Part A: SVR on noisy sine data with different kernels ----
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(80)

models = {
    "RBF": SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.1),
    "Linear": SVR(kernel='linear', C=100, epsilon=0.1),
    "Poly": SVR(kernel='poly', C=100, degree=3, epsilon=0.1, coef0=1),
}

plt.figure(figsize=(7, 4))
plt.scatter(X, y, color='gray', s=15, label='data')
for name, model in models.items():
    model.fit(X, y)
    p = model.predict(X)
    print(f"{name:6s} kernel -> R2 = {r2_score(y, p):.4f}, support vectors = {len(model.support_)}")
    plt.plot(X, p, label=name)
plt.legend(); plt.title("SVR with different kernels")
plt.tight_layout()
plt.savefig("a7_kernels.png", dpi=110)

# ---- Part B: SVR on diabetes dataset ----
data = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
sx, sy = StandardScaler(), StandardScaler()
Xtr_s = sx.fit_transform(Xtr); Xte_s = sx.transform(Xte)
ytr_s = sy.fit_transform(ytr.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf', C=1.0, epsilon=0.2)
svr.fit(Xtr_s, ytr_s)
pred = sy.inverse_transform(svr.predict(Xte_s).reshape(-1, 1)).ravel()

print("\nSVR on diabetes dataset")
print("RMSE:", round(np.sqrt(mean_squared_error(yte, pred)), 3))
print("R2  :", round(r2_score(yte, pred), 4))
