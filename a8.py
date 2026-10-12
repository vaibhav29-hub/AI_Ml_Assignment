# Assignment 8: Regularization to avoid overfitting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# small noisy dataset -> easy to overfit
np.random.seed(1)
X = np.sort(np.random.rand(40, 1) * 6, axis=0)
y = np.sin(X).ravel() + np.random.randn(40) * 0.3
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=3)

degree = 15   # very high degree polynomial
models = {
    "No regularization": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=0.01),
    "Lasso (L1)": Lasso(alpha=0.01, max_iter=50000),
}

xs = np.linspace(0, 6, 300).reshape(-1, 1)
plt.figure(figsize=(8, 4.5))
plt.scatter(Xtr, ytr, c='blue', s=20, label='train')
plt.scatter(Xte, yte, c='orange', s=20, label='test')

print(f"Polynomial degree = {degree}\n")
print(f"{'Model':20s} {'Train MSE':>10s} {'Test MSE':>10s} {'Max |coef|':>12s}")
for name, reg in models.items():
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), reg)
    model.fit(Xtr, ytr)
    tr = mean_squared_error(ytr, model.predict(Xtr))
    te = mean_squared_error(yte, model.predict(Xte))
    coef = np.abs(model[-1].coef_).max()
    print(f"{name:20s} {tr:10.4f} {te:10.4f} {coef:12.2f}")
    if name == "Lasso (L1)":
        print("   Lasso coefficients set to zero:", int(np.sum(model[-1].coef_ == 0)), "of", len(model[-1].coef_))
    plt.plot(xs, model.predict(xs), label=name)

plt.ylim(-2, 2)
plt.legend(); plt.title("Effect of Regularization (degree 15)")
plt.tight_layout()
plt.savefig("a8_reg.png", dpi=110)

# effect of alpha in Ridge
print("\nRidge: effect of alpha")
for a in [0.0001, 0.01, 1, 10, 100]:
    m = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
    print(f"alpha={a:<8} test MSE = {mean_squared_error(yte, m.predict(Xte)):.4f}")
