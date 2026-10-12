# Assignment 6: Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---- Part A: simple linear regression from scratch ----
# hours studied vs marks
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
marks = np.array([35, 40, 50, 55, 60, 66, 72, 78, 85, 90])

x_mean, y_mean = hours.mean(), marks.mean()
m = np.sum((hours - x_mean) * (marks - y_mean)) / np.sum((hours - x_mean) ** 2)
c = y_mean - m * x_mean
print(f"From scratch: slope m = {m:.3f}, intercept c = {c:.3f}")
print(f"Predicted marks for 7.5 hours: {m * 7.5 + c:.2f}")

plt.figure(figsize=(6, 4))
plt.scatter(hours, marks, label='Actual')
plt.plot(hours, m * hours + c, color='red', label='Fitted line')
plt.xlabel("Hours studied"); plt.ylabel("Marks")
plt.title("Simple Linear Regression"); plt.legend()
plt.tight_layout()
plt.savefig("a6_simple.png", dpi=110)

# ---- Part B: multiple linear regression using sklearn ----
data = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(Xtr, ytr)
pred = lr.predict(Xte)

print("\nMultiple Linear Regression (diabetes dataset)")
print("Intercept:", round(lr.intercept_, 3))
for name, coef in zip(data.feature_names, lr.coef_):
    print(f"  {name:4s}: {coef:9.3f}")
print("MSE :", round(mean_squared_error(yte, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(yte, pred)), 3))
print("R2  :", round(r2_score(yte, pred), 4))

plt.figure(figsize=(6, 4))
plt.scatter(yte, pred, s=15)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--')
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("a6_pred.png", dpi=110)
