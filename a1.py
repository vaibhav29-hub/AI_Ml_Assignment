# Assignment 1: Association between dependent and independent variables
import pandas as pd
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 12)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_diabetes

# load diabetes dataset (target = disease progression after one year)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Shape of data:", df.shape)
print(df.head().round(3))

# Pearson correlation of every independent variable with target
print("\nPearson correlation with target:")
corr_target = df.corr()['target'].drop('target').sort_values(ascending=False)
print(corr_target.round(3))

# significance test (p-value) for each feature
print("\nFeature   r       p-value")
for col in data.feature_names:
    r, p = stats.pearsonr(df[col], df['target'])
    print(f"{col:6s}  {r:6.3f}  {p:.2e}")

# Spearman correlation for strongest feature
rho, p = stats.spearmanr(df['bmi'], df['target'])
print(f"\nSpearman correlation (bmi vs target): {rho:.3f}, p = {p:.2e}")

# heatmap of correlation matrix
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("a1_heatmap.png", dpi=110)

# scatter plot of bmi vs target with regression line
plt.figure(figsize=(6, 4))
sns.regplot(x='bmi', y='target', data=df, line_kws={'color': 'red'})
plt.title("BMI vs Disease Progression")
plt.tight_layout()
plt.savefig("a1_scatter.png", dpi=110)
