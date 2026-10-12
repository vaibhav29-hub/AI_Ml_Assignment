# Assignment 4: Naive Bayes classification
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---- Part A: Gaussian Naive Bayes on iris ----
iris = load_iris()
Xtr, Xte, ytr, yte = train_test_split(iris.data, iris.target, test_size=0.3, random_state=1)

gnb = GaussianNB()
gnb.fit(Xtr, ytr)
pred = gnb.predict(Xte)

print("Gaussian NB Accuracy:", round(accuracy_score(yte, pred), 4))
print("\nClassification Report:")
print(classification_report(yte, pred, target_names=iris.target_names))
cm = confusion_matrix(yte, pred)
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title("Gaussian NB - Confusion Matrix")
plt.tight_layout()
plt.savefig("a4_cm.png", dpi=110)

# ---- Part B: Multinomial Naive Bayes for spam detection ----
messages = ["win money now", "free lottery prize claim", "call to claim free reward",
            "cheap loan offer win", "meeting at 10 tomorrow", "please send the notes",
            "lunch with team today", "project report submitted", "free entry win cash",
            "see you at college"]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]   # 1 = spam, 0 = ham

cv = CountVectorizer()
Xtext = cv.fit_transform(messages)
mnb = MultinomialNB().fit(Xtext, labels)

test_msgs = ["claim your free cash prize", "send report before meeting"]
result = mnb.predict(cv.transform(test_msgs))
print("\nMultinomial NB (spam filter):")
for m, r in zip(test_msgs, result):
    print(f"'{m}' -> {'SPAM' if r == 1 else 'HAM'}")
