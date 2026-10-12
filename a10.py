# Assignment 10: Fake news detection from multimodal data (text + image)
import os, io, random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack, csr_matrix

random.seed(7); np.random.seed(7)

# ------------------ Step 1: create a multimodal dataset ------------------
# each news post = headline text + image. label 1 = fake, 0 = real
real_tpl = ["{org} announces new {thing} for {group}", "{org} releases report on {thing}",
            "{city} to get new {thing} by next year", "{org} meeting held on {thing} today",
            "{group} welcome decision on {thing}", "Officials confirm {thing} work in {city}"]
fake_tpl = ["SHOCKING! {org} hiding truth about {thing}", "You won't believe what {group} did with {thing}",
            "Viral: {thing} cures all diseases say {group}", "BREAKING!!! {city} {thing} scam exposed, share now",
            "Secret {thing} plan leaked, {org} in panic", "Doctors hate this {thing} trick from {city}"]
neutral_tpl = ["Photo of {thing} in {city} goes viral", "{group} react to {thing} news",
               "New video shows {thing} at {city}"]   # used by both classes (makes text alone confusing)
words = dict(org=["Government", "Ministry", "Police", "University", "Election Commission", "WHO"],
             thing=["vaccine", "metro", "water supply", "exam", "budget", "road", "hospital", "election"],
             group=["students", "farmers", "doctors", "citizens", "traders"],
             city=["Pune", "Mumbai", "Delhi", "Nagpur", "Nashik"])

def make_text(label):
    if random.random() < 0.30:
        t = random.choice(neutral_tpl)
    else:
        t = random.choice(fake_tpl if label == 1 else real_tpl)
    return t.format(**{k: random.choice(v) for k, v in words.items()})

# images: real frames taken from CCTV video, fake ones are edited (spliced / over enhanced)
cap = cv2.VideoCapture("vtest.avi")
frames = []
for i in range(0, 790, 4):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    if ok:
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((256, 192)))

def to_jpeg(img, q=90):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def manipulate(img):
    img = img.copy()
    other = random.choice(frames)
    # splicing: paste a patch from another photo (saved at different quality)
    x, y, w, h = random.randint(0, 150), random.randint(0, 100), random.randint(50, 100), random.randint(40, 90)
    patch = to_jpeg(other.crop((x, y, x + w, y + h)), q=random.randint(40, 70))
    img.paste(patch, (random.randint(0, 150), random.randint(0, 100)))
    if random.random() < 0.6:   # over saturate / sharpen like sensational posts
        img = ImageEnhance.Color(img).enhance(random.uniform(1.5, 2.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(1.2, 1.6))
    return to_jpeg(img, 90)

os.makedirs("news_images", exist_ok=True)
rows = []
for i in range(400):
    label = i % 2
    base = to_jpeg(random.choice(frames), 90)
    if label == 1:
        img = manipulate(base) if random.random() < 0.75 else base    # some fake posts use genuine photos
    else:
        img = manipulate(base) if random.random() < 0.10 else base    # some real posts have edited photos
    path = f"news_images/post_{i}.jpg"
    img.save(path, "JPEG", quality=90)
    rows.append([make_text(label), path, label])

df = pd.DataFrame(rows, columns=["text", "image", "label"])
df.to_csv("multimodal_news.csv", index=False)
print("Dataset size:", df.shape, "| fake:", df.label.sum(), "real:", (df.label == 0).sum())
print(df[["text", "label"]].head().to_string())

# ------------------ Step 2: image features ------------------
def ela_image(path, q=90):
    # Error Level Analysis: resave at known quality and see difference
    org = Image.open(path).convert("RGB")
    diff = ImageChops.difference(org, to_jpeg(org, q))
    return np.asarray(diff).astype(float)

def image_features(path):
    ela = ela_image(path).mean(axis=2)
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blocks = [ela[r:r + 32, c:c + 32].mean() for r in range(0, 192, 32) for c in range(0, 256, 32)]
    return [ela.mean(), ela.std(), ela.max(), np.std(blocks), max(blocks) - min(blocks),
            hsv[:, :, 1].mean(), hsv[:, :, 1].std(), gray.std(),
            cv2.Laplacian(gray, cv2.CV_64F).var()]

img_feat = np.array([image_features(p) for p in df.image])
print("\nImage feature matrix shape:", img_feat.shape)

# ------------------ Step 3: train / test split ------------------
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=42, stratify=df.label)
y_tr, y_te = df.label.values[idx_tr], df.label.values[idx_te]

# text model (TF-IDF + logistic regression)
tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
Xt_tr = tfidf.fit_transform(df.text.values[idx_tr])
Xt_te = tfidf.transform(df.text.values[idx_te])
text_clf = LogisticRegression(max_iter=1000).fit(Xt_tr, y_tr)
p_text = text_clf.predict_proba(Xt_te)[:, 1]

# image model
sc = StandardScaler()
Xi_tr = sc.fit_transform(img_feat[idx_tr]); Xi_te = sc.transform(img_feat[idx_te])
img_clf = LogisticRegression(max_iter=1000).fit(Xi_tr, y_tr)
p_img = img_clf.predict_proba(Xi_te)[:, 1]

# early fusion: join text and image features into one vector
X_tr = hstack([Xt_tr, csr_matrix(Xi_tr)]); X_te = hstack([Xt_te, csr_matrix(Xi_te)])
early_clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
p_early = early_clf.predict_proba(X_te)[:, 1]

# late fusion: average the probabilities of the two models
p_late = (p_text + p_img) / 2

results = {}
print("\nModel                     Accuracy")
for name, p in [("Text only", p_text), ("Image only", p_img),
                ("Early fusion (multimodal)", p_early), ("Late fusion (multimodal)", p_late)]:
    results[name] = accuracy_score(y_te, (p >= 0.5).astype(int))
    print(f"{name:26s} {results[name]:.4f}")

best = max(results, key=results.get)
p_best = {"Text only": p_text, "Image only": p_img, "Early fusion (multimodal)": p_early,
          "Late fusion (multimodal)": p_late}[best]
pred = (p_best >= 0.5).astype(int)
print("\nBest model:", best)
print(classification_report(y_te, pred, target_names=["real", "fake"]))
cm = confusion_matrix(y_te, pred)
print("Confusion Matrix:\n", cm)

# ------------------ Step 4: plots ------------------
plt.figure(figsize=(7, 4))
plt.bar(range(4), list(results.values()), color=['gray', 'gray', 'green', 'green'])
plt.xticks(range(4), ["Text", "Image", "Early\nfusion", "Late\nfusion"])
plt.ylim(0.5, 1.0); plt.ylabel("Accuracy"); plt.title("Unimodal vs Multimodal")
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
plt.tight_layout(); plt.savefig("a10_compare.png", dpi=110)

ConfusionMatrixDisplay(cm, display_labels=["real", "fake"]).plot(cmap='Blues')
plt.title(best); plt.tight_layout(); plt.savefig("a10_cm.png", dpi=110)

# show one real and one manipulated image with their ELA
r_path = df[(df.label == 0)].image.iloc[0]
f_idx = [p for p, l in zip(df.image, df.label) if l == 1 and image_features(p)[3] > np.median(img_feat[:, 3])][0]
fig, ax = plt.subplots(2, 2, figsize=(8, 6))
for row, (p, t) in enumerate([(r_path, "Real post"), (f_idx, "Fake post (edited)")]):
    ax[row, 0].imshow(Image.open(p)); ax[row, 0].set_title(t)
    ax[row, 1].imshow(np.clip(ela_image(p) * 15, 0, 255).astype(np.uint8)); ax[row, 1].set_title("ELA of " + t)
    ax[row, 0].axis('off'); ax[row, 1].axis('off')
plt.tight_layout(); plt.savefig("a10_ela.png", dpi=110)

# ------------------ Step 5: test on a new post ------------------
new_text = "SHOCKING! Police hiding truth about metro, share now"
new_img = df.image.iloc[idx_te[0]]
xt = tfidf.transform([new_text]); xi = sc.transform([image_features(new_img)])
prob = early_clf.predict_proba(hstack([xt, csr_matrix(xi)]))[0, 1]
print(f"\nNew post: '{new_text}'")
print(f"Probability of fake = {prob:.3f} -> {'FAKE' if prob >= 0.5 else 'REAL'}")
