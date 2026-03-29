import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

# === 1. 读取训练数据 ===
train_df = train_df = train_df = pd.read_csv("data_processed/core_sample_for_labeling.csv", encoding="cp1252")

train_df = train_df.dropna(subset=["label"])
train_df["label"] = train_df["label"].astype(int)

texts = (train_df["title"].fillna("") + " " + train_df["abstract"].fillna("")).tolist()
labels = train_df["label"].values

# === 2. 向量化 ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# === 3. 训练分类器 ===
clf = LogisticRegression(max_iter=2000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(clf, embeddings, labels, cv=cv, scoring="accuracy")
f1 = cross_val_score(clf, embeddings, labels, cv=cv, scoring="f1")

print("CV Accuracy:", accuracy.mean())
print("CV F1:", f1.mean())

clf.fit(embeddings, labels)

# === 4. 对169核心子库打分 ===
core_df = pd.read_csv("data_processed/consumer_ugly_core_subset.csv")
core_texts = (core_df["title"].fillna("") + " " + core_df["abstract"].fillna("")).tolist()

core_embeddings = model.encode(core_texts, show_progress_bar=True)
probs = clf.predict_proba(core_embeddings)[:, 1]

core_df["relevance_score"] = probs

core_df = core_df.sort_values("relevance_score", ascending=False)

core_df.to_csv("data_processed/core_ranked_by_model.csv",
               index=False,
               encoding="utf-8-sig")

print("Saved ranked file.")