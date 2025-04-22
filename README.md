# 📰 News Article Classification using Metadata

This project demonstrates how to classify news articles into categories such as **Sports**, **Tech**, **Business**, etc., using only **article metadata** (like word count, presence of keywords, and estimated read time) — without relying on the article's full text.

## 📁 Dataset

The dataset contains the following columns:

- `word_count`: Total number of words in the article  
- `has_keywords`: Whether the article has associated keywords (0 or 1)  
- `read_time`: Estimated time to read the article (in minutes)  
- `category`: The actual category of the article (Target Label)

## 💡 Problem Statement

The goal is to train a machine learning classifier that can predict the category of a news article using structured metadata features.

## 🧠 Methodology

1. Load and inspect the data.
2. Clean the dataset and handle missing values.
3. Encode categorical target labels.
4. Use a Random Forest classifier for its performance with tabular data.
5. Evaluate the model using classification metrics.

## 🧾 Code Summary

```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
