print("Program started")

import pandas as pd
from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import joblib
import os

# load dataset
df = pd.read_csv("dataset/spam.csv", encoding='latin-1')

print("Dataset loaded")

# keep only required columns
df = df[['v1', 'v2']]

# rename columns
df.columns = ['label', 'message']

# convert labels to numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# clean text
df['message'] = df['message'].apply(clean_text)

print("Preprocessing complete")

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

print("Dataset split complete")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF complete")

# Logistic Regression
print("Training Logistic Regression")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_tfidf, y_train)

lr_predictions = lr_model.predict(X_test_tfidf)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Logistic Regression Accuracy:", lr_accuracy)

# SVM
print("Training SVM")

svm_model = SVC(
    kernel='linear',
    probability=True
)

svm_model.fit(X_train_tfidf, y_train)

svm_predictions = svm_model.predict(X_test_tfidf)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM Accuracy:", svm_accuracy)

# create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# save model
joblib.dump(svm_model, "models/spam_model.pkl")

# save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully!")