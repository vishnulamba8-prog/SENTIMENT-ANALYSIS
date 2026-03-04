# Perform Sentiment Analysis on Twitter data using NLP

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columns=['target','id','date','flag','user','text']
df=pd.read_csv(r"C:\Users\91783\Downloads\CODTECH IT SOLUTIONS - INTERNSHIP\TASK 4\training.1600000.processed.noemoticon.csv",encoding="latin-1",names=columns)

# Keep only required columns
df=df[['target','text']]

# REDUCE DATA SIZE (FOR FASTER EXECUTION)
# As original dataset has 1.6 million rows.
# We are going to sample 10,000 rows only for faster training.

df=df.sample(n=10000,random_state=42)

# CONVERT NUMERIC LABELS TO TEXT LABELS
# 0 -> Negative and 4 -> Positive

df['target']=df['target'].replace(0,'negative')
df['target']=df['target'].replace(4,'positive')

# VISUALIZE SENTIMENT DISTRIBUTION
plt.figure(figsize=(5,4))
sns.countplot(x='target',data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# CLEAN TEXT DATA (NLP PREPROCESSING)
# Steps:
# 1. Remove URLs
# 2. Remove @mentions
# 3. Remove special characters
# 4. Convert to lowercase

def clean_text(text):
    text=re.sub(r'http\S+','',text)        # Remove URLs
    text=re.sub(r'@\w+','',text)           # Remove mentions
    text=re.sub(r'[^a-zA-Z\s]','',text)    # Remove special characters
    text=text.lower()                      # Convert to lowercase
    return text
df['clean_text']=df['text'].apply(clean_text)

# CONVERT TEXT TO NUMERICAL FORMAT (TF-IDF) as the Machine learning models require numerical input.
vectorizer=TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(df['clean_text'])
y=df['target']

# TRAIN-TEST SPLIT (80% TRAIN, 20% TEST)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# BUILD AND TRAIN LOGISTIC REGRESSION MODEL
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# MAKE PREDICTIONS
y_pred=model.predict(X_test)

# MODEL EVALUATION
accuracy=accuracy_score(y_test,y_pred)
print("MODEL PERFORMANCE RESULTS")
print("Accuracy: {:.2f}%".format(accuracy*100))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX VISUALIZATION
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Negative', 'Positive'],yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()


# CONCLUSION:
# The Logistic Regression model successfully classifies tweets into positive and negative sentiments.
# TF-IDF feature extraction helps convert text into meaningful numerical features for prediction.

