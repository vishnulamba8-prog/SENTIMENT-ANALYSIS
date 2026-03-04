# SENTIMENT-ANALYSIS

COMPANY - CODTECH IT SOLUTIONS PRIVATE LIMITED

NAME - VISHNU LAMBA

INTERN ID - CTIS2986

DOMAIN - DATA ANALYTICS

DURATION - 8 WEEKS

MENTOR - NEELA SANTOSH

DESCRIPTION OF THE TASK 4

## Project Overview
This project focuses on performing Sentiment Analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. The main objective of this task was to build a predictive model capable of classifying tweets into positive and negative sentiments based on textual content.
The dataset used in this project is the Sentiment140, which contains 1.6 million labeled tweets. Each tweet is classified as either positive (4) or negative (0). 
For efficient processing and faster execution, a sample of 10,000 tweets was used for model training and evaluation.
This project demonstrates a complete end-to-end Machine Learning workflow, including data cleaning, feature extraction, model training, evaluation, and visualization.

 ## Objective of the Task
The key objectives of this project were:
1. To preprocess raw Twitter text data
2. To convert textual data into numerical features
3. To train a classification model for sentiment prediction
4. To evaluate the performance of the model
5. To visualize results using graphical representations

## Tools & Technologies Used
- Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
These libraries were used for data manipulation, visualization, feature extraction and model building.

## Dataset Description
The dataset contains the following columns:
- Target (0 = Negative, 4 = Positive)
- Tweet ID
- Date
- Flag
- Username
- Tweet Text
  
For this project, only two columns were used:
Target and Tweet Text.

The numeric labels were converted into readable labels ("negative" and "positive") for better clarity.

## Data Preprocessing Steps
Before training the model, the tweet text was cleaned using the following steps:
- Removal of URLs
- Removal of Twitter mentions (@username)
- Removal of special characters and punctuation
- Conversion of text into lowercase

Text preprocessing is a crucial step in NLP because Machine Learning models cannot directly understand raw textual data.

## Feature Engineering
To convert text into numerical format, TF-IDF (Term Frequency – Inverse Document Frequency) vectorization was used.
TF-IDF helps in:
- Identifying important words in each tweet
- Reducing the weight of commonly occurring words
- Converting unstructured text into structured numerical data

A maximum of 5000 features was used to balance performance and computational efficiency.

## Model Building
A Logistic Regression classifier was used to perform binary sentiment classification.
The dataset was split as:
- 80% Training Data
- 20% Testing Data
The model was trained on the training dataset and tested on unseen data to evaluate its predictive performance.
Logistic Regression is widely used for binary classification tasks and performs efficiently when combined with TF-IDF features.

## Model Evaluation
The model was evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
The confusion matrix was visualized using a heatmap to clearly show correct and incorrect predictions.
The model achieved strong classification performance, proving that traditional Machine Learning techniques are effective for sentiment analysis tasks.

## Visualizations Included
1️. Sentiment Distribution Chart
- Displays the number of positive and negative tweets in the dataset

2️. Confusion Matrix Heatmap
- Shows model prediction performance visually

These visualizations enhance the interpretability and presentation quality of the project.

## Conclusion
This project successfully implements a complete Sentiment Analysis pipeline using Machine Learning techniques. 
From data cleaning and feature extraction to model training and evaluation, all essential steps of a real-world NLP project were covered.
The results demonstrate that Logistic Regression combined with TF-IDF is an effective approach for text classification problems.

Through this task, I strengthened my understanding of:
- Natural Language Processing fundamentals
- Text preprocessing techniques
- Feature engineering methods
- Machine Learning model evaluation
- Practical implementation of sentiment analysis in Python

This project serves as a strong foundation for exploring advanced NLP techniques such as deep learning-based sentiment analysis models in the future.

## OUTPUT
<img width="618" height="582" alt="image" src="https://github.com/user-attachments/assets/cc1d5b27-2b03-40ec-a9bb-b66e0d832748" />

<img width="567" height="537" alt="image" src="https://github.com/user-attachments/assets/259e548c-0071-4a01-999d-c23f1c9cb529" />

<img width="564" height="267" alt="image" src="https://github.com/user-attachments/assets/936f2781-a498-4482-b592-f5a9d3e44c84" /> 

