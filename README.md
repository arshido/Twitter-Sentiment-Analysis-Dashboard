# Twitter-Sentiment-Analysis-Dashboard
# 🐦 Twitter Sentiment Analysis Dashboard (1.6 Million Tweets)

A robust Natural Language Processing (NLP) and Machine Learning pipeline built to analyze and classify the emotional sentiment of Twitter data. This project leverages the **Sentiment140 dataset** to train a highly efficient classifier capable of processing massive text dimensions in real-time.

---

## 🚀 Project Overview
This project extracts, cleans, and structures 1.6 million raw tweets to predict public opinion. While the underlying training dataset provides binary labels, this pipeline implements an advanced **Confidence Threshold Method** to dynamically infer a **Neutral** classification without expanding memory overhead or requiring external dataset downloads.

### Key Metrics Achieved:
* **Dataset Volume:** 1,600,000 processed tweets
* **Vocabulary Features:** 100,000 optimized n-grams (unigrams & bigrams)
* **Model Accuracy:** 79.77% on hidden test data
* **Classification Balance:** Equal precision/recall performance across sentiment classes

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python
* **Environment:** Google Colab / Jupyter Notebooks
* **Data Core:** Pandas, NumPy
* **NLP Processing:** NLTK (Natural Language Toolkit)
* **Machine Learning:** Scikit-Learn
* **Interactive Visualization:** Plotly, Seaborn, Matplotlib

---

## 📦 System Architecture & Pipeline

### 1. Data Ingestion & Wrangling
* Automated dataset downloads directly via the **Kaggle API CLI**.
* Handled raw Twitter strings using `ISO-8859-1` encoding to safe-guard against special character parsing crashes.
* Streamlined system memory by dropping irrelevant metadata (IDs, dates, user handles) to prioritize the core text and target columns.

### 2. Natural Language Preprocessing
Every tweet passes through an aggressive regex scrubbing function that performs:
* Total case normalization (lowercasing).
* Striping of all website URLs (`https?://...`).
* Elimination of user mentions (`@username`) and special characters.
* Removal of English stopwords using NLTK corpora to focus purely on high-signal emotional tokens.

### 3. Feature Engineering (Vectorization)
* Split the dataset into a balanced **80% Training / 20% Testing** ratio using stratified sampling.
* Converted raw strings into numerical matrices via **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization.
* Implemented a structured **100,000 max_feature cap** to actively protect against model overfitting while optimizing matrix operations and Colab RAM utilization.

### 4. Model Architecture & Optimization
* Trained a **Logistic Regression** classifier optimized with an accelerated multi-core configuration (`n_jobs=-1`).
* **The Neutral Detection Strategy:** Because the dataset natively lacks neutral tags, the deployment script utilizes `predict_proba()`. If the absolute probability gap between a positive and negative prediction falls below a specific threshold ($15\%$), the engine intercepts the execution path and flags the tweet as **NEUTRAL 😐**.

---

## 📊 Evaluation Results

The model displays exceptional stability, showing a tight, reliable distribution between both primary classes:

* **Overall Accuracy Score:** 79.77%

```text
Detailed Classification Report:
              precision    recall  f1-score   support

    Negative       0.81      0.78      0.79    160000
    Positive       0.79      0.82      0.80    160000

    accuracy                           0.80    320000
